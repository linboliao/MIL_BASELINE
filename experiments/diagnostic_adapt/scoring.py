"""WSI scoring, foreground sampling and patient aggregation."""
import math
import sys

from .artifacts import put, require, sha


def batch_forward(net, bags, device):
    import torch
    require(bool(bags) and all(len(x) > 0 for x in bags), 'empty bag')
    n = max(len(x) for x in bags)
    x = torch.zeros((len(bags), n, 1536), dtype=torch.float32, device=device)
    mask = torch.zeros((len(bags), n), dtype=torch.bool, device=device)
    for i, b in enumerate(bags):
        x[i, :len(b)] = b.to(device=device, dtype=torch.float32)
        mask[i, :len(b)] = True
    h = net.feature(x)
    a = net.attention(h).squeeze(-1).masked_fill(~mask, -torch.inf)
    a = torch.softmax(a, dim=1)
    pooled = torch.sum(h * a.unsqueeze(-1), dim=1)
    logits = net.classifier(pooled)
    return logits[:, 1] - logits[:, 0]


def foreground_pool(thumb, coords, ps, dims):
    import numpy as np
    from PIL import Image, ImageFilter
    hsv = np.asarray(thumb.convert('RGB').convert('HSV'))
    binary = ((hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 245))
    mask = np.asarray(Image.fromarray((binary * 255).astype('uint8')).filter(ImageFilter.MaxFilter(3))) > 0
    tw, th = thumb.size
    xx = np.clip(((coords[:, 0] + ps / 2) / dims[0] * tw).astype(int), 0, tw - 1)
    yy = np.clip(((coords[:, 1] + ps / 2) / dims[1] * th).astype(int), 0, th - 1)
    ids = np.flatnonzero(mask[yy, xx])
    return ids, dict(thumbnail_width=tw, thumbnail_height=th, thumbnail_mask_fraction=float(mask.mean()),
                    thumbnail_S_median=float(np.median(hsv[:, :, 1])), thumbnail_V_median=float(np.median(hsv[:, :, 2])))


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if a | b else None


def rank_auc(scores, labels):
    positive = [s for s, y in zip(scores, labels) if y == 1]
    negative = [s for s, y in zip(scores, labels) if y == 0]
    if not positive or not negative:
        return None
    return sum(float(p > n) + .5 * float(p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def aggregate(rows):
    groups = {}
    for r in rows:
        p = r['patient_group_id']
        s = groups.setdefault(p, dict(original=-math.inf, foreground=-math.inf, fallback_any=False, WSI=0))
        s['original'] = max(s['original'], r['original_logit'])
        s['foreground'] = max(s['foreground'], r['foreground_logit'])
        s['fallback_any'] = s['fallback_any'] or r['foreground_fallback']
        s['WSI'] += 1
    return groups


def score_one(r, include_diagnostics, *, store, policy, net, device, inputs, audit, out_dir, prepath):
    import numpy as np
    import torch
    import openslide
    from PIL import Image
    t, coords, ps, dims = store.load(r)
    seed = policy.stable_seed(42, int(r['io_row_id']), 'mve_eval4096')
    original_bag, original_ids = store.bag(r, 4096, seed)
    net.eval()
    with torch.inference_mode():
        original = float(batch_forward(net, [original_bag], device)[0].cpu())
    raw = inputs.assert_file(r, 'raw_path')
    st = raw.stat()
    audit.record(raw, 'raw_WSI_open', bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                      hash_scope='raw WSI stat only, not whole-image hash')
    if str(raw).lower().endswith('.kfb'):
        sys.path.insert(0, str(prepath))
        from wsi_core.Aslide.kfb.kfb_slide import KfbSlide
        slide = KfbSlide(str(raw))
    else:
        slide = openslide.OpenSlide(str(raw))
    try:
        require(tuple(slide.dimensions) == dims, 'raw WSI / geometry dimensions differ')
        thumb = openslide.AbstractSlide.get_thumbnail(slide, (1024, 1024)).convert('RGB')
    finally:
        slide.close()
    require(thumb.width > 0 and thumb.height > 0 and max(thumb.size) <= 1024, 'invalid full-field thumbnail')
    # One persistent full-field geometry witness per format per fold, for human review.
    fmt = raw.suffix.lower().lstrip('.')
    witness = out_dir / ('fullfield_' + fmt + '.png')
    if not witness.exists():
        with witness.open('xb') as f:
            thumb.save(f, format='PNG')
        put(out_dir / ('fullfield_' + fmt + '_geometry.json'), dict(stem=r['stem'], raw_path=str(raw),
            level0_dimensions=list(dims), thumbnail_dimensions=list(thumb.size),
            invocation='openslide.AbstractSlide.get_thumbnail', thumbnail_sha256=sha(witness),
            validation='full-field API plus exact level0 dimensions; human visual review artifact'))
    eligible, quality = foreground_pool(thumb, coords, ps, dims)
    fallback = len(eligible) < 32
    fg_ids = original_ids if fallback else policy.indices(eligible, 4096, seed)
    if fallback:
        foreground = original
    else:
        with torch.inference_mode():
            foreground = float(batch_forward(net, [t[fg_ids]], device)[0].cpu())
    require(math.isfinite(original) and math.isfinite(foreground), 'nonfinite inference')
    out = {k: r[k] for k in ('patient_group_id', 'stem', 'role')}
    out.update(original_logit=original, foreground_logit=foreground, patches=len(t), eligible_patches=len(eligible),
               foreground_fallback=fallback, original_used=len(original_ids), foreground_used=len(fg_ids))
    # Save actual indices and first-use witnesses; fixed score provenance, no target truth.
    with (out_dir / 'indices' / (r['stem'] + '.npz')).open('xb') as f:
        np.savez_compressed(f, original=original_ids, foreground=fg_ids, eligible=eligible)
    if include_diagnostics:
        shadow_seed = policy.stable_seed(42, int(r['io_row_id']), 'mve_diag_index4096')
        shadow = policy.indices(np.arange(len(t)), 4096, shadow_seed)
        quality.update(foreground_fraction=len(eligible) / len(t),
                       original_sampling_jaccard=jaccard(original_ids, shadow),
                       foreground_sampling_jaccard=None if fallback else jaccard(fg_ids, policy.indices(eligible, 4096, shadow_seed)),
                       physical_input_jaccard=None if fallback else jaccard(original_ids, fg_ids))
        out.update(quality)
    return out
