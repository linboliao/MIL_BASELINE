#!/usr/bin/env python3
"""Registration-controlled matched-feature analysis for SerialPanelA.

This script does NOT re-extract PFM features. It reuses:
  1) existing contour-filtered patch coordinates,
  2) existing patch-level .pt embeddings for 8 PFMs,
  3) existing mask JPGs only for provenance/QC context.

Pipeline per case:
  - construct a low-resolution tissue occupancy mask directly from patch coordinates;
  - choose a target/medoid slide using rotation/scale/translation-invariant Hu moments;
  - affine-register each source occupancy mask to the target (coarse PCA/centroid init + ECC);
  - map existing patch centers to target coordinates;
  - perform mutual-nearest matching with a grid-aware distance threshold;
  - keep target locations matched across all six centers;
  - index the existing PFM tensors at matched rows and compute same-location vs
    different-location feature distances and leave-one-case-out center-prototype accuracy.

All outputs are written to a new output directory; source data/features are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from itertools import combinations
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import balanced_accuracy_score

REPO = Path("/NAS2/Data1/lbliao/Code-195/MIL_BASELINE")
FEAT_ROOT = Path("/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis/SerialPanelA")
PATCH_DIR = FEAT_ROOT / "patches_0_224/patches"
MASK_DIR = FEAT_ROOT / "patches_0_224/masks"
PT_ROOT = FEAT_ROOT / "feat_0_224/pt_files"
FH5_ROOT = FEAT_ROOT / "feat_0_224/h5_files"
META_CSV = REPO / "datasets/ProstateDiagnosis/serial_sections/panel_A_6center.csv"
PAIRED_CASES = REPO / "datasets/ProstateDiagnosis/serial_sections/paired_cases_all6centers.csv"
MODELS = ["conch", "uni", "uni2", "virchow2", "h-optimus-1", "mstar", "gigapath", "gpfm"]
RNG_SEED = 42


def atomic_new_dir(path: Path):
    path.mkdir(parents=True, exist_ok=False)
    (path / "qc").mkdir()
    (path / "matches").mkdir()


def safe_json(path: Path, obj):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")


def safe_csv(path: Path, rows):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    with open(path, "x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)


def json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def slide_id_from_filename(fn: str) -> str:
    p = Path(fn)
    return p.stem


def load_slide(slide_id: str):
    p = PATCH_DIR / f"{slide_id}.h5"
    with h5py.File(p, "r") as f:
        c = f["coords"][:].astype(np.float64)
        attrs = dict(f["coords"].attrs)
    patch_size = float(attrs["patch_size"])
    level_dim = np.asarray(attrs["level_dim"], dtype=np.float64)
    # level_dim is [W,H] in the current PrePATH output.
    centers = c + patch_size / 2.0
    return {
        "slide_id": slide_id,
        "coords": c,
        "centers": centers,
        "patch_size": patch_size,
        "level_dim": level_dim,
        "n_patches": len(c),
        "patch_h5": str(p),
        "mask_jpg": str(MASK_DIR / f"{slide_id}.jpg"),
    }


def estimate_grid_step(coords: np.ndarray, fallback: float):
    # Coordinates are generated on a regular grid but only tissue-valid sites remain.
    vals = []
    for ax in [0, 1]:
        u = np.unique(coords[:, ax].astype(np.int64))
        d = np.diff(np.sort(u))
        d = d[d > max(2, 0.2 * fallback)]
        if len(d):
            # low quantile is robust to gaps where intermediate tissue sites are absent
            vals.append(float(np.quantile(d, 0.10)))
    if not vals:
        return float(fallback)
    v = float(np.median(vals))
    # protect against irregular coordinate differences smaller than the true lattice step
    if not (0.5 * fallback <= v <= 1.5 * fallback):
        return float(fallback)
    return v


def occupancy_mask(slide, downsample: int, canvas_wh=None):
    W, H = slide["level_dim"]
    if canvas_wh is None:
        cw = int(math.ceil(W / downsample)) + 4
        ch = int(math.ceil(H / downsample)) + 4
    else:
        cw, ch = canvas_wh
    m = np.zeros((ch, cw), np.uint8)
    ps = slide["patch_size"]
    # Fill the exact selected patch support; coordinates are already contour-filtered.
    for x, y in slide["coords"]:
        x0 = max(0, int(math.floor(x / downsample)))
        y0 = max(0, int(math.floor(y / downsample)))
        x1 = min(cw - 1, int(math.ceil((x + ps) / downsample)))
        y1 = min(ch - 1, int(math.ceil((y + ps) / downsample)))
        cv2.rectangle(m, (x0, y0), (x1, y1), 255, thickness=-1)
    # close tiny raster gaps only; do not materially expand tissue support
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return m


def hu_descriptor(slide, ds=128):
    m = occupancy_mask(slide, ds)
    mom = cv2.moments((m > 0).astype(np.uint8), binaryImage=True)
    hu = cv2.HuMoments(mom).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def choose_medoid(slides):
    desc = np.stack([hu_descriptor(s) for s in slides], axis=0)
    # Robustly standardize Hu dimensions so one moment does not dominate.
    med = np.median(desc, axis=0)
    mad = np.median(np.abs(desc - med), axis=0)
    z = (desc - med) / np.maximum(mad, 1e-3)
    D = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    # Small area term discourages selecting a strongly under/over-covered section.
    areas = np.asarray([s["n_patches"] * s["patch_size"] ** 2 for s in slides], float)
    A = np.abs(np.log(areas[:, None] / areas[None, :]))
    score = (D + 0.5 * A).sum(axis=1)
    idx = int(np.argmin(score))
    return idx, score, D


def pca_angle(points):
    p = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(p.T)
    w, v = np.linalg.eigh(cov)
    e = v[:, np.argmax(w)]
    return math.atan2(e[1], e[0]), cov


def affine_from_similarity(src_pts, tgt_pts, angle_delta, scale):
    cs = src_pts.mean(axis=0)
    ct = tgt_pts.mean(axis=0)
    c = math.cos(angle_delta) * scale
    s = math.sin(angle_delta) * scale
    A = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = ct - A @ cs
    return np.column_stack([A, t]).astype(np.float32)


def apply_affine(points, M):
    return points @ M[:, :2].T + M[:, 2]


def dice_iou(a, b):
    aa = a > 0
    bb = b > 0
    inter = np.logical_and(aa, bb).sum()
    sa = aa.sum(); sb = bb.sum()
    dice = 2.0 * inter / max(sa + sb, 1)
    union = np.logical_or(aa, bb).sum()
    iou = inter / max(union, 1)
    return float(dice), float(iou)


def affine_lr_to_level0(M_lr, ds):
    # x_t_lr = A*x_s_lr + t_lr; with x_lr=x0/ds, so t0=ds*t_lr.
    M = M_lr.astype(np.float64).copy()
    M[:, 2] *= ds
    return M


def register_pair(src, tgt, ds=64, ecc_iters=120):
    maxW = max(src["level_dim"][0], tgt["level_dim"][0])
    maxH = max(src["level_dim"][1], tgt["level_dim"][1])
    cw = int(math.ceil(maxW / ds)) + 8
    ch = int(math.ceil(maxH / ds)) + 8
    src_m = occupancy_mask(src, ds, (cw, ch))
    tgt_m = occupancy_mask(tgt, ds, (cw, ch))
    sp = src["centers"] / ds
    tp = tgt["centers"] / ds
    ang_s, cov_s = pca_angle(sp)
    ang_t, cov_t = pca_angle(tp)
    scale0 = math.sqrt(max(np.trace(cov_t), 1e-6) / max(np.trace(cov_s), 1e-6))
    scale0 = float(np.clip(scale0, 0.70, 1.35))

    candidate_angles = [
        ang_t - ang_s,
        ang_t - ang_s + math.pi,
        0.0,
        math.pi / 2,
        math.pi,
        -math.pi / 2,
    ]
    best = None
    for a in candidate_angles:
        for sm in [0.95, 1.0, 1.05]:
            M = affine_from_similarity(sp, tp, a, scale0 * sm)
            warped = cv2.warpAffine(src_m, M, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)
            d, i = dice_iou(tgt_m, warped)
            rec = (d, i, M)
            if best is None or rec[0] > best[0]:
                best = rec
    init_dice, init_iou, M_st_init = best

    # ECC wants a warp convention used with WARP_INVERSE_MAP to align input to template.
    # Therefore initialize with target->source, then invert the result back to source->target.
    M_ts_init = cv2.invertAffineTransform(M_st_init).astype(np.float32)
    tgt_f = cv2.GaussianBlur((tgt_m > 0).astype(np.float32), (0, 0), 3.0)
    src_f = cv2.GaussianBlur((src_m > 0).astype(np.float32), (0, 0), 3.0)
    M_ts = M_ts_init.copy()
    ecc_ok = True
    ecc_cc = float("nan")
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(ecc_iters), 1e-5)
        ecc_cc, M_ts = cv2.findTransformECC(tgt_f, src_f, M_ts, cv2.MOTION_AFFINE, criteria, None, 5)
        M_st_ecc = cv2.invertAffineTransform(M_ts).astype(np.float32)
        warped = cv2.warpAffine(src_m, M_st_ecc, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)
        ecc_dice, ecc_iou = dice_iou(tgt_m, warped)
        A = M_st_ecc[:, :2]
        det = abs(float(np.linalg.det(A)))
        # Reject unstable ECC refinements or refinements that reduce overlap.
        if not np.isfinite(ecc_dice) or ecc_dice + 0.01 < init_dice or not (0.35 <= det <= 2.5):
            ecc_ok = False
    except cv2.error:
        ecc_ok = False

    if ecc_ok:
        M_lr = M_st_ecc
        final_dice, final_iou = ecc_dice, ecc_iou
        method = "coarse+ecc_affine"
    else:
        M_lr = M_st_init
        final_dice, final_iou = init_dice, init_iou
        method = "coarse_similarity"

    warped = cv2.warpAffine(src_m, M_lr, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)
    # QC overlay: target=green, aligned source=magenta; overlap tends toward white.
    overlay = np.zeros((ch, cw, 3), np.uint8)
    overlay[..., 1] = (tgt_m > 0).astype(np.uint8) * 180
    overlay[..., 0] = (warped > 0).astype(np.uint8) * 180
    overlay[..., 2] = (warped > 0).astype(np.uint8) * 180
    cv2.putText(overlay, f"Dice={final_dice:.3f} IoU={final_iou:.3f} {method}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return {
        "M_lr": M_lr.astype(np.float64),
        "M_level0": affine_lr_to_level0(M_lr, ds),
        "init_dice": init_dice,
        "init_iou": init_iou,
        "final_dice": final_dice,
        "final_iou": final_iou,
        "ecc_cc": ecc_cc,
        "method": method,
        "overlay": overlay,
        "canvas_wh": (cw, ch),
    }


def match_existing_patches(src, tgt, M_src_to_tgt, tau_factor=0.75):
    src_t = apply_affine(src["centers"], M_src_to_tgt)
    tgt_p = tgt["centers"]
    tree_s = cKDTree(src_t)
    tree_t = cKDTree(tgt_p)
    d_t2s, j = tree_s.query(tgt_p, k=1)
    d_s2t, i_back = tree_t.query(src_t, k=1)
    mutual = i_back[j] == np.arange(len(tgt_p))
    # Distance is evaluated in target coordinates; account for affine scale of source grid.
    scale_eff = math.sqrt(abs(float(np.linalg.det(M_src_to_tgt[:, :2]))))
    step_t = estimate_grid_step(tgt["coords"], tgt["patch_size"])
    step_s = estimate_grid_step(src["coords"], src["patch_size"]) * scale_eff
    step_ref = 0.5 * (step_t + step_s)
    tau = tau_factor * step_ref
    ok = mutual & (d_t2s <= tau)
    target_idx = np.flatnonzero(ok).astype(np.int64)
    source_idx = j[ok].astype(np.int64)
    dist = d_t2s[ok].astype(np.float64)
    sens = {}
    for f in [0.50, 0.75, 1.00]:
        sens[f"n_match_tau_{f:.2f}"] = int(np.sum(mutual & (d_t2s <= f * step_ref)))
    return {
        "target_idx": target_idx,
        "source_idx": source_idx,
        "distance": dist,
        "step_ref": float(step_ref),
        "tau": float(tau),
        "n_mutual": int(mutual.sum()),
        **sens,
    }


def verify_feature_coords(model, slide_id, patch_coords):
    hp = FH5_ROOT / model / f"{slide_id}.h5"
    with h5py.File(hp, "r") as f:
        c = f["coords"][:]
    if c.shape != patch_coords.shape or not np.array_equal(c, patch_coords.astype(c.dtype)):
        raise RuntimeError(f"feature coords mismatch: {model}/{slide_id}")


def load_feature_rows(model, slide_id, indices, expected_n):
    pt = PT_ROOT / model / f"{slide_id}.pt"
    t = torch.load(str(pt), map_location="cpu", mmap=True, weights_only=True)
    if t.ndim != 2 or t.shape[0] != expected_n:
        raise RuntimeError(f"feature shape mismatch: {model}/{slide_id} {tuple(t.shape)} expected rows {expected_n}")
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return np.empty((0, int(t.shape[1])), np.float32)
    # Sorted access is friendlier to mmap/NAS page reads; restore original order afterward.
    order = np.argsort(idx)
    sorted_idx = idx[order]
    vals = t[torch.from_numpy(sorted_idx)].float().numpy()
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    vals = vals[inv]
    if not np.isfinite(vals).all():
        raise RuntimeError(f"non-finite features: {model}/{slide_id}")
    return vals


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def deterministic_subsample(indices, max_n):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= max_n:
        return indices
    pos = np.linspace(0, len(indices) - 1, max_n, dtype=np.int64)
    return indices[pos]


def feature_metrics(cases_data, models, max_locations_per_case=512):
    model_rows = []
    pair_rows = []
    # Cache metadata needed for prototype evaluation but release one PFM before the next.
    centers_order = sorted({x["center"] for cd in cases_data for x in cd["slides"]})
    for model in models:
        print(f"[metrics] {model}", flush=True)
        same_all = []
        diff_all = []
        case_feature_blocks = []
        case_center_labels = []
        case_group_labels = []
        for cd in cases_data:
            common = cd["common_target_idx"]
            if len(common) < 2:
                continue
            chosen_pos = deterministic_subsample(np.arange(len(common)), max_locations_per_case)
            common_sub = common[chosen_pos]
            target_slide_idx = cd["target_slide_idx"]
            per_center = {}
            # Build mapping from target index to each slide index.
            for si, s in enumerate(cd["slides"]):
                if si == target_slide_idx:
                    idx = common_sub
                else:
                    mp = cd["match_maps"][si]
                    idx = np.asarray([mp[int(t)] for t in common_sub], dtype=np.int64)
                verify_feature_coords(model, s["slide_id"], s["coords"])
                f = load_feature_rows(model, s["slide_id"], idx, s["n_patches"])
                per_center[s["center"]] = l2norm(f)

            # Matched-location and deterministic different-location controls.
            for ca, cb in combinations(centers_order, 2):
                A = per_center[ca]; B = per_center[cb]
                n = len(A)
                if n < 2:
                    continue
                same = 1.0 - np.sum(A * B, axis=1)
                shift = max(1, n // 2)
                Bneg = np.roll(B, shift=shift, axis=0)
                diff = 1.0 - np.sum(A * Bneg, axis=1)
                same_all.append(same); diff_all.append(diff)
                pair_rows.append({
                    "model": model, "case_id": cd["case_id"], "center_a": ca, "center_b": cb,
                    "n_locations": n,
                    "same_location_cosine_mean": float(np.mean(same)),
                    "same_location_cosine_median": float(np.median(same)),
                    "different_location_cosine_mean": float(np.mean(diff)),
                    "same_over_different_ratio": float(np.mean(same) / max(np.mean(diff), 1e-12)),
                })

            # For center prototype LOO: each row remains one matched tissue location from one center.
            Xs=[]; ys=[]
            for c in centers_order:
                Xs.append(per_center[c])
                ys.extend([c] * len(per_center[c]))
            Xcase = np.concatenate(Xs, axis=0)
            case_feature_blocks.append(Xcase)
            case_center_labels.extend(ys)
            case_group_labels.extend([cd["case_id"]] * len(Xcase))

        if not same_all:
            continue
        same_all = np.concatenate(same_all)
        diff_all = np.concatenate(diff_all)
        X = np.concatenate(case_feature_blocks, axis=0)
        y = np.asarray(case_center_labels)
        groups = np.asarray(case_group_labels)
        unique_cases = np.unique(groups)
        if len(unique_cases) >= 2:
            preds = np.empty_like(y, dtype=object)
            for held in unique_cases:
                tr = groups != held; te = groups == held
                prototypes = {}
                for c in centers_order:
                    z = X[tr & (y == c)]
                    p = z.mean(axis=0)
                    p = p / max(np.linalg.norm(p), 1e-12)
                    prototypes[c] = p
                P = np.stack([prototypes[c] for c in centers_order], axis=0)
                sim = X[te] @ P.T
                preds[te] = np.asarray(centers_order, dtype=object)[np.argmax(sim, axis=1)]
            bacc = balanced_accuracy_score(y, preds)
        else:
            bacc = float("nan")
        model_rows.append({
            "model": model,
            "n_cases": int(len(np.unique(groups))),
            "n_center_samples": int(len(y)),
            "n_same_location_pairs": int(len(same_all)),
            "same_location_cosine_mean": float(np.mean(same_all)),
            "same_location_cosine_median": float(np.median(same_all)),
            "different_location_cosine_mean": float(np.mean(diff_all)),
            "different_location_cosine_median": float(np.median(diff_all)),
            "same_over_different_ratio": float(np.mean(same_all) / max(np.mean(diff_all), 1e-12)),
            "center_prototype_loo_bacc": float(bacc),
            "center_chance_bacc": float(1.0 / len(centers_order)),
        })
    return pd.DataFrame(model_rows), pd.DataFrame(pair_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--n_cases", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--reg_downsample", type=int, default=64)
    ap.add_argument("--tau_factor", type=float, default=0.75)
    ap.add_argument("--ecc_iters", type=int, default=120)
    ap.add_argument("--max_locations_per_case", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    atomic_new_dir(out)

    meta = pd.read_csv(META_CSV, dtype=str)
    paired = pd.read_csv(PAIRED_CASES, dtype=str, encoding="utf-8-sig")["case_id"].astype(str).tolist()
    cases = args.cases if args.cases else paired[: args.n_cases]
    models = args.models
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")

    run_meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_head": git_head(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "cases": cases,
        "models": models,
        "reg_downsample": args.reg_downsample,
        "tau_factor": args.tau_factor,
        "ecc_iters": args.ecc_iters,
        "max_locations_per_case": args.max_locations_per_case,
        "feature_reextraction": False,
        "registration_basis": "existing contour-filtered patch occupancy",
        "matching": "mutual nearest existing patch centers after affine registration",
    }
    safe_json(out / "run_meta.json", run_meta)

    pair_qc = []
    case_qc = []
    transform_obj = {}
    cases_data = []

    for case_id in cases:
        rows = meta[meta["case_id"].astype(str) == str(case_id)].copy()
        if len(rows) != 6:
            raise RuntimeError(f"{case_id}: expected 6 Panel A slides, got {len(rows)}")
        rows = rows.sort_values("center")
        slides = []
        for _, r in rows.iterrows():
            sid = slide_id_from_filename(r["filename"])
            s = load_slide(sid)
            s["center"] = r["center"]
            s["label"] = r["label"]
            slides.append(s)
        target_idx, medoid_score, medoid_dist = choose_medoid(slides)
        tgt = slides[target_idx]
        print("[{}] target={} center={} patches={}".format(case_id, tgt["slide_id"], tgt["center"], tgt["n_patches"]), flush=True)
        match_maps = {target_idx: {int(i): int(i) for i in range(tgt["n_patches"])}}
        valid_sets = [set(range(tgt["n_patches"]))]
        transform_obj[case_id] = {
            "target_slide": tgt["slide_id"], "target_center": tgt["center"],
            "medoid_scores": {slides[i]["slide_id"]: float(medoid_score[i]) for i in range(6)},
            "source_to_target": {},
        }
        case_pair_dice=[]
        for si, src in enumerate(slides):
            if si == target_idx:
                continue
            reg = register_pair(src, tgt, ds=args.reg_downsample, ecc_iters=args.ecc_iters)
            mat = match_existing_patches(src, tgt, reg["M_level0"], tau_factor=args.tau_factor)
            mp = {int(t): int(s) for t, s in zip(mat["target_idx"], mat["source_idx"])}
            match_maps[si] = mp
            valid_sets.append(set(mp.keys()))
            case_pair_dice.append(reg["final_dice"])
            qname = "{}__{}__to__{}.png".format(case_id, src["slide_id"], tgt["slide_id"])
            cv2.imwrite(str(out / "qc" / qname), reg["overlay"])
            transform_obj[case_id]["source_to_target"][src["slide_id"]] = {
                "center": src["center"], "M_level0": reg["M_level0"],
                "final_dice": reg["final_dice"], "final_iou": reg["final_iou"], "method": reg["method"],
            }
            pair_qc.append({
                "case_id": case_id, "target_slide": tgt["slide_id"], "target_center": tgt["center"],
                "source_slide": src["slide_id"], "source_center": src["center"],
                "source_patches": src["n_patches"], "target_patches": tgt["n_patches"],
                "init_dice": reg["init_dice"], "final_dice": reg["final_dice"], "final_iou": reg["final_iou"],
                "method": reg["method"], "ecc_cc": reg["ecc_cc"],
                "grid_step_ref": mat["step_ref"], "match_tau_px": mat["tau"], "n_mutual": mat["n_mutual"],
                "n_match_tau_0.50": mat["n_match_tau_0.50"], "n_match_tau_0.75": mat["n_match_tau_0.75"],
                "n_match_tau_1.00": mat["n_match_tau_1.00"],
                "median_match_distance_px": float(np.median(mat["distance"])) if len(mat["distance"]) else np.nan,
                "p90_match_distance_px": float(np.quantile(mat["distance"], 0.90)) if len(mat["distance"]) else np.nan,
            })
        common = np.asarray(sorted(set.intersection(*valid_sets)), dtype=np.int64)
        # Store compact six-way mapping without modifying any source feature/coord file.
        map_cols = {"target_idx": common}
        for si, s in enumerate(slides):
            if si == target_idx:
                map_cols["idx__{}__{}".format(s["center"], s["slide_id"])] = common
            else:
                map_cols["idx__{}__{}".format(s["center"], s["slide_id"])] = np.asarray([match_maps[si][int(t)] for t in common], dtype=np.int64)
        safe_csv(out / "matches" / f"{case_id}.csv", pd.DataFrame(map_cols))
        coverage = len(common) / max(tgt["n_patches"], 1)
        case_qc.append({
            "case_id": case_id, "target_slide": tgt["slide_id"], "target_center": tgt["center"],
            "target_patches": tgt["n_patches"], "common_sixway_locations": len(common),
            "common_sixway_fraction_of_target": coverage,
            "mean_pair_dice": float(np.mean(case_pair_dice)), "min_pair_dice": float(np.min(case_pair_dice)),
            "label": tgt["label"],
        })
        cases_data.append({
            "case_id": case_id, "slides": slides, "target_slide_idx": target_idx,
            "common_target_idx": common, "match_maps": match_maps,
        })
        print(f"[{case_id}] common6={len(common)} ({coverage:.1%}) meanDice={np.mean(case_pair_dice):.3f}", flush=True)

    safe_csv(out / "registration_pair_qc.csv", pair_qc)
    safe_csv(out / "case_qc.csv", case_qc)
    safe_json(out / "transforms.json", transform_obj)

    model_df, pair_df = feature_metrics(cases_data, models, max_locations_per_case=args.max_locations_per_case)
    model_df = model_df.sort_values(["same_over_different_ratio", "center_prototype_loo_bacc"], ascending=[True, True])
    safe_csv(out / "model_metrics.csv", model_df)
    safe_csv(out / "model_center_pair_metrics.csv", pair_df)

    q = pd.DataFrame(case_qc)
    p = pd.DataFrame(pair_qc)
    report = []
    report.append("# Panel A registration-controlled matched-feature analysis\n")
    report.append(f"Cases: {len(cases)}; models: {len(models)}; Git: `{git_head()}`\n")
    report.append("No PFM feature extraction was performed. Existing patch coordinates and existing .pt embeddings were reused.\n")
    report.append("## Registration / matching QC\n")
    report.append(f"- Mean pairwise affine Dice: {p.final_dice.mean():.3f}\n")
    report.append(f"- Median pairwise affine Dice: {p.final_dice.median():.3f}\n")
    report.append(f"- Minimum pairwise affine Dice: {p.final_dice.min():.3f}\n")
    report.append(f"- Median six-way common locations per case: {q.common_sixway_locations.median():.0f}\n")
    report.append(f"- Median six-way common coverage of target grid: {q.common_sixway_fraction_of_target.median():.1%}\n")
    report.append("\n### Case QC\n\n")
    report.append(q.to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\n## 8-PFM matched-feature metrics\n\n")
    report.append(model_df.to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\nInterpretation: lower same-location cosine distance and lower same/different-location ratio indicate stronger content consistency across centers. Lower leave-one-case-out center prototype balanced accuracy indicates less center/domain information retained.\n")
    with open(out / "REPORT.md", "x", encoding="utf-8") as f:
        f.write("".join(report))

    print("DONE", out, flush=True)
    print(model_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
