"""Frozen input validation and fold-scoped feature access."""
import collections
import importlib.util
import io
import hashlib
import time
import json
from pathlib import Path

from .artifacts import require, sha, utc


def load_policy(config, config_sha, datasets, policy_path):
    require(sha(config) == config_sha, 'frozen configuration changed during run')
    c = json.loads(config.read_text())
    policy_verified = False
    for x in c['lineage']:
        original = Path(x['path'])
        path = policy_path if original.name == 'diagnostic_adaptation_mve_20261117_v1.py' else original
        require(sha(path) == x['sha256'], 'frozen dependency changed: ' + str(path))
        policy_verified |= path == policy_path
    require(policy_verified, 'frozen configuration must pin the policy')
    for fold, inputs in c['inputs'].items():
        for kind, item in inputs.items():
            path = datasets / fold
            if kind != 'source':
                path /= 'repeat0'
            path /= Path(item['path']).name
            require(sha(path) == item['sha256'], 'frozen input changed: ' + str(path))
            item['path'] = str(path)
    spec = importlib.util.spec_from_file_location('frozen_mve_policy', policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return c, policy


class ReadAudit:
    def __init__(self, root):
        self.phase = 'initialize'
        self.f = (root / 'read_audit.jsonl').open('x')

    def record(self, path, kind, digest=None, **extra):
        self.f.write(json.dumps(dict(at=utc(), phase=self.phase, kind=kind,
                                     path=str(path), sha256=digest, **extra), ensure_ascii=False) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


class Inputs:
    def __init__(self, config, policy, slug, audit, config_sha):
        self.c, self.policy, self.slug, self.audit = config, policy, slug, audit
        self.config_sha = config_sha
        self.lock_path = None
        self.lock_hash = None
        self.allowed = {}

    def authorize_evaluation(self, path, expected_checkpoint_hash):
        lock = json.loads(path.read_text())
        require(lock['config_sha256'] == self.config_sha and lock['fold'] == self.slug,
                'lock belongs to other config/fold')
        require(lock['checkpoint_sha256'] == expected_checkpoint_hash, 'wrong checkpoint lock')
        require(lock['selection']['action'] in self.policy.PRIORITY, 'invalid selection lock')
        require(lock['complete'] is True and set(lock['actions']) == set(self.policy.PRIORITY), 'incomplete action lock')
        self.lock_path, self.lock_hash = path, sha(path)
        self.audit.record(path, 'action_lock', self.lock_hash)

    def read(self, kind):
        allowed = {'source': ('train', 'calibration'), 'budget_wsi': ('budget',),
                   'budget_labels': ('budget',), 'evaluation_wsi': ('evaluation_inference',)}
        require(self.audit.phase in allowed.get(kind, ()), 'metadata kind forbidden in current phase')
        if kind == 'evaluation_wsi':
            require(self.lock_path is not None and sha(self.lock_path) == self.lock_hash,
                    'evaluation read forbidden before unchanged action lock')
        audit = []
        rows = self.policy.guarded_read(self.c, self.slug, kind, repeat=0, audit=audit)
        for x in audit:
            self.audit.record(x['path'], 'metadata_' + kind, x['sha256'], rows=x['rows'])
        for r in rows:
            if 'pt_path' in r:
                old = self.allowed.setdefault(r['stem'], r)
                require(all(old[k] == r[k] for k in ('pt_path', 'raw_path', 'feature_h5_path', 'patient_group_id')),
                        'ambiguous WSI mapping')
        return rows

    def assert_file(self, row, field):
        r = self.allowed.get(row['stem'])
        require(r is not None and row[field] == r[field], 'file outside current fold allowlist')
        return Path(r[field])


def available_memory_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable; do not guess preload capacity')


class FeatureStore:
    def __init__(self, inputs, policy, cache_bytes=16 * 2**30):
        self.inputs, self.policy = inputs, policy
        self.cache = collections.OrderedDict()
        self.size, self.limit = 0, cache_bytes
        self.witnesses = {}
        self.validated = {}
        self.resident = {}
        self.resident_hashes = {}
        self.resident_active = False

    def preload(self, rows, headroom_bytes=35 * 2**30):
        """One explicit source_design preload; no target or calibration prefetch."""
        require(not self.resident_active and not self.resident, 'preload already active')
        require(rows and all(r['role'] == 'source_design' for r in rows), 'preload only source_design')
        require(len({r['stem'] for r in rows}) == len(rows), 'duplicate preload WSI')
        sizes = [self.inputs.assert_file(r, 'pt_path').stat().st_size for r in rows]
        payload = sum(sizes)
        # Include coords/overhead plus a conservative transient deserialization buffer.
        required = int(payload * 1.05) + 2 * max(sizes) + headroom_bytes
        initial = available_memory_bytes()
        require(initial >= required, f'preload RAM insufficient: required={required}, available={initial}')
        start = time.monotonic()
        held = 0
        low = initial
        self.inputs.audit.record('RAM', 'preload_start', WSI=len(rows), payload_bytes=payload,
                                 required_bytes=required, available_bytes=initial)
        for i, row in enumerate(sorted(rows, key=lambda r: r['stem']), 1):
            available = available_memory_bytes()
            low = min(low, available)
            size = self.inputs.assert_file(row, 'pt_path').stat().st_size
            require(available >= headroom_bytes + 2 * size,
                    f'preload memory headroom lost before {i}/{len(rows)}')
            item = self.load(row)
            self.resident[row['stem']] = item
            self.resident_hashes[row['stem']] = row['feature_sha256']
            held += item[0].nelement() * item[0].element_size() + item[1].nbytes
            if i % 100 == 0 or i == len(rows):
                print('PRELOAD', i, '/', len(rows), 'GiB', round(held/2**30, 2), flush=True)
        require(available_memory_bytes() >= headroom_bytes, 'preload final headroom insufficient')
        self.cache.clear()  # Release only redundant Python references, never disk files.
        self.size = 0
        self.resident_active = True
        summary = dict(WSI=len(rows), resident_bytes=held, serialized_PT_bytes=payload,
                       available_bytes_at_start=initial, minimum_observed_available_bytes=low,
                       headroom_bytes=headroom_bytes, seconds=time.monotonic()-start,
                       dtype='float32', disk_staging=False, training_loop_reads='resident RAM only')
        self.inputs.audit.record('RAM', 'preload_complete', **summary)
        return summary

    def release_preload(self):
        count = len(self.resident)
        self.resident_active = False
        self.resident.clear()
        self.resident_hashes.clear()
        self.cache.clear()
        self.size = 0
        self.inputs.audit.record('RAM', 'preload_released', WSI=count,
                                 meaning='RAM references only; no stored file deletion')

    def load(self, row):
        import numpy as np
        import torch
        import h5py
        key = row['stem']
        if self.resident_active:
            require(key in self.resident, 'training requested WSI outside completed preload')
            require(self.resident_hashes[key] == row['feature_sha256'], 'resident expected checksum changed')
            # Frozen validated snapshot: no NAS read or stat during optimizer updates.
            return self.resident[key]
        paths = {field: self.inputs.assert_file(row, field) for field in
                 ('pt_path', 'feature_h5_path', 'patch_h5_geometry_path')}

        def signatures():
            out = {}
            for field, path in paths.items():
                st = path.stat()
                out[field] = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            return out

        before = signatures()
        prior = self.validated.get(key)
        if prior is not None:
            require(prior['feature_sha256'] == row['feature_sha256'], 'PT expected checksum changed ' + key)
        unchanged = prior is not None and prior['signatures'] == before
        if key in self.cache:
            if unchanged:
                self.cache.move_to_end(key)
                return self.cache[key][0]
            _, old_size = self.cache.pop(key)
            self.size -= old_size
        p = paths['pt_path']
        raw = p.read_bytes()
        # Reloading identical, previously verified files does not repeat full hashes.
        digest = row['feature_sha256'] if unchanged else hashlib.sha256(raw).hexdigest()
        require(digest == row['feature_sha256'], 'PT checksum mismatch ' + key)
        self.inputs.audit.record(p, 'feature_PT', digest,
                                 verification='prior_hash_and_unchanged_stat' if unchanged else 'full_hash')
        t = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        del raw
        if isinstance(t, dict):
            t = t.get('feats', t.get('features'))
        if t is not None and t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        require(t is not None and t.ndim == 2 and t.shape[1] == 1536 and t.dtype == torch.float32,
                'invalid feature shape or dtype ' + key)
        require(len(t) > 0, 'empty feature ' + key)
        if not unchanged:
            require(torch.isfinite(t).all().item(), 'nonfinite feature ' + key)
        hp = paths['feature_h5_path']
        gp = paths['patch_h5_geometry_path']
        with h5py.File(hp, 'r') as f:
            coords = f['coords'][:]
        require(coords.shape == (len(t), 2) and len(t) == int(row['manifest_patches']), 'PT/coords count mismatch')
        if unchanged:
            witness = self.witnesses[key]
            digest_h5, digest_g = witness['feature_h5_sha256'], witness['geometry_sha256']
            ps, dims = witness['patch_size'], tuple(witness['level_dim'])
        else:
            digest_h5, digest_g = sha(hp), sha(gp)
            with h5py.File(gp, 'r') as f:
                attrs = dict(f['coords'].attrs)
            dims = tuple(map(int, attrs['level_dim']))
            ps, level = int(attrs['patch_size']), int(attrs['patch_level'])
            require(np.isfinite(coords).all() and (coords >= 0).all(), 'invalid coords')
            require(level == 0 and ps > 0 and len(dims) == 2 and min(dims) > 0, 'invalid geometry')
            require((coords[:, 0] < dims[0]).all() and (coords[:, 1] < dims[1]).all(), 'coords outside dimensions')
        mode = 'prior_hash_and_unchanged_stat' if unchanged else 'full_hash'
        self.inputs.audit.record(hp, 'paired_feature_H5', digest_h5, verification=mode)
        self.inputs.audit.record(gp, 'patch_geometry_H5', digest_g, verification=mode,
                                 opened=not unchanged)
        witness = dict(feature_h5_sha256=digest_h5, geometry_sha256=digest_g,
                       patch_size=ps, level_dim=list(dims), patches=len(t))
        require(key not in self.witnesses or self.witnesses[key] == witness, 'H5/geometry changed during run')
        require(before == signatures(), 'input changed while being loaded ' + key)
        self.witnesses[key] = witness
        self.validated[key] = dict(signatures=before, feature_sha256=digest)
        result = (t, coords, ps, dims)
        size = t.nelement() * t.element_size() + coords.nbytes
        while self.cache and self.size + size > self.limit:
            _, (_, old_size) = self.cache.popitem(last=False)
            self.size -= old_size
        if size <= self.limit:
            self.cache[key] = (result, size)
            self.size += size
        return result

    def bag(self, row, limit, seed, pool=None):
        import numpy as np
        t, _, _, _ = self.load(row)
        if pool is None:
            pool = np.arange(len(t))
        ids = self.policy.indices(pool, limit, seed)
        require(len(ids) > 0 and ids.min() >= 0 and ids.max() < len(t), 'bad feature index')
        return t[ids], ids
