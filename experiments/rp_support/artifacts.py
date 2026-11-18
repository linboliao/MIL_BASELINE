"""Write and verify native RP support artifacts."""
import hashlib,json,os,time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def json_x(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def csv_x(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        frame.to_csv(f, index=False)


def read(path):
    return pd.read_csv(path, dtype=str).fillna('')


def event(run, kind, **kwargs):
    record = {'event': kind, 'utc': utc(), 'pid': os.getpid(), **kwargs}
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    json_x(run/'events'/f'{time.time_ns()}-{os.getpid()}-{kind}.json', record)

def manifest(root, dest, exclude=()):
    rows = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p != dest and p.name not in exclude:
            rows.append({'relative_path': str(p.relative_to(root)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)})
    csv_x(dest, pd.DataFrame(rows)); return rows


def verify_manifest(root, path):
    df = read(path)
    for r in df.itertuples(index=False):
        p = root/r.relative_path
        assert p.is_file() and p.stat().st_size == int(r.size_bytes) and sha(p) == r.sha256, str(p)
    return len(df)
