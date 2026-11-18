"""Exclusive artifact writers and provenance helpers."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def put(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    require(bool(rows), 'refuse empty scientific output')
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
