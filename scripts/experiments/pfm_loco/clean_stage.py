"""Remove only verified feature copies listed by an owned staging manifest."""
import argparse
import json
from pathlib import Path
import socket
import os
import sys

if __package__:
    from .stage import digest
else:
    from stage import digest


LOCAL_CACHE_ROOTS=(Path('/data2/lbliao'),Path('/data14/lbliao'),Path('/dev/shm'))


def deletion_plan(manifest_path):
    manifest_path=Path(manifest_path).absolute()
    m=json.loads(manifest_path.read_text())
    if m.get('kind')!='ephemeral-feature-staging-v1' or m.get('host')!=socket.gethostname():
        raise ValueError('Wrong manifest kind or host')
    root=Path(m['cache_root'])
    if not any(base in root.parents for base in LOCAL_CACHE_ROOTS):
        raise ValueError('Cleanup is restricted to explicitly local staging roots')
    if not any('cache' in part.lower() or 'staging' in part.lower() for part in root.parts):
        raise ValueError('Directory must identify itself as a feature cache/staging area')
    if root.resolve()!=root or manifest_path!=root/'manifest.json' or manifest_path.is_symlink():
        raise ValueError('Manifest/root mismatch or symlink')
    paths=[]
    for item in m['files']:
        path=Path(item['path'])
        if path.parent!=root/'features' or path.suffix not in ('.pt','.h5'):
            raise ValueError('Manifest contains a non-feature or out-of-root path')
        if path.is_symlink() or path.resolve()!=path or not path.is_file():
            raise ValueError('Missing or unsafe cache feature')
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('Cached feature has changed')
        if path in paths:
            raise ValueError('Duplicate cache feature')
        paths.append(path)
    return paths


def clean(manifest_path,receipt):
    paths=deletion_plan(manifest_path)  # Validate EVERY target before the first deletion.
    with Path(receipt).open('x') as f:
        f.write(json.dumps({'status':'cleanup_started','manifest':str(manifest_path),'files':len(paths)})+'\n')
        f.flush()
        for path in paths:
            # Recheck immutable identity immediately before deleting an owned copy.
            if path.is_symlink() or path.resolve()!=path:
                raise ValueError('Cache path changed during cleanup')
            os.unlink(path)
            f.write(json.dumps({'removed':str(path)})+'\n')
            f.flush()
        f.write(json.dumps({'status':'completed'})+'\n')
    # Manifest, adapted CSVs, receipts and directories intentionally remain.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--receipt',help='New receipt path; required with --execute')
    p.add_argument('--execute',action='store_true',help='Default is read-only inspection')
    a=p.parse_args()
    if a.execute:
        if not a.receipt:
            p.error('--execute requires --receipt')
        clean(a.manifest,a.receipt)
    else:
        print(json.dumps([str(p) for p in deletion_plan(a.manifest)],indent=2))


if __name__=='__main__':
    main()
