"""Byte-preserving feature staging, independent of training and model config."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import socket
import os


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def stage(datasets,cache):
    cache=Path(cache).absolute()
    if cache.is_symlink() or cache.resolve()!=cache:
        raise ValueError('Cache path may not contain symlinks')
    tables=[]
    sources=set()
    for dataset in datasets:
        with Path(dataset).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            fields=list(reader.fieldnames or [])
            data=list(reader)
        path_fields=[k for k in fields if k.endswith('_slide_path')]
        if not data or not path_fields:
            raise ValueError('Expected a nonempty wide-format dataset CSV')
        for row in data:
            for key in path_fields:
                if row[key]:
                    source=Path(row[key])
                    if not source.is_absolute() or source.is_symlink() or not source.is_file():
                        raise ValueError('Invalid source '+str(source))
                    if source.suffix not in ('.pt','.h5'):
                        raise ValueError('Only PT/H5 feature copies are supported')
                    sources.add(str(source))
        tables.append((Path(dataset),fields,data,path_fields))
    ancestor=cache.parent
    while not ancestor.exists():
        ancestor=ancestor.parent
    size=sum(Path(s).stat().st_size for s in sources)
    if shutil.disk_usage(ancestor).free < size+1024**3:
        raise RuntimeError('Insufficient staging capacity (requires payload plus 1 GiB)')
    cache.mkdir(parents=True,exist_ok=False)
    (cache/'features').mkdir()
    (cache/'datasets').mkdir()
    mapping={}
    files=[]
    for source in sorted(sources):
        src=Path(source)
        before=src.stat()
        target=cache/'features'/(hashlib.sha256(source.encode()).hexdigest()+src.suffix)
        with src.open('rb') as f,target.open('xb') as out:
            shutil.copyfileobj(f,out,1024*1024)
        after=src.stat()
        checksum=digest(target)
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or checksum!=digest(src):
            raise RuntimeError('Feature changed or copy verification failed: '+source)
        mapping[source]=str(target)
        files.append({'source':source,'path':str(target),'bytes':target.stat().st_size,'sha256':checksum})
    generated=[]
    for i,(source,fields,data,path_fields) in enumerate(tables):
        for key in path_fields:
            id_key=key.replace('_slide_path','_sample_id')
            if id_key not in fields:
                fields.append(id_key)
            for row in data:
                row.setdefault(id_key,row[key])
                if row[key]:
                    row[key]=mapping[row[key]]
        target=cache/'datasets'/(str(i)+'.csv')
        with target.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        generated.append({'source':str(source.resolve()),'source_sha256':digest(source),
                          'path':str(target),'sha256':digest(target)})
    manifest={'kind':'ephemeral-feature-staging-v1','host':socket.gethostname(),'cache_root':str(cache),
              'files':files,'datasets':generated,'feature_conversion':'none'}
    with (cache/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',action='append',required=True,help='Repeat for each fold to stage together')
    p.add_argument('--cache',required=True,help='New exclusively-owned per-server feature staging directory')
    a=p.parse_args()
    manifest=stage(a.dataset,a.cache)
    print(json.dumps({'manifest':str(Path(a.cache)/'manifest.json'),'datasets':manifest['datasets']},indent=2))


if __name__=='__main__':
    main()
