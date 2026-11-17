"""Optional adapter: explicit frozen role CSV paths -> train/val and test CSVs."""
import argparse
import csv
from pathlib import Path


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write(path,data,fields):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def prepare(train, selection, test, output, calibration=None):
    roles={'train':read(train),'selection':read(selection),'test':read(test)}
    if calibration:
        roles['calibration']=read(calibration)
    patients,paths=set(),set()
    for role,data in roles.items():
        if not data:
            raise ValueError('Empty '+role)
        current_patients,current_paths=set(),set()
        for row in data:
            path=row['slide_path']
            if not row['patient_group_id'] or row['label'] not in ('0','1'):
                raise ValueError('Invalid patient or label')
            if not Path(path).is_absolute() or path in current_paths:
                raise ValueError('Invalid or duplicate feature path')
            if row.get('partition',role)!=role:
                raise ValueError('Wrong partition')
            current_patients.add(row['patient_group_id'])
            current_paths.add(path)
        if patients&current_patients or paths&current_paths:
            raise ValueError('Cross-role patient or slide overlap')
        patients.update(current_patients)
        paths.update(current_paths)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    fields=['train_slide_path','train_label','val_slide_path','val_label','test_slide_path','test_label']
    result=[]
    for i in range(max(len(roles['train']),len(roles['selection']))):
        row=dict.fromkeys(fields,'')
        for role,prefix in (('train','train'),('selection','val')):
            if i<len(roles[role]):
                source=roles[role][i]
                row[prefix+'_slide_path']=source['slide_path']
                row[prefix+'_label']=source['label']
        result.append(row)
    write(output/'train_val.csv',result,fields)
    write(output/'test.csv',[{'test_slide_path':r['slide_path'],'test_label':r['label']}
                            for r in roles['test']],['test_slide_path','test_label'])
    # Preserve the membership evidence; the originals stay at their supplied paths.
    for role,source in [('train',train),('selection',selection),('test',test)]+(
            [('calibration',calibration)] if calibration else []):
        with (output/('source_'+role+'.csv')).open('xb') as f:
            f.write(Path(source).read_bytes())
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','selection','test','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--calibration')
    a=p.parse_args()
    print(prepare(a.train,a.selection,a.test,a.output,a.calibration))


if __name__=='__main__':
    main()
