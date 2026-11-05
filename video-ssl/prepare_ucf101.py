"""Use the supplied official split lists; hold out training SOURCE GROUPS for val."""
import argparse
import csv
import random
import re
from pathlib import Path
from collections import defaultdict
from data import read_manifest


def prepare(root, split_dir, output, split=1, seed=17):
    root, split_dir, output = Path(root).resolve(), Path(split_dir), Path(output).resolve()
    by_class, rows = defaultdict(lambda: defaultdict(list)), []
    def source(name):
        match = re.fullmatch(r'(v_.+_g\d+)_c\d+\.avi', Path(name).name)
        if not match:
            raise ValueError(f'Not an original UCF101 filename: {name}')
        return match[1]
    for line in (split_dir/f'trainlist{split:02}.txt').read_text().splitlines():
        name = line.split()[0]
        by_class[Path(name).parts[0]][source(name)].append(name)
    rng = random.Random(seed)
    for label in sorted(by_class):
        groups = sorted(by_class[label]); rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError('Not enough training groups for validation')
        nval = max(1, round(0.2*len(groups)))
        for i, group in enumerate(groups):
            for name in by_class[label][group]:
                rows.append({'path':str(root/name),'label':label,
                             'split':'val' if i<nval else 'train','source_id':group})
    for line in (split_dir/f'testlist{split:02}.txt').read_text().splitlines():
        name=line.split()[0]
        rows.append({'path':str(root/name),'label':Path(name).parts[0],
                     'split':'test','source_id':source(name)})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','label','split','source_id'])
        writer.writeheader(); writer.writerows(rows)
    read_manifest(output)  # Includes source leakage and actual video existence checks.
    print(f'Validated {len(rows)} videos: {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--videos',required=True)
    p.add_argument('--splits',required=True)
    p.add_argument('--output',default='data/manifest.csv')
    p.add_argument('--split',type=int,choices=[1,2,3],default=1)
    a=p.parse_args(); prepare(a.videos,a.splits,a.output,a.split)
