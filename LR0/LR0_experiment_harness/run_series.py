"""One training command per run; refuse duplicate series keys (including failures)."""
import argparse
import json
import subprocess
import sys
from harness import ROOT, git_commit

def main():
    p=argparse.ArgumentParser();p.add_argument('series',choices=['repeatability','base_seeds','final'])
    a=p.parse_args();git_commit()
    existing=[]
    for path in (ROOT/'runs').glob('*/metadata.json'):
        existing.append(json.loads(path.read_text(encoding='utf-8')))
    if a.series=='repeatability':jobs=[('base',0),('base',0)]
    elif a.series=='base_seeds':jobs=[('base',s) for s in range(1,6)]
    else:jobs=[(kind,s) for s in range(101,106) for kind in ['base','best']]
    slots=set()
    for m in existing:
        if m['series']!=a.series:continue
        if m['slot'] in slots:raise RuntimeError('Duplicate saved series slot')
        slots.add(m['slot'])
    for index,(kind,seed) in enumerate(jobs):
        slot=f'{index:02d}_{kind}_{seed}'
        if slot in slots:
            print('Already recorded; preserve outcome:',a.series,slot,flush=True);continue
        cmd=[sys.executable,'train.py','--config','configs/'+kind+'.yaml','--seed',str(seed),'--series',a.series]
        cmd+=['--slot',slot]
        if a.series=='final':cmd.append('--eval-test')
        subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
