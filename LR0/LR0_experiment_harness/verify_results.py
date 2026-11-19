"""Audit files, metrics, provenance, resume count and deterministic predictions."""
import csv
import json
from collections import Counter
import numpy as np
import optuna
from harness import ROOT, recalculate, sha256, write_json

def records():
    out=[]
    for path in sorted((ROOT/'runs').glob('*/metadata.json')):
        meta=json.loads(path.read_text(encoding='utf-8'))
        if meta['commit']=='DEBUG' or meta['series'] not in ['repeatability','base_seeds','search','final']: continue
        config=__import__('yaml').safe_load((path.parent/'config.yaml').read_text(encoding='utf-8'))
        metrics=json.loads((path.parent/'metrics.json').read_text(encoding='utf-8'))
        out.append({'dir':path.parent,'meta':meta,'config':config,'metrics':metrics})
    return out

def history(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def audit(require_final=False):
    rows=records()
    for row in rows:
        meta=row['meta'];directory=row['dir']
        assert meta['status'] in ['completed','pruned','failed'],meta
        assert len(meta['commit'])==40
        assert sha256(directory/'config.yaml')==meta['config_sha256']
        assert sha256(ROOT/'data/split.json')==meta['split_sha256']
        assert recalculate(directory)==row['metrics'],directory
        if meta['status']=='completed':
            p=history(directory/'predictions.csv')
            for name in ['val']+(['test'] if row['config']['eval_test'] else []):
                part=[r for r in p if r['split']==name]
                assert len(part)==len({r['index'] for r in part})==360
                assert abs(float(history(directory/'history.csv')[-1]['val_acc'])-row['metrics']['val_acc'])<1e-15
        if row['meta']['series']!='final':assert 'test_acc' not in row['metrics']
    repeat=[r for r in rows if r['meta']['series']=='repeatability']
    assert len(repeat)==2
    first,second=[r['dir'] for r in repeat]
    predictions_equal=(first/'predictions.csv').read_bytes()==(second/'predictions.csv').read_bytes()
    a=history(first/'history.csv');b=history(second/'history.csv')
    deterministic_equal=all({k:v for k,v in x.items() if k!='epoch_s'}=={k:v for k,v in y.items() if k!='epoch_s'} for x,y in zip(a,b)) and len(a)==len(b)==30
    assert predictions_equal and deterministic_equal
    base=[r for r in rows if r['meta']['series']=='base_seeds']
    assert Counter(r['meta']['seed'] for r in base)==Counter(range(1,6))
    assert (next(r['dir'] for r in base if r['meta']['seed']==1)/'predictions.csv').read_bytes()!=(first/'predictions.csv').read_bytes()
    result={'formal_runs':len(rows),'states':dict(Counter(r['meta']['status'] for r in rows)),
            'seed0_predictions_byte_equal':predictions_equal,'seed0_history_metrics_exactly_equal':deterministic_equal,
            'seed0_history_including_time_byte_equal':(first/'history.csv').read_bytes()==(second/'history.csv').read_bytes(),
            'timing_note':'Actual wall-clock epoch times differ; never fabricated to satisfy byte equality.',
            'base_seed1_predictions_differ_from_seed0':True,'metrics_recomputed_from_files':True}
    if require_final:
        study=optuna.load_study(study_name='LR0_experiment_harness_search',storage='sqlite:///'+(ROOT/'optuna.db').as_posix())
        assert len(study.trials)==40
        state=json.loads((ROOT/'.search_state.json').read_text(encoding='utf-8'))
        assert [(s['before'],s['after']) for s in state['sessions']]==[(0,10),(10,40)]
        final=[r for r in rows if r['meta']['series']=='final']
        assert len(final)==10 and len(set(r['meta']['commit'] for r in final))==1
        assert Counter(r['meta']['seed'] for r in final)==Counter({s:2 for s in range(101,106)})
        assert len(rows)==57 # 2 + 5 + 40 + 10; PRUNED/FAIL remain visible
        for r in final:assert r['meta']['status']=='completed' and r['metrics']['test_n']==360
        selection=json.loads((ROOT/'configs/selection.json').read_text(encoding='utf-8'))
        assert sha256(ROOT/'configs/best.yaml')==selection['best_config_sha256']
        for r in final:
            kind=r['meta']['slot'].split('_')[1]
            expected=__import__('yaml').safe_load((ROOT/f'configs/{kind}.yaml').read_text(encoding='utf-8'))
            for key in ['model','train','runtime','split_seed']:
                assert r['config'][key]==expected[key],(r['meta']['run_id'],key)
        result.update(trials=40,search_sessions=state['sessions'],final_runs=10,
                      final_selection_unchanged=True)
    write_json(ROOT/'checks/results_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args();audit(args.final)
