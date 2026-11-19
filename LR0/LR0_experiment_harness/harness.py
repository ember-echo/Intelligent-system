"""Shared reproducible training implementation, used unchanged by Optuna."""
import copy
import csv
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_commit(root=ROOT):
    def git(*args):
        return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
    dirty = git('status','--porcelain')
    if dirty:
        raise RuntimeError('Uncommitted changes: final run refused.\n'+dirty)
    return git('rev-parse','HEAD')

def load_config(path, seed=None):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if seed is not None: config['seed'] = seed
    validate_config(config)
    return config

def validate_config(c):
    assert c['split_seed'] == 0, 'split_seed must be 0'
    assert isinstance(c['seed'],int) and c['seed'] >= 0, 'seed must be a nonnegative integer'
    m,t,r = c['model'], c['train'], c['runtime']
    assert m['n_layers'] in [1,2,3] and isinstance(m['hidden'],int) and 32 <= m['hidden'] <= 512
    assert 0 <= m['dropout'] <= 0.5
    assert np.isfinite(t['lr']) and t['lr'] > 0, 'lr must be positive'
    assert np.isfinite(t['weight_decay']) and t['weight_decay'] >= 0
    assert t['batch_size'] in [32,64,128] and t['epochs'] == 30
    assert r['device'] == 'cpu' and r['num_threads'] == 1, 'This protocol fixes CPU, one thread'

def read_data():
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    assert sha256(ROOT/'data/split.json') == manifest['files']['data/split.json']['sha256'], 'split hash mismatch'
    split = json.loads((ROOT/'data/split.json').read_text(encoding='utf-8'))
    X,y = load_digits(return_X_y=True)
    assert hashlib.sha256(X.tobytes()).hexdigest() == manifest['raw_X_sha256']
    assert hashlib.sha256(y.tobytes()).hexdigest() == manifest['raw_y_sha256']
    parts = [split[k] for k in ['train','val','test']]
    assert [len(v) for v in parts] == [1077,360,360]
    flat = sum(parts,[])
    assert len(flat) == len(set(flat)) == len(y) and set(flat) == set(range(len(y)))
    assert all(len(np.unique(y[v])) == 10 for v in parts)
    return torch.tensor(X/16,dtype=torch.float32), torch.tensor(y,dtype=torch.long), split

def accuracy(true,pred):
    if not len(true): raise ValueError('empty predictions')
    return float(np.mean(np.asarray(true)==np.asarray(pred)))

def recalculate(run_dir):
    with (run_dir/'history.csv').open(newline='',encoding='utf-8') as f: hist=list(csv.DictReader(f))
    pred=[]
    if (run_dir/'predictions.csv').exists():
        with (run_dir/'predictions.csv').open(newline='',encoding='utf-8') as f: pred=list(csv.DictReader(f))
    result={'epochs_completed':len(hist),'train_time_s':sum(float(r['epoch_s']) for r in hist)}
    for name in ['val','test']:
        rows=[r for r in pred if r['split']==name]
        if rows:
            result[name+'_acc']=accuracy([int(r['y_true']) for r in rows],[int(r['y_pred']) for r in rows])
            result[name+'_loss']=float(np.mean([float(r['nll']) for r in rows]))
            result[name+'_n']=len(rows)
    if 'val_acc' not in result and hist:
        result.update(val_acc=float(hist[-1]['val_acc']),val_loss=float(hist[-1]['val_loss']))
    return result

def configure_mlflow():
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'mlflow.db').as_posix())
    client=mlflow.tracking.MlflowClient()
    if not client.get_experiment_by_name('LR0_experiment_harness'):
        client.create_experiment('LR0_experiment_harness',artifact_location=(ROOT/'mlartifacts').as_uri())
    mlflow.set_experiment('LR0_experiment_harness')
    return mlflow

def run_config(config, series, eval_test=False, trial=None, debug=False):
    validate_config(config)
    commit = 'DEBUG' if debug else git_commit()
    if eval_test and series != 'final': raise ValueError('Test is reserved for the final series')
    X,y,split = read_data()
    seed=config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    layers=[]; inputs=64
    for _ in range(config['model']['n_layers']):
        layers += [nn.Linear(inputs,config['model']['hidden']),nn.ReLU(),nn.Dropout(config['model']['dropout'])]
        inputs=config['model']['hidden']
    model=nn.Sequential(*layers,nn.Linear(inputs,10))
    optimizer=torch.optim.Adam(model.parameters(),lr=config['train']['lr'],weight_decay=config['train']['weight_decay'])
    loader=DataLoader(TensorDataset(X[split['train']],y[split['train']]),
                      batch_size=config['train']['batch_size'],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    run_dir=ROOT/'runs'/run_id; run_dir.mkdir(parents=True)
    effective=copy.deepcopy(config); effective.update(series=series,eval_test=eval_test)
    (run_dir/'config.yaml').write_text(yaml.safe_dump(effective,sort_keys=False),encoding='utf-8')
    meta={'run_id':run_id,'experiment':'LR0_experiment_harness','series':series,'status':'running',
          'started_at':datetime.now(timezone.utc).isoformat(),'commit':commit,'seed':seed,
          'python':__import__('sys').version,'torch_build':torch.__version__,'versions':{p:importlib.metadata.version(p) for p in
          ['torch','scikit-learn','numpy','optuna','mlflow','PyYAML']},'device':'cpu','num_threads':1,
          'cpu':__import__('platform').processor(),'split_sha256':sha256(ROOT/'data/split.json'),
          'manifest_sha256':sha256(ROOT/'data/manifest.json'),'config_sha256':sha256(run_dir/'config.yaml'),
          'timing_scope':'sum of epoch train+validation compute; excludes setup, CSV and MLflow writes',
          'trial_number':None if trial is None else trial.number,'slot':config.get('slot')}
    write_json(run_dir/'metadata.json',meta)
    mlflow=configure_mlflow()
    import optuna
    metrics={}
    with mlflow.start_run(run_name=series+'_'+str(seed)+'_'+run_id,nested=trial is not None) as active:
        meta['mlflow_run_id']=active.info.run_id
        mlflow.set_tags({'commit':commit,'series':series,'seed':seed,'work':'LR0_experiment_harness','model':'MLP',
                         'run_id':run_id,'trial_number':meta['trial_number'] if trial else 'none'})
        for section,value in effective.items():
            if isinstance(value,dict): mlflow.log_params({section+'.'+k:v for k,v in value.items()})
            else: mlflow.log_param(section,value)
        try:
            with (run_dir/'history.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','val_loss','val_acc','epoch_s']);writer.writeheader();f.flush()
                for epoch in range(1,31):
                    start=time.perf_counter();model.train();total_loss=0.0
                    for xb,yb in loader:
                        optimizer.zero_grad();loss=nn.functional.cross_entropy(model(xb),yb)
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                        loss.backward();optimizer.step();total_loss += loss.item()*len(yb)
                    model.eval()
                    with torch.no_grad():
                        logits=model(X[split['val']]); labels=y[split['val']]
                        nll=nn.functional.cross_entropy(logits,labels,reduction='none')
                        val_acc=accuracy(labels.numpy(),logits.argmax(1).numpy())
                    elapsed=time.perf_counter()-start
                    row={'epoch':epoch,'train_loss':total_loss/len(split['train']),
                         'val_loss':float(nll.mean()),'val_acc':val_acc,'epoch_s':elapsed}
                    writer.writerow(row);f.flush()
                    mlflow.log_metrics({k:v for k,v in row.items() if k!='epoch'},step=epoch)
                    if trial is not None:
                        trial.report(val_acc,epoch)
                        if trial.should_prune(): raise optuna.TrialPruned()
            with (run_dir/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['split','index','y_true','y_pred','nll']);writer.writeheader()
                for name in ['val']+(['test'] if eval_test else []):
                    indices=split[name]
                    with torch.no_grad():
                        logits=model(X[indices]); losses=nn.functional.cross_entropy(logits,y[indices],reduction='none')
                    for i,answer,loss in zip(indices,logits.argmax(1).tolist(),losses.tolist()):
                        writer.writerow({'split':name,'index':i,'y_true':int(y[i]),'y_pred':answer,'nll':loss})
                    f.flush()
            meta['status']='completed'
        except optuna.TrialPruned:
            meta['status']='pruned'; raise
        except Exception:
            meta['status']='failed';meta['error']=traceback.format_exc()
            (run_dir/'error.txt').write_text(meta['error'],encoding='utf-8');raise
        finally:
            meta['ended_at']=datetime.now(timezone.utc).isoformat()
            metrics=recalculate(run_dir)
            write_json(run_dir/'metrics.json',metrics);write_json(run_dir/'metadata.json',meta)
            mlflow.log_metrics(metrics);mlflow.set_tag('state',{'completed':'COMPLETE','pruned':'PRUNED','failed':'FAIL'}[meta['status']])
            mlflow.log_artifacts(str(run_dir),artifact_path='run')
            if trial is not None:
                trial.set_user_attr('run_id',run_id);trial.set_user_attr('epochs_completed',metrics['epochs_completed'])
                trial.set_user_attr('train_time_s',metrics['train_time_s'])
    print(json.dumps({'run_id':run_id,'series':series,'seed':seed,**metrics}),flush=True)
    return metrics,run_id
