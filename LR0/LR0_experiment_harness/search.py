import argparse
import copy
import json
from datetime import datetime, timezone
import optuna
import yaml
from harness import ROOT, configure_mlflow, git_commit, load_config, run_config, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--stop-after',type=int)
    a=p.parse_args();commit=git_commit()
    spec=yaml.safe_load((ROOT/'configs/search.yaml').read_text(encoding='utf-8'))
    baseline=load_config(ROOT/'configs/base.yaml')
    state_path=ROOT/'.search_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'sessions':[]}
    mlflow=configure_mlflow()
    study=optuna.create_study(study_name=spec['study_name'],storage='sqlite:///'+(ROOT/'optuna.db').as_posix(),
        direction='maximize',load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=5))
    already=len(study.trials)
    if already>=spec['total_trials']: print('Already has 40 trials; no rerun.');return
    if 'parent_run_id' not in state:
        with mlflow.start_run(run_name='LR0_experiment_harness_search') as parent: state['parent_run_id']=parent.info.run_id
        write_json(state_path,state)
    def objective(trial):
        c=copy.deepcopy(baseline)
        for name,space in spec['space'].items():
            if isinstance(space,list): value=trial.suggest_categorical(name,space)
            elif name in ['n_layers','hidden']: value=trial.suggest_int(name,**space)
            else: value=trial.suggest_float(name,**space)
            c['model' if name in ['n_layers','hidden','dropout'] else 'train'][name]=value
        metrics,_=run_config(c,'search',trial=trial)
        return metrics['val_acc']
    count=spec['total_trials']-already
    if a.stop_after is not None: count=min(count,a.stop_after)
    session={'started_at':datetime.now(timezone.utc).isoformat(),'commit':commit,'before':already,
             'planned_trials_this_session':count,'sampler_rng_restored':False}
    state['sessions'].append(session);write_json(state_path,state)
    try:
        with mlflow.start_run(run_id=state['parent_run_id']):
            mlflow.set_tags({'work':'LR0_experiment_harness','series':'search_parent','commit':commit,'model':'MLP'})
            study.optimize(objective,n_trials=count,n_jobs=1,catch=(FloatingPointError,))
            mlflow.log_param('total_budget',40)
    finally:
        session.update(after=len(study.trials),ended_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path,state)
    print('Trial count:',len(study.trials),flush=True)

if __name__=='__main__':main()
