"""Freeze selection using only COMPLETE trials and last-epoch validation accuracy."""
import copy
import optuna
import yaml
from harness import ROOT, git_commit, load_config, sha256, write_json

def main():
    commit=git_commit()
    study=optuna.load_study(study_name='LR0_experiment_harness_search',storage='sqlite:///'+(ROOT/'optuna.db').as_posix())
    assert len(study.trials)==40, 'Selection requires all 40 trials'
    c=copy.deepcopy(load_config(ROOT/'configs/base.yaml'))
    for key,value in study.best_params.items():
        c['model' if key in ['n_layers','hidden','dropout'] else 'train'][key]=value
    dest=ROOT/'configs/best.yaml'
    if dest.exists(): raise FileExistsError('Selection already frozen; refusing to overwrite')
    dest.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    write_json(ROOT/'configs/selection.json',{'criterion':'maximum last-epoch val_acc among COMPLETE; test not read',
        'trial_number':study.best_trial.number,'val_acc':study.best_value,'source_commit':commit,
        'best_config_sha256':sha256(dest),'search_trials':40})
    print(study.best_trial.number,study.best_value,study.best_params)

if __name__=='__main__':main()
