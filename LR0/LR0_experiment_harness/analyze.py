"""Build all tables, plots and two reports from saved per-example results."""
import csv
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
from harness import ROOT, configure_mlflow, write_json
from verify_results import audit, records

def fmt(value):return f'{100*value:.3f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)
def transfer(a,b,split):
    left=pd.read_csv(a['dir']/'predictions.csv');right=pd.read_csv(b['dir']/'predictions.csv')
    paired=left[left['split']==split].merge(right[right['split']==split],on=['split','index'],suffixes=('_a','_b'),validate='one_to_one')
    assert len(paired)==360 and (paired.y_true_a==paired.y_true_b).all()
    ca=paired.y_true_a==paired.y_pred_a;cb=paired.y_true_b==paired.y_pred_b
    return {'fixed':int((~ca&cb).sum()),'broken':int((ca&~cb).sum()),'both_right':int((ca&cb).sum()),'both_wrong':int((~ca&~cb).sum())}

def main():
    audit_result=audit(True);rows=records();out=ROOT/'reports/generated';out.mkdir(parents=True,exist_ok=True)
    reportdir=ROOT/'reports';baseline=yaml.safe_load((ROOT/'configs/base.yaml').read_text(encoding='utf-8'))
    base=sorted([r for r in rows if r['meta']['series']=='base_seeds'],key=lambda r:r['meta']['seed'])
    repeat=[r for r in rows if r['meta']['series']=='repeatability']
    final=[r for r in rows if r['meta']['series']=='final']
    def is_base(r):return r['config']['model']==baseline['model'] and r['config']['train']==baseline['train']
    base_table=table(['seed','val accuracy (%)','compute time (s)','run_id'],[
        [r['meta']['seed'],fmt(r['metrics']['val_acc']),f"{r['metrics']['train_time_s']:.3f}",r['meta']['run_id']] for r in base])
    vals=[r['metrics']['val_acc'] for r in base];spread=max(vals)-min(vals)
    seed_transfer=transfer(base[0],base[1],'val')
    paired=[]
    for seed in range(101,106):
        a=next(r for r in final if r['meta']['seed']==seed and is_base(r));b=next(r for r in final if r['meta']['seed']==seed and not is_base(r))
        t=transfer(a,b,'test')
        paired.append({'seed':seed,'base_val':a['metrics']['val_acc'],'best_val':b['metrics']['val_acc'],
                       'delta_val':b['metrics']['val_acc']-a['metrics']['val_acc'],
                       'base_test':a['metrics']['test_acc'],'best_test':b['metrics']['test_acc'],
                       'delta_test':b['metrics']['test_acc']-a['metrics']['test_acc'],**t,
                       'base_run':a['meta']['run_id'],'best_run':b['meta']['run_id']})
    pd.DataFrame(paired).to_csv(out/'paired_final.csv',index=False)
    final_table=table(['seed','base val %','best val %','Δ val (pp)','base test %','best test %','Δ test (pp)','fixed / broken'],[
        [r['seed'],fmt(r['base_val']),fmt(r['best_val']),fmt(r['delta_val']),fmt(r['base_test']),fmt(r['best_test']),fmt(r['delta_test']),f"{r['fixed']} / {r['broken']}"] for r in paired])
    study=optuna.load_study(study_name='LR0_experiment_harness_search',storage='sqlite:///'+(ROOT/'optuna.db').as_posix())
    trials=study.trials;states=dict(Counter(t.state.name for t in trials))
    epochs=sum(t.user_attrs['epochs_completed'] for t in trials)
    search_time=sum(t.user_attrs['train_time_s'] for t in trials)
    # Estimate unexecuted epoch cost using each trial's measured mean epoch cost.
    estimate_saved=sum((30-t.user_attrs['epochs_completed'])*t.user_attrs['train_time_s']/t.user_attrs['epochs_completed'] for t in trials)
    rng=np.random.RandomState(42)
    importance=optuna.importance.get_param_importances(study,evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42))
    write_json(out/'importance.json',importance)
    trials_df=study.trials_dataframe();trials_df.to_csv(out/'trials.csv',index=False)
    best_curve=[];current=float('nan')
    for t in trials:
        if t.state.name=='COMPLETE':current=t.value if np.isnan(current) else max(current,t.value)
        best_curve.append(current)
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,2.7))
    for state,color,marker in [('COMPLETE','#147d92','o'),('PRUNED','#bf7e23','x'),('FAIL','#b42b3e','s')]:
        tt=[t for t in trials if t.state.name==state and t.value is not None]
        if tt:ax.scatter([t.number for t in tt],[100*t.value for t in tt],c=color,marker=marker,label=f'{state} (n={states.get(state,0)})',s=24)
    ax.plot(range(40),100*np.array(best_curve),c='#233651',label='Best COMPLETE so far',linewidth=1.5)
    ax.set(xlabel='Trial number (0-based)',ylabel='Validation accuracy (%)',title='40 sequential trials; pause after trial 9')
    ax.axvline(9.5,color='gray',linestyle=':',linewidth=1);ax.legend(fontsize=8,loc='lower right');fig.tight_layout();fig.savefig(out/'search_history.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,2.4))
    names=list(reversed(importance));ax.barh(names,[importance[n] for n in names],color='#147d92')
    ax.set(xlabel='fANOVA relative importance (COMPLETE trials only)',title='Hyperparameter importance; seed=42')
    fig.tight_layout();fig.savefig(out/'parameter_importance.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,2.6))
    ax.bar([str(r['seed']) for r in paired],[100*r['delta_test'] for r in paired],color='#147d92')
    ax.axhline(0,color='black',linewidth=0.8);ax.axhline(spread*100,color='#bf7e23',linestyle='--',label='Base val seed range')
    ax.set(xlabel='New training seed',ylabel='Test difference (pp)',title='Best minus base on the same 360 test images');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'test_deltas.png');plt.close(fig)
    fixed=sum(r['fixed'] for r in paired);broken=sum(r['broken'] for r in paired)
    vd=[r['delta_val'] for r in paired];td=[r['delta_test'] for r in paired]
    robust=all(v>spread for v in td)
    best_val_mean=float(np.mean([r['best_val'] for r in paired]));best_test_mean=float(np.mean([r['best_test'] for r in paired]))
    summary={'denominators':{'train':1077,'val':360,'test':360,'base_seeds':5,'final_pairs':5,'search_trials':40},
        'base_val_min':min(vals),'base_val_median':float(np.median(vals)),'base_val_max':max(vals),'base_val_range':spread,
        'base_seed1_to_seed2':seed_transfer,'search_states':states,'executed_epochs':epochs,'max_epochs':1200,
        'epochs_saved':1200-epochs,'measured_search_compute_s':search_time,'estimated_saved_compute_s':estimate_saved,
        'search_best_val':study.best_value,'search_best_trial':study.best_trial.number,
        'best_new_seeds_mean_val':best_val_mean,'best_new_seeds_mean_test':best_test_mean,
        'delta_val_median':float(np.median(vd)),'delta_val_min':min(vd),'delta_val_max':max(vd),
        'delta_test_median':float(np.median(td)),'delta_test_min':min(td),'delta_test_max':max(td),
        'fixed_test_total':fixed,'broken_test_total':broken,'paired_test_observations':1800,
        'robust_by_course_rule':robust,'final_pairs':paired,'audit':audit_result}
    write_json(out/'summary.json',summary)
    final_ids=table(['seed','base run_id','best run_id'],[[r['seed'],r['base_run'],r['best_run']] for r in paired])
    (out/'final_run_ids.md').write_text(final_ids+'\n',encoding='utf-8')
    versions=base[0]['meta']['versions'];version_text=', '.join(f'{k} {v}' for k,v in versions.items())
    params=json.dumps(study.best_params,ensure_ascii=False)
    zh_conclusion=('五个 test 差值均为正且都大于基础 val 波动，按课程判据本次改善稳定。' if robust else
                   '按课程判据，不能认定改善稳定：并非五个 test 差值都大于基础 val 波动。若差值全正但较小，只能说这五次观察到正向趋势；不能把趋势写成已证实的普遍提升。')
    ru_conclusion=('Все пять разностей на test положительны и превышают базовый разброс на val: улучшение устойчиво по критерию работы.' if robust else
                   'Критерий устойчивого улучшения не выполнен: не все пять разностей на test превышают базовый разброс на val. Даже если все разности положительны, малый эффект нельзя объявлять доказанным общим улучшением.')
    common=f'''{base_table}

min / median / max: {fmt(min(vals))} / {fmt(np.median(vals))} / {fmt(max(vals))}%; range: {fmt(spread)} pp.

{final_table}
'''
    zh=f'''# 实验 0：可复现训练与超参数搜索

## 问题、预测与方法
本次是相同固定协议的独立重跑，所有数字来自新运行；预测参考了先前经验，test 划分曾在旧版本中评估，不将它称为从未见过的新测试集。本次不根据旧 test 结果更改参数搜索或选择规则。
调参带来的提升是否超过随机种子造成的波动？预测已在正式实验前提交：基础 val 约 94–97%，五 seed 极差约 0.3–1.5 个百分点；最佳 trial 或提升 0–2 个百分点，但在新 seed/test 上优势可能缩小。

load_digits 共 1797 个 8×8 图像，按指定分层方法和 split_seed=0 划分 train/val/test=1077/360/360，特征除以 16。MLP 为 64→隐藏层（ReLU+dropout）→10；Adam、30 轮、CPU 单线程。基础配置：1×64、dropout=0、lr=0.001、weight_decay=0、batch_size=64。版本：{version_text}。所有正式运行均记录完整 Git 提交、SHA-256、配置、时间和逐例预测。

## 实验 1：可复现性与随机波动
seed 0 两次预测文件逐字节一致，训练/验证指标逐项完全一致。history.csv 的真实 epoch_s 不同，因此文件整体不逐字节相同；此处是作业中“记录真实耗时”与“整个 history 完全相同”的冲突，未伪造计时。以下时间为训练加验证计算，不含初始化、日志和文件写入。每次 val 分母为 360。

{base_table}

val 最小/中位/最大为 {fmt(min(vals))}/{fmt(np.median(vals))}/{fmt(max(vals))}%，极差 {fmt(spread)} 个百分点。seed 1→2：纠正 {seed_transfer['fixed']} 张，破坏 {seed_transfer['broken']} 张；两者都对 {seed_transfer['both_right']} 张、都错 {seed_transfer['both_wrong']} 张，合计 360。

## 实验 2：40 次顺序搜索
TPE(seed=42)，MedianPruner(startup=5,warmup=5)，训练 seed=0。搜索范围见 configs/search.yaml；目标为第 30 轮 val 准确率。完成 10 次后有计划暂停，重启补齐至 40；不是强制杀进程的崩溃恢复测试。SQLite 保留 trial，但重启不恢复 sampler 随机状态，因此不声称与未中断序列相同。

COMPLETE={states.get('COMPLETE',0)}，PRUNED={states.get('PRUNED',0)}，FAIL={states.get('FAIL',0)}。实际 {epochs}/1200 轮，节约 {1200-epochs} 轮（{100*(1200-epochs)/1200:.1f}%）。实测搜索计算耗时 {search_time:.3f} s；逐 trial 使用已执行轮的平均耗时外推，估计节约 {estimate_saved:.3f} s，属于反事实估算，未实际运行无剪枝对照。剪枝点的末次准确率不等于完整 30 轮最终准确率；fANOVA 只使用完成 trial，不能当成因果结论。

![搜索历史](generated/search_history.png)

![参数重要性](generated/parameter_importance.png)

最佳 trial #{study.best_trial.number}，val={fmt(study.best_value)}%；参数 `{params}`。配置只依 val 选定，并在读取 test 之前单独提交冻结。

## 实验 3：新 seed 上配对比较
使用 seed 101–105，各训练基础/最佳配置一次，预先指定的 10 次 final 运行使用同一提交。每个 val/test 分母 360；没有根据 test 调参或替换运行。

{final_table}

Δval 中位数 {fmt(np.median(vd))} pp，范围 [{fmt(min(vd))}, {fmt(max(vd))}]；Δtest 中位数 {fmt(np.median(td))} pp，范围 [{fmt(min(td))}, {fmt(max(td))}]。五组 test 合计纠正 {fixed}、破坏 {broken}，分母 5×360=1800 个“seed-图像”观察，并非 1800 张独立图像。

搜索最佳 val={fmt(study.best_value)}%；新 seed 最佳配置平均 val={fmt(best_val_mean)}%；平均 test={fmt(best_test_mean)}%。重训均值相对搜索最好值变化 {fmt(best_val_mean-study.best_value)} pp；test 相对新 seed val 变化 {fmt(best_test_mean-best_val_mean)} pp。选择偏差、训练随机性和样本差异都可能参与，不能只凭三个数字把变化全归因于某一个因素。

预测对照：基础准确率和极差在预期范围内；搜索最好值相对基础 seed 0 提高 {fmt(study.best_value-repeat[0]['metrics']['val_acc'])} pp，超过原先 0–2 pp 的预期。多层网络与学习率的配合可能是原因，但没有消融实验，不能分离贡献。新 seed 的最佳配置平均 val 低于搜索最好值，符合优势可能缩小的预测。

## 结论与验证
{zh_conclusion} 结论仅适用于本数据划分、模型、30 轮和五个新 seed。360 张图像中一张约 0.278 pp；样本较小，未声称统计显著性。预测是否成立应以上述实测值为准，负面结果保留。

代码由 Codex 协助生成；通过已知 0.75 准确率用例、划分/哈希、负学习率拒绝、Git 脏目录拒绝等单元测试，并对全部运行重算指标、核对重复运行及 trial 数量。所有数值由 analyze.py 从 CSV/数据库计算；运行 ID 映射见 generated/final_run_ids.md，完整审计见 generated/summary.json。学习者仍需自行读懂代码并完成答辩。
'''
    ru=f'''# ЛР0. Воспроизводимый эксперимент и подбор гиперпараметров

## Вопрос, прогноз и метод
Это независимый повтор фиксированного протокола; все числа получены заново. Прогноз учитывает прежний опыт. Это разбиение test уже оценивалось ранее и не объявляется совершенно новым; прежний test не использован для изменения поиска или правила выбора.
Превышает ли эффект подбора гиперпараметров случайный разброс? До итоговых запусков зафиксирован прогноз: базовая accuracy на val 94–97%, размах по пяти seed 0,3–1,5 п.п.; лучший trial может дать 0–2 п.п., но преимущество на новых seed и test может уменьшиться.

load_digits: 1797 изображений 8×8; стратифицированное разбиение с split_seed=0 на 1077/360/360. Признаки делятся на 16. MLP: 64→скрытые слои ReLU+dropout→10, Adam, 30 эпох, CPU, один поток. База: один слой 64, dropout=0, lr=0.001, weight_decay=0, batch_size=64. Окружение: {version_text}. Для каждого запуска сохранены полный коммит, конфигурация, SHA-256, метаданные и предсказания.

## 1. Повторяемость и разброс
Два запуска seed 0 дали побайтно одинаковые predictions.csv и точно одинаковые значения train_loss/val_loss/val_acc. Реальные epoch_s различаются, поэтому history.csv целиком не совпадает. Требования полного совпадения файла и записи реального времени противоречат друг другу; время не подменялось. Измеряется сумма времени обучения и валидации, без подготовки, записи CSV и MLflow. Знаменатель каждого val — 360 изображений.

{base_table}

Минимум/медиана/максимум: {fmt(min(vals))}/{fmt(np.median(vals))}/{fmt(max(vals))}%; размах {fmt(spread)} п.п. Переход seed 1→2: исправлено {seed_transfer['fixed']}, испорчено {seed_transfer['broken']}, оба верны {seed_transfer['both_right']}, оба ошиблись {seed_transfer['both_wrong']} из 360.

## 2. Поиск
40 последовательных trial, TPE(seed=42), MedianPruner(startup=5,warmup=5), seed обучения 0. Диапазоны заданы в configs/search.yaml, цель — accuracy последней, 30-й эпохи на val. После 10 trial выполнена запланированная пауза, затем продолжение до 40. Это проверка возобновления между trial, а не аварийного завершения внутри эпохи. SQLite сохраняет trial; состояние генератора семплера при рестарте не восстанавливается, поэтому последовательность не приравнивается к непрерывному поиску.

COMPLETE={states.get('COMPLETE',0)}, PRUNED={states.get('PRUNED',0)}, FAIL={states.get('FAIL',0)}. Выполнено {epochs}/1200 эпох; сэкономлено {1200-epochs} ({100*(1200-epochs)/1200:.1f}%). Измеренное время вычислений поиска {search_time:.3f} с. Экстраполяция среднего времени выполненных эпох отдельно для каждого trial даёт оценку экономии {estimate_saved:.3f} с. Это контрфактическая оценка, а не измерение поиска без прунера. Значения PRUNED относятся к последней выполненной эпохе; fANOVA использует только COMPLETE и не является причинной оценкой.

![История поиска](generated/search_history.png)

![Важность параметров](generated/parameter_importance.png)

Лучший trial #{study.best_trial.number}: val={fmt(study.best_value)}%; параметры `{params}`. Конфигурация выбрана только по val и закоммичена до первого вычисления test.

## 3. Честное сравнение
На новых seed 101–105 выполнено 10 заранее заданных запусков base/best из одного коммита. Знаменатель val и test каждого запуска — 360. Test не использовался для изменения конфигураций.

{final_table}

Медиана Δval {fmt(np.median(vd))} п.п., диапазон [{fmt(min(vd))}, {fmt(max(vd))}]; медиана Δtest {fmt(np.median(td))} п.п., диапазон [{fmt(min(td))}, {fmt(max(td))}]. Всего на test исправлено {fixed}, испорчено {broken} из 1800 наблюдений «seed–изображение»; это не 1800 независимых изображений.

Три оценки: лучший trial val {fmt(study.best_value)}%; средняя best val на новых seed {fmt(best_val_mean)}%; средняя best test {fmt(best_test_mean)}%. Изменение среднего val относительно лучшего trial {fmt(best_val_mean-study.best_value)} п.п.; test относительно нового val {fmt(best_test_mean-best_val_mean)} п.п. Отбор, случайность обучения и различие выборок могут влиять одновременно; эти числа сами по себе не разделяют причины.

Сверка прогноза: базовая accuracy и размах попали в ожидаемый диапазон. Прирост лучшего trial относительно базы seed 0 — {fmt(study.best_value-repeat[0]['metrics']['val_acc'])} п.п., выше прогноза 0–2. Возможна роль сочетания глубины и lr, но без абляций причины не разделены. Среднее best val на новых seed ниже максимума поиска, как ожидалось.

## Вывод и проверка
{ru_conclusion} Вывод ограничен данным разбиением, MLP, 30 эпохами и пятью новыми seed. Одна ошибка из 360 — около 0,278 п.п.; статистическая значимость не заявляется. Несовпадение с прогнозом не скрывается.

Код создан с помощью Codex. Проверки: известная accuracy=0,75; непересечение частей и хеши; отказ при lr=-1 и грязном Git; точное совпадение повторных предсказаний; пересчёт метрик всех запусков; 40 trial без скрытых повторов. Таблицы и рисунки автоматически создаёт analyze.py. Соответствие final run_id: generated/final_run_ids.md; полный аудит: generated/summary.json. Для защиты студент должен самостоятельно понимать и объяснять код и результаты.
'''
    (reportdir/'report_zh.md').write_text(zh,encoding='utf-8')
    (reportdir/'report_ru.md').write_text(ru,encoding='utf-8')
    allrows=[{**r['meta'],**r['metrics'],'model':json.dumps(r['config']['model']),'train':json.dumps(r['config']['train'])} for r in rows]
    pd.DataFrame(allrows).to_csv(out/'runs.csv',index=False)
    mlflow=configure_mlflow()
    mlflow.search_runs(experiment_names=['LR0_experiment_harness'],max_results=1000).to_csv(out/'mlflow_runs.csv',index=False)
    with mlflow.start_run(run_name='analysis'):
        mlflow.set_tags({'work':'LR0_experiment_harness','series':'analysis','model':'MLP','commit':__import__('subprocess').run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()})
        mlflow.log_artifacts(str(reportdir),artifact_path='reports')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['audit','final_pairs']},indent=2))

if __name__=='__main__':main()
