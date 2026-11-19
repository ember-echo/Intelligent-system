"""Render the already-computed LR0 results into three-page PDFs, without training."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parent
S=json.loads((ROOT/'reports/generated/summary.json').read_text(encoding='utf-8'))
rows=[]
for p in (ROOT/'runs').glob('*/metadata.json'):
    m=json.loads(p.read_text(encoding='utf-8'))
    if m['series']=='base_seeds':rows.append((m,json.loads((p.parent/'metrics.json').read_text(encoding='utf-8'))))
rows.sort(key=lambda r:r[0]['seed'])
selection=json.loads((ROOT/'configs/selection.json').read_text(encoding='utf-8'))
trials=json.loads((ROOT/'reports/generated/importance.json').read_text(encoding='utf-8'))
pdfmetrics.registerFont(TTFont('RU','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('RUbold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('ZH','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('ZHbold','C:/Windows/Fonts/msyhbd.ttc'))
NAVY=colors.HexColor('#17344a');TEAL=colors.HexColor('#147d92');PALE=colors.HexColor('#edf4f6')
WIDTH=A4[0]-86
def pct(v):return f'{100*v:.3f}'

def render(lang):
    zh=lang=='zh';font='ZH' if zh else 'RU';bold=font+'bold'
    body=ParagraphStyle('body',fontName=font,fontSize=9 if zh else 9.2,leading=14 if zh else 13,spaceAfter=7,wordWrap='CJK' if zh else 'LTR',textColor=colors.HexColor('#233541'))
    heading=ParagraphStyle('heading',parent=body,fontName=bold,fontSize=13,leading=19,spaceBefore=7,spaceAfter=9,textColor=NAVY)
    title=ParagraphStyle('title',parent=heading,fontSize=21,leading=29,spaceAfter=13)
    small=ParagraphStyle('small',parent=body,fontSize=7.1,leading=10,spaceAfter=5)
    tstyle=ParagraphStyle('cell',parent=body,fontSize=7.1,leading=10,spaceAfter=0)
    story=[]
    def p(text,style=body):
        text=text.replace('∈',' in ').replace('–','-').replace('—','-').replace('‑','-')
        return Paragraph(escape(text).replace('\n','<br/>'),style)
    def add(text,style=body):story.append(p(text,style))
    def table(headers,data,widths):
        vals=[[p(str(v),tstyle) for v in headers]]+[[p(str(v),tstyle) for v in row] for row in data]
        tab=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LINEBELOW',(0,0),(-1,0),1,TEAL),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#d4dfe5')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
        story.extend([tab,Spacer(1,9)])
    def pagebreak():story.append(PageBreak())
    add('LR0_experiment_harness',title)
    add('研究问题：调参带来的提升，是否大于随机种子导致的波动？' if zh else 'Вопрос: превышает ли эффект подбора гиперпараметров случайный разброс?',heading)
    add('预测在正式运行前提交：基础 val 约 94–97%，五个 seed 极差约 0.3–1.5 个百分点；搜索最好值可能提高 0–2 个百分点，新 seed 与 test 上优势可能缩小。完整预测见 forecast.md。' if zh else 'Прогноз закоммичен до итоговых запусков: базовая accuracy на val 94–97%, размах по seed 0,3–1,5 п.п.; лучший trial может дать 0–2 п.п., но преимущество на новых seed и test может уменьшиться. Полный прогноз: forecast.md.')
    add('本次为固定协议独立重跑，所有结果重新生成。预测参考既有经验；test 划分在旧版本中已评估，并非全新未见数据；未据旧 test 更改搜索或选择规则。' if zh else 'Независимый повтор фиксированного протокола; все результаты получены заново. Прогноз учитывает прежний опыт; test уже оценивался ранее и не считается совершенно новым. Правило выбора и поиск по прежнему test не менялись.',small)
    add('数据与实现' if zh else 'Данные и реализация',heading)
    add('load_digits：1797 张 8×8 图像、10 类。按作业要求分层划分，split_seed=0，train/val/test=1077/360/360，特征除以 16。MLP：64→隐藏层（ReLU+dropout）→10；Adam、交叉熵、固定 30 轮。基础配置：1×64，dropout=0，lr=0.001，weight_decay=0，batch_size=64。' if zh else 'load_digits: 1797 изображений 8×8, 10 классов. Стратифицированное разбиение, split_seed=0: train/val/test=1077/360/360; признаки делятся на 16. MLP: 64 → скрытые слои ReLU+dropout → 10; Adam, cross-entropy, 30 эпох. База: 1×64, dropout=0, lr=0.001, weight_decay=0, batch_size=64.')
    v=rows[0][0]['versions']
    add('CPU / 1 thread; Python 3.11.9; '+', '.join(k+' '+v[k] for k in ['torch','scikit-learn','optuna','mlflow'])+'.',small)
    add('1｜复现检查和 seed 波动' if zh else '1. Повторяемость и разброс',heading)
    add('seed 0 的两次预测文件逐字节一致，训练/验证指标完全一致。真实 epoch_s 不同，所以 history.csv 整体不一致：作业中“记录耗时”与“完整文件相同”不能同时满足，未伪造时间。下面的时间只含训练和验证计算，不含初始化、写文件和 MLflow。' if zh else 'Два запуска seed 0 дали побайтно одинаковые predictions.csv и точно одинаковые метрики истории. Реальные epoch_s различаются, поэтому history.csv целиком не совпадает: запись реального времени несовместима с полным совпадением файла. Время не подменялось. Ниже измерены обучение + валидация без подготовки, файлов и MLflow.')
    table(['seed','val, %','time, s','run_id'],[[m['seed'],pct(q['val_acc']),f"{q['train_time_s']:.3f}",m['run_id']] for m,q in rows],[40,67,67,WIDTH-174])
    add(('每次 val 分母 360。最小/中位/最大：' if zh else 'Знаменатель val: 360. Минимум/медиана/максимум: ')+f"{pct(S['base_val_min'])}/{pct(S['base_val_median'])}/{pct(S['base_val_max'])}%; "+('极差 ' if zh else 'размах ')+pct(S['base_val_range'])+' pp.')
    st=S['base_seed1_to_seed2']
    add(f"seed 1→2：纠正 {st['fixed']}，破坏 {st['broken']}，都正确 {st['both_right']}，都错误 {st['both_wrong']}，合计 360 张。准确率接近不代表错误相同。" if zh else f"seed 1→2: исправлено {st['fixed']}, испорчено {st['broken']}, оба верны {st['both_right']}, оба ошиблись {st['both_wrong']} из 360. Близкая accuracy не означает одинаковые ошибки.")
    pagebreak()
    add('2｜40 次顺序超参数搜索' if zh else '2. Поиск: 40 последовательных trial',title)
    add('TPE(seed=42)，MedianPruner(startup=5,warmup=5)，训练 seed=0。lr∈[1e-4,1e-1]（log），层数 1–3，hidden∈[32,512]（整数/log），dropout∈[0,0.5]，weight_decay∈[1e-6,1e-2]（log），batch_size∈{32,64,128}。目标是第 30 轮 val 准确率。' if zh else 'TPE(seed=42), MedianPruner(startup=5, warmup=5), seed обучения 0. lr∈[1e-4,1e-1] (log), слои 1–3, hidden∈[32,512] (int/log), dropout∈[0,0.5], weight_decay∈[1e-6,1e-2] (log), batch_size∈{32,64,128}. Цель: accuracy на val после 30-й эпохи.')
    add('完成 10 trial 后有计划暂停，再从 SQLite 继续至 40。这不是在 epoch 中强制杀进程的测试；重启不恢复 sampler 随机状态，因此不声称与不中断搜索等价。' if zh else 'После 10 trial — запланированная пауза и продолжение из SQLite до 40. Это не тест аварийного прерывания внутри эпохи. RNG семплера не восстанавливается; эквивалентность непрерывному поиску не заявляется.')
    states=S['search_states']
    add(f"COMPLETE={states.get('COMPLETE',0)}; PRUNED={states.get('PRUNED',0)}; FAIL={states.get('FAIL',0)}. "+('实际轮数：' if zh else 'Выполнено эпох: ')+f"{S['executed_epochs']}/1200; "+('节省 ' if zh else 'сэкономлено ')+f"{S['epochs_saved']} ({100*S['epochs_saved']/1200:.1f}%).")
    story.append(Image(str(ROOT/'reports/generated/search_history.png'),width=WIDTH,height=WIDTH*2.7/7))
    story.append(Image(str(ROOT/'reports/generated/parameter_importance.png'),width=WIDTH,height=WIDTH*2.4/7))
    add(f"实测搜索计算时间 {S['measured_search_compute_s']:.3f} s；按各 trial 已执行轮平均时间外推，估计节省 {S['estimated_saved_compute_s']:.3f} s。这是估计，未实测无剪枝对照。剪枝点分数不是完整 30 轮分数；fANOVA 只用 COMPLETE，不是因果结论。" if zh else f"Измеренное время вычислений: {S['measured_search_compute_s']:.3f} с. Оценка экономии по среднему времени эпох каждого trial: {S['estimated_saved_compute_s']:.3f} с; это не измеренный контроль без прунера. Оценки PRUNED не равны итогам 30 эпох; fANOVA использует только COMPLETE и не доказывает причинность.",small)
    # Read selected config without adding a YAML dependency to the renderer.
    trialrows=__import__('csv').DictReader((ROOT/'reports/generated/trials.csv').open(encoding='utf-8',newline=''))
    best=next(r for r in trialrows if int(r['number'])==S['search_best_trial'])
    param_text=', '.join(k.removeprefix('params_')+'='+str(val) for k,val in best.items() if k.startswith('params_'))
    add((f"最佳 trial #{S['search_best_trial']}，val={pct(S['search_best_val'])}%。" if zh else f"Лучший trial #{S['search_best_trial']}: val={pct(S['search_best_val'])}%. ")+param_text,small)
    add('只按 val 选配置，best.yaml 与 selection.json 已在 test 前独立提交。' if zh else 'Выбор только по val; best.yaml и selection.json зафиксированы отдельным коммитом до test.',small)
    pagebreak()
    add('3｜新 seed 上的最终比较' if zh else '3. Сравнение на новых seed',title)
    add('seed 101–105，各训练 base/best 一次，共 10 次正式运行，采用同一代码提交。每次 val/test 分母为 360；下表差值均为 best−base（百分点）。没有根据 test 重新调参或替换运行。' if zh else 'seed 101–105: по одному запуску base/best, всего 10, один коммит. Знаменатель val/test каждого запуска — 360. Разности: best−base, п.п. После test параметры не менялись и запуски не подбирались.')
    pairs=S['final_pairs']
    table(['seed','base\nval %','best\nval %','Δval\npp','base\ntest %','best\ntest %','Δtest\npp','fix /\nbreak'],[
        [r['seed'],pct(r['base_val']),pct(r['best_val']),pct(r['delta_val']),pct(r['base_test']),pct(r['best_test']),pct(r['delta_test']),f"{r['fixed']}/{r['broken']}"] for r in pairs],[37,65,65,63,65,65,63,WIDTH-423])
    add(f"Δval 中位数 {pct(S['delta_val_median'])} pp，范围 [{pct(S['delta_val_min'])}, {pct(S['delta_val_max'])}]；Δtest 中位数 {pct(S['delta_test_median'])} pp，范围 [{pct(S['delta_test_min'])}, {pct(S['delta_test_max'])}]。" if zh else f"Медиана Δval {pct(S['delta_val_median'])} п.п., диапазон [{pct(S['delta_val_min'])}, {pct(S['delta_val_max'])}]; медиана Δtest {pct(S['delta_test_median'])} п.п., диапазон [{pct(S['delta_test_min'])}, {pct(S['delta_test_max'])}].")
    add(f"五组 test 共纠正 {S['fixed_test_total']}、破坏 {S['broken_test_total']}，分母 1800 个 seed-图像观察；这是相同 360 张图像的五次评估，不是 1800 张独立图像。" if zh else f"На test исправлено {S['fixed_test_total']}, испорчено {S['broken_test_total']} из 1800 наблюдений «seed–изображение». Это пять оценок тех же 360 изображений, не 1800 независимых примеров.")
    table(['estimate','accuracy %'],[
        ['Search best / val',pct(S['search_best_val'])],['New seeds / best mean val',pct(S['best_new_seeds_mean_val'])],['New seeds / best mean test',pct(S['best_new_seeds_mean_test'])]],[WIDTH-100,100])
    add('搜索选优、训练随机性和样本差异都可能影响以上三个值，不能仅凭它们把变化全归因于选择偏差。所有负面或未提升的结果均保留。' if zh else 'На три оценки могут одновременно влиять отбор, случайность обучения и различие выборок; эти числа сами по себе не разделяют причины. Отрицательные результаты не скрываются.')
    add('结论及局限' if zh else 'Вывод и границы',heading)
    if S['robust_by_course_rule']:
        add('五个 test 差值均为正且都大于基础 val 极差，按课程判据，本次调参改善稳定。' if zh else 'Все пять разностей на test положительны и превышают базовый размах на val; по критерию работы улучшение устойчиво.')
    else:
        add('并非五个 test 差值都大于基础 val 极差，因此按课程判据，不能认定调参改善稳定。即使五次都为正，若效应小于随机波动，也不能宣称已证实普遍提升。' if zh else 'Не все пять разностей на test превышают базовый размах на val. Критерий устойчивого улучшения не выполнен. Даже положительный, но малый эффект нельзя объявлять доказанным общим улучшением.')
    add('原预测与实际结果应结合第一页的基础波动、第二页的搜索最好值及本页的新 seed 平均值比较。不同 seed 和数据样本都可能影响差异；没有消融实验，不能分离各因素的贡献。' if zh else 'Прогноз сопоставляется с базовым разбросом, максимумом поиска и средним на новых seed. Случайность обучения и состав выборок влияют одновременно; без абляций вклад факторов не разделяется.')
    add('结论仅限本划分、MLP、30 轮与五个新 seed。一张 test 图像约 0.278 pp，未声称统计显著性。对照实测结果核对原预测；未证实的预期不改写为事实。' if zh else 'Вывод ограничен разбиением, MLP, 30 эпохами и пятью seed. Одна ошибка из 360 — около 0,278 п.п.; статистическая значимость не заявляется. Исходный прогноз сопоставляется с результатами, а не переписывается.')
    add('生成与验证' if zh else 'Кодогенерация и проверка',heading)
    add('代码由 Codex 协助生成。已通过 0.75 准确率示例、划分/哈希、负学习率和脏 Git 拒绝测试；独立重算全部正式运行指标，核对复现、40 trial 与中断继续记录。表图来自 analyze.py；完整 run_id 对照和审计见 reports/generated。学生仍需自己理解代码并完成答辩。' if zh else 'Код создан с помощью Codex. Проверены accuracy=0,75, разбиение/хеши, отказ при lr=-1 и грязном Git. Метрики всех итоговых запусков пересчитаны; проверены повторяемость, 40 trial и возобновление. Таблицы и графики: analyze.py; run_id и аудит: reports/generated. Студент должен понимать код и самостоятельно защищать работу.',small)
    dest=ROOT/f'reports/report_{lang}.pdf'
    def decorate(canvas,doc):
        canvas.setStrokeColor(TEAL);canvas.setLineWidth(0.8);canvas.line(43,A4[1]-32,A4[0]-43,A4[1]-32)
        canvas.setFont(font,7);canvas.setFillColor(NAVY)
        canvas.drawString(43,24,'LR0 | load_digits | CPU, 1 thread | 40 trials')
        canvas.drawRightString(A4[0]-43,24,f'{doc.page} / 3')
    doc=SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=43,bottomMargin=40,
                          title='LR0: reproducible experiment',author='Codex-assisted laboratory report')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(dest)

if __name__=='__main__':
    render('zh');render('ru')
