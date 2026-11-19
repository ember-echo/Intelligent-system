import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
if ROOT.name == "_work":
    ROOT = ROOT.parent / "LR4_tool_agent"
OUT = ROOT / "reports"
MODES = ("react", "plan")
NAMES = ("ReAct", "Plan-then-execute")


def read(folder, name):
    return json.loads((folder/name).read_text(encoding="utf-8"))


def report(lang, source, narratives, metadata):
    doc = Document()
    ru = lang == "ru"
    def tr(a,b):
        return a if ru else b
    section = doc.sections[0]
    section.page_width,section.page_height=Mm(210),Mm(297)
    section.top_margin,section.bottom_margin=Mm(17),Mm(16)
    section.left_margin=section.right_margin=Mm(18)
    for name,size,after,before in (("Normal",11,7,0),("Title",22,12,0),("Heading 1",15,6,10),("Heading 2",11,4,9)):
        s=doc.styles[name]
        s.font.name="Calibri"
        s.font.size=Pt(size)
        s.font.color.rgb=RGBColor(0,0,0)
        s.paragraph_format.line_spacing=1.10
        s.paragraph_format.space_after=Pt(after)
        s.paragraph_format.space_before=Pt(before)
        for border in s.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    def p(a,b):
        return doc.add_paragraph(tr(a,b))
    def heading(a,b):
        return doc.add_heading(tr(a,b),1)
    def table(headers, rows, widths):
        tab=doc.add_table(rows=0,cols=len(headers))
        tab.style="Table Grid"
        tab.autofit=False
        for column,width in zip(tab.columns,widths):
            column.width=Mm(width)
        for i, values in enumerate((headers,*rows)):
            for j,(cell,value) in enumerate(zip(tab.add_row().cells,values)):
                cell.text=str(value)
                cell.width=Mm(widths[j])
                para=cell.paragraphs[0]
                para.paragraph_format.space_before=para.paragraph_format.space_after=Pt(3)
                if j:
                    para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size=Pt(9.5)
                    run.bold=i==0
                if i==0:
                    fill=OxmlElement("w:shd")
                    fill.set(qn("w:fill"),"E2EBF2")
                    cell._tc.get_or_add_tcPr().append(fill)
    def value(group):
        n=group["n"]
        return f"{group['success']}/{n} ({100*group['success']/n:.1f}%)" if n else "—"
    hold=source["holdout"]
    dev=source["dev"]
    data=hold["modes"]
    success=[value(data[m]["groups"]["all"]) for m in MODES]
    doc.add_paragraph(tr("Лабораторная работа 4 Агент с инструментами","实验 4 工具调用 Agent"),"Title")
    p(f"На 30 отложенных задачах AgentDojo ReAct решил {success[0]}, а фиксированный план — {success[1]}. Цель работы — выяснить, когда выбор следующего инструмента по результату предыдущего шага оправдывает дополнительные вызовы и контекст.",
      f"在 AgentDojo 的 30 道保留任务上，ReAct 成功 {success[0]}，固定计划成功 {success[1]}。实验检验根据工具返回选择下一步，在哪些情况下值得付出额外的调用与上下文成本。")
    heading("Условия и прогноз","实验条件与预测")
    p("В стенде workspace v1.2.2 доступны 24 имитированных инструмента почты, контактов, календаря и файлов. Использован Qwen3-4B Q4_K_M из ЛР1, температура 0, контекст 16 384, выход до 1024 токенов за вызов. Среда перед каждым эпизодом восстанавливается. На эпизод выделено до 15 обращений к модели и 15 попыток вызова инструментов.",
      "workspace v1.2.2 提供 24 个模拟邮件、联系人、日历和文件工具。沿用 LR1 的 Qwen3-4B Q4_K_M，温度 0，上下文 16,384，每次最多输出 1024 token。每回合重置环境。每回合最多 15 次模型调用和 15 次工具尝试。")
    p("ReAct получает все описания инструментов на каждом шаге и выбирает следующее действие. В режиме плана первый вызов заранее фиксирует до восьми шагов. На каждом шаге модель видит только разрешённый инструмент, а завершает эпизод отдельным текстовым ответом.",
      "ReAct 每步看到所有工具说明，按观察结果选择下一步。计划模式先固定最多八个工具步骤，执行时每步只提供该工具，最后单独生成文字答复。")
    p("До итоговой серии ожидалось, что короткие задачи с явным инструментом доступны обоим режимам. На задачах, где действие зависит от добытых данных, ожидалось преимущество ReAct. Ожидалось, что медиана входных токенов плана будет ниже на 20–50 % за счёт сокращённого набора инструментов.",
      "正式运行前预计，工具明确的短任务两种模式都可完成，需要根据中间数据改变动作的任务更有利于 ReAct。计划模式每步提供的工具较少，预计输入 token 中位数低 20%–50%。")
    p(f"На 10 задачах dev получено {value(dev['modes']['react']['groups']['all'])} для ReAct и {value(dev['modes']['plan']['groups']['all'])} для плана. Затем конфигурация и прогноз зафиксированы в коммите {metadata['commit'][:12]}. На holdout выполнено 60 эпизодов, по одному на режим и задачу, с чередованием порядка.",
      f"10 道开发题上，ReAct 为 {value(dev['modes']['react']['groups']['all'])}，计划为 {value(dev['modes']['plan']['groups']['all'])}。配置和预测固定在提交 {metadata['commit'][:12]}。保留集共运行 60 回合，每题每模式一次，交替执行顺序。")
    heading("Успешность на holdout","保留集成功率")
    groups=[("all",tr("Все задачи","全部")),("one",tr("Один эталонный вызов","标准单调用")),
            ("several",tr("Несколько эталонных вызовов","标准多调用")),
            ("read",tr("Чтение","只读")),("write",tr("Изменение среды","更改状态"))]
    table([tr("Группа","分组"),*NAMES],[(label,*[value(data[m]["groups"][k]) for m in MODES]) for k,label in groups],[60,48,66])
    doc.add_page_break()
    heading("Попарные исходы и причины ошибок","配对结果与失败原因")
    keys=[("both",tr("Оба режима","两模式均成功")),("react_only",tr("Только ReAct","仅 ReAct")),
          ("plan_only",tr("Только план","仅计划")),("neither",tr("Ни один","两模式均失败"))]
    table([tr("Исход","结果"),tr("Задач из 30","题数／30")],
          [(label,hold["paired"][k]) for k,label in keys],[105,69])
    causes={"model_error":tr("Ошибка запроса к модели","模型请求错误"),
            "context_limit":tr("Лимит контекста","上下文超限"),
            "max_calls":tr("Лимит вызовов модели","模型调用上限"),
            "max_tools":tr("Лимит инструментов","工具调用上限"),
            "plan_invalid":tr("Невалидный план","无效计划"),
            "tool_error":tr("Ошибка инструмента","工具错误"),
            "plan_step_error":tr("Пропущенный шаг плана","跳过计划步骤"),
            "wrong_answer_or_state":tr("Неверный ответ или состояние","回答或状态错误")}
    reason_keys=set(data["react"]["fail"])|set(data["plan"]["fail"])
    heading("Классификация неудачных эпизодов","失败分布")
    table([tr("Причина","原因"),*NAMES],
          [(causes[k],data["react"]["fail"].get(k,0),data["plan"]["fail"].get(k,0)) for k in causes if k in reason_keys],
          [68,43,63])
    heading("Стоимость эпизода","每回合成本")
    p("В ячейках указаны медиана и диапазон. Входные токены суммируются по всем обращениям, включая повторно переданные историю и описание инструментов. Время включает токенизацию, генерацию и исполнение инструментов.",
      "单元格为中位数及最小–最大范围。输入 token 累计所有模型请求，包括重复传入的历史和工具说明。耗时包含分词、生成和工具执行。")
    def stat(sub,key):
        if not sub:
            return "—"
        v=sub[key]
        decimals=1 if key=="seconds" else 0
        middle = max(decimals, int(v["median"] % 1 != 0))
        return f"{v['median']:,.{middle}f}\n{v['min']:,.{decimals}f}–{v['max']:,.{decimals}f}".replace(","," ")
    names=("model_calls","tool_calls","input_tokens","output_tokens","seconds")
    state=["all","success","failure"]
    captions=[tr("Все","全部"),tr("Успех","成功"),tr("Неудача","失败")]
    table([tr("Режим и исход","模式与结果"),tr("Модель","模型"),tr("Вызовы","工具"),tr("Вход","输入"),tr("Выход","输出"),tr("Время с","耗时 s")],
          [(f"{name} {title}",*[stat(data[m]["cost"][group],k) for k in names])
           for m,name in zip(MODES,NAMES) for group,title in zip(state,captions)],[49,20,20,29,28,28])
    doc.add_page_break()
    heading("Разбор эпизодов","四个案例")
    for example in narratives["examples"]:
        doc.add_heading(tr(example["title_ru"],example["title_zh"]),2)
        p(example["ru"],example["zh"])
    heading("Вывод и воспроизводимость","结论与复现")
    p(narratives["conclusion_ru"],narratives["conclusion_zh"])
    p("Код подготовлен при помощи Codex. Сценарными тестами проверены неверные имена, обязательные аргументы, границы числа вызовов и ограничения фиксированного плана. Контрольный прогон эталонных вызовов дал 36/40. Метрики пересчитаны из автоматически сохранённых эпизодов. Источники результатов: runs/holdout/results.jsonl и reports/summary.json.",
      "代码由 Codex 协助。脚本化测试检查了错误工具名、缺少必填参数、调用上限和固定计划限制。指标从自动保存的逐回合结果重算。数据见 runs/holdout/results.jsonl 与 reports/summary.json。")
    doc.save(OUT/tr("LR4_俄文报告.docx","LR4_中文报告.docx"))


if __name__=="__main__":
    data=read(OUT,"summary.json")
    note=read(OUT,"narratives.json")
    meta=read(ROOT/"runs/holdout","provenance.json")
    for lang in ("ru","zh"):
        report(lang,data,note,meta)


