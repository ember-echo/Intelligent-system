import sys
import json
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
if ROOT.name == "_work":
    ROOT = ROOT.parent / "LR3_rag_retrieval"
OUT = ROOT / "reports"
TEMPLATE = Path("C:/Users/FF/xwechat_files/wxid_0g2139p57mad22_05c1/msg/file/2026-11/Титульный лист(1).docx")
NAMES = {"dense": "Dense", "bm25": "BM25", "rrf": "RRF", "ce": "RRF + CE"}


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def assign_text(paragraph, value, use_run=0):
    for i, run in enumerate(paragraph.runs):
        run.text = value if i == use_run else ''


def add_cover(doc, number, topic, template_path):
    cover = Document(template_path)
    paragraphs = cover.paragraphs
    assign_text(paragraphs[13], f'ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ № {number}')
    paragraphs[15].runs[0].text = 'Интеллектуальные системы'
    assign_text(cover.tables[0].cell(1, 0).paragraphs[0], topic)
    for table in cover.tables[1:]:
        for cell in table.rows[1].cells:
            for paragraph in cell.paragraphs:
                assign_text(paragraph, '')
    paragraphs[-1]._p.getparent().remove(paragraphs[-1]._p)
    for blank in paragraphs[27:31]:
        blank._p.getparent().remove(blank._p)
    for table in cover.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.0
    city = paragraphs[-2]
    city._p.get_or_add_pPr().append(deepcopy(cover.sections[0]._sectPr))

    cover_style = doc.styles.add_style('Lab Cover', WD_STYLE_TYPE.PARAGRAPH)
    defaults = cover.styles.element.find(qn('w:docDefaults'))
    for prop in ('pPr', 'rPr'):
        data = defaults.find(qn(f'w:{prop}Default')).find(qn(f'w:{prop}'))
        cover_style.element.append(deepcopy(data))
    cover_style.font.name = 'Calibri'

    table_style = deepcopy(cover.styles.element.xpath('./w:style[@w:styleId="a9"]')[0])
    table_style.set(qn('w:styleId'), 'LabCoverTable')
    table_style.find(qn('w:name')).set(qn('w:val'), 'Lab Cover Table')
    table_style.find(qn('w:basedOn')).set(qn('w:val'), 'TableNormal')
    doc.styles.element.append(table_style)

    target = doc.element.body
    first_body = target[0]
    cover_ids = {}
    for relationship in cover.part.rels.values():
        if relationship.reltype.endswith('/image'):
            image_part = doc.part.package.get_or_add_image_part(
                __import__('io').BytesIO(relationship.target_part.blob)
            )
            cover_ids[relationship.rId] = doc.part.relate_to(image_part, relationship.reltype)

    next_id = doc.part.next_id
    for child in cover.element.body:
        if child.tag == qn('w:sectPr'):
            continue
        node = deepcopy(child)
        for paragraph in node.iter(qn('w:p')):
            props = paragraph.get_or_add_pPr()
            props.get_or_add_pStyle().val = cover_style.style_id
        for style in node.iter(qn('w:tblStyle')):
            style.set(qn('w:val'), 'LabCoverTable')
        for blip in node.iter(qn('a:blip')):
            key = qn('r:embed')
            blip.set(key, cover_ids[blip.get(key)])
        for drawing in node.iter(qn('wp:docPr')):
            drawing.set('id', str(next_id))
            next_id += 1
        first_body.addprevious(node)


def cover(doc):
    import hashlib
    template_path = Path(sys.argv[1]) if len(sys.argv) > 1 else TEMPLATE
    assert hashlib.sha256(template_path.read_bytes()).hexdigest() == "729c2b8bc99b0e1b0dc07c952774e1362acc2e9ea8d9fa6202272DB1B5EFFBC1".lower()
    add_cover(doc, 3, "Улучшение поиска в RAG с BM25, RRF и cross-encoder", template_path)


def fmt(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100*x:.1f}%"


def report(lang, data, traces):
    ru = lang == "ru"
    tr = lambda a,b: a if ru else b
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = Mm(16)
    sec.top_margin = sec.bottom_margin = Mm(15)
    for name, size in (("Normal", 10.5), ("Title", 22), ("Heading 1", 14), ("Heading 2", 11)):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0,0,0)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.line_spacing = 1.10
    for border in doc.styles["Title"].element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    p = doc.add_paragraph
    h = lambda x,y: doc.add_heading(tr(x,y), 1)
    title = tr("Лабораторная работа 3 Улучшение поиска в RAG", "实验 3 改进 RAG 检索")
    alt = data["selected"]
    dev, hold = data["dev"], data["holdout"]
    p(title, "Title")
    p(tr(f"По заранее заданному правилу выбран {NAMES[alt]}. На holdout из 210 утверждений Accuracy составила {pct(hold['generation'][alt]['accuracy'])} против {pct(hold['generation']['dense']['accuracy'])} у Dense.",
         f"按预先冻结的规则选中 {NAMES[alt]}。在 210 条保留样本上，准确率为 {pct(hold['generation'][alt]['accuracy'])}，Dense 为 {pct(hold['generation']['dense']['accuracy'])}。"))
    h("Прогноз и условия", "预测与实验条件")
    p(tr("Ожидалось, что BM25 поможет на редких маркерах, Dense — при перефразировании, RRF повысит полноту, а переранжирование улучшит MRR и слегка повысит Accuracy. Сохранены корпус и разбиение ЛР2, Qwen3-4B, промпт, JSON Schema и бюджет 2048 токенов. Генератор получает до трёх документов.",
         "预计 BM25 擅长罕见术语，Dense 擅长改写，RRF 提高召回率，重排提高 MRR 并小幅提高准确率。沿用 LR2 语料、划分、Qwen3-4B、提示词、JSON Schema 和 2048 token 预算。生成器最多接收三篇文档。"))
    p(tr("Dense использует нормированные векторы MiniLM. BM25 индексирует заголовки и аннотации с k1 = 1,2 и b = 0,75. RRF объединяет два top-20 с константой 60. Cross-encoder переставляет 20 кандидатов RRF, ограничивая пару 512 токенами. При равенстве оценок первым идёт меньший doc_id.",
         "Dense 使用归一化 MiniLM 向量。BM25 索引标题和摘要，k1=1.2、b=0.75。RRF 合并两份 top-20，常数为 60。Cross-encoder 用最多 512 token 给原有 20 个候选重排。同分时 doc_id 较小者优先。"))
    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Light Shading Accent 1"
        for c,v in zip(t.rows[0].cells, headers):
            c.text = str(v)
        for row in rows:
            for c,v in zip(t.add_row().cells,row):
                c.text = str(v)
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(2)
                    para.paragraph_format.space_before = Pt(1)
                    for run in para.runs:
                        run.font.size = Pt(9)
        doc.add_paragraph().paragraph_format.space_after = Pt(0)

    h("Поиск и дополнительность на dev", "开发集检索与互补性")
    p(tr("Dev Q+ = 60. Первые пять результатов Dense совпали с ЛР2 на всех 90 запросах. Recall@20 RRF и RRF + CE совпадает, поскольку набор кандидатов сохраняется.",
         "开发集 Q+ 共 60 条。全部 90 条查询的 Dense 前五名均与 LR2 一致。RRF 与重排使用同样的候选集合，Recall@20 相同。"))
    table([tr("Метод","方案"),"Hit@3","Recall@3","MRR@3","Recall@20"],
          [[NAMES[m]]+[fmt(v[k]) for k in ("hit@3","recall@3","mrr@3","recall@20")] for m,v in dev["retrieval"].items()])
    groups = [("both",tr("Оба","两者都有")),("dense_only",tr("Только Dense","仅 Dense")),
              ("bm25_only",tr("Только BM25","仅 BM25")),("neither",tr("Ни один","两者均无"))]
    table([tr("Группа","组别"),tr("@3 всего / RRF@3","@3 总数 / RRF@3 命中"),tr("@20 всего / RRF@3","@20 总数 / RRF@3 命中")],
          [[name]+[f"{dev['complement'][k][g]['n']} / {dev['complement'][k][g]['rrf_hit3']}" for k in ("3","20")] for g,name in groups])

    h("Вывод и воспроизводимость", "结论与复现")
    direction = hold["generation"][alt]["accuracy"] - hold["generation"]["dense"]["accuracy"]
    advice = NAMES[alt] if direction > 0 else "Dense"
    p(tr(f"Рекомендация по Accuracy на holdout — {advice}. Вывод ограничен SciFact, данным разбиением, одной моделью и фиксированным бюджетом. Полные ранги, сообщения и ответы сохранены в runs и reports/traces.json.",
         f"按保留集准确率建议 {advice}。结论适用于 SciFact、本次划分、固定模型及上下文预算。完整排名、消息和回答见 runs 与 reports/traces.json。"))
    dv, hv = (s["generation"] for s in (dev, hold))
    delta_dev, delta_hold = ((s[alt]["accuracy"] - s["dense"]["accuracy"])*100 for s in (dv, hv))
    p(tr(f"Разница Accuracy составила {delta_dev:+.1f} п.п. на dev и {delta_hold:+.1f} п.п. на holdout. На holdout полнота NOINFO у альтернативы {pct(hv[alt]['class_recall']['NOINFO'])} против {pct(hv['dense']['class_recall']['NOINFO'])} у Dense при большей доле полного обоснования.",
         f"准确率差异在开发集为 {delta_dev:+.1f} 个百分点，在保留集为 {delta_hold:+.1f} 个百分点。保留集中，重排传入更多完整证据，但 NOINFO 召回率由 {pct(hv['dense']['class_recall']['NOINFO'])} 降为 {pct(hv[alt]['class_recall']['NOINFO'])}。"))
    p(tr("Код подготовлен с Codex. Проверены ручной BM25, пример RRF, сохранение кандидатов CE, совпадение с LR2 и пересчёт из результатов.",
         "代码由 Codex 协助整理。核对了 BM25 手算、RRF 示例、CE 候选集合、与 LR2 的检索一致性及从保存结果重算。"))

    doc.add_page_break()
    h("Ответы на dev и стоимость поиска", "开发集回答与检索耗时")
    p(tr("На всех 90 утверждениях использованы 90 совместимых ответов Dense из ЛР2 и 180 новых ответов RRF и RRF + CE. Верные ссылки и переданное обоснование считаются на 60 утверждениях Q+. Ошибки и невалидный JSON входят в знаменатель.",
         "90 条样本复用了 LR2 中条件一致的 Dense 回答，对 RRF 和 RRF + CE 新生成 180 次。正确引用和传入证据按 60 条 Q+ 样本计算。错误和无效 JSON 均计入分母。"))
    def generated(split):
        return [[NAMES[m], pct(v["accuracy"]),pct(v["citation"]),pct(v["context"]),
                 " / ".join(pct(v["class_recall"][key]) for key in ("SUPPORT","CONTRADICT","NOINFO"))]
                for m,v in split["generation"].items()]
    columns = [tr("Метод","方案"),"Accuracy",tr("Ссылки","正确引用"),tr("Обоснование","传入证据"),"Recall S / C / N"]
    table(columns,generated(dev))
    for m,v in dev["transitions"].items():
        p(tr(f"{NAMES[m]} против Dense: исправлено {v['fixed']}, испорчено {v['broken']} из {v['n']}.",
             f"{NAMES[m]} 对比 Dense：{v['n']} 条中修正 {v['fixed']} 条、由对变错 {v['broken']} 条。"))
    p(tr("Время CPU на 90 запросах. Интервалы исключают загрузку, построение индекса, прогрев и запись. Dense включает токенизацию и кодирование запроса. RRF и CE указаны отдельно и вместе с предыдущими этапами.",
         "CPU 计时共 90 条查询，排除模型加载、建索引、预热和写盘。Dense 包含查询分词及编码。RRF 与 CE 同时列出单阶段和端到端时间。"))
    table([tr("Этап","阶段"),tr("Медиана мс","中位数 ms"),tr("Мин–макс мс","最小–最大 ms")],
          [[k,f"{v['median']:.3f}",f"{v['low']:.3f}–{v['high']:.3f}"] for k,v in dev["timing_ms"].items()])
    h("Отложенная проверка", "保留集验证")
    p(tr(f"Вариант {NAMES[alt]} зафиксирован по MRR@3 на dev. При равенстве предпочитается больший Recall@3, далее более простой метод. Holdout Q+ = {hold['qplus']}, генерация = 210 на вариант.",
         f"按开发集 MRR@3 冻结 {NAMES[alt]}。同分先看 Recall@3，再选较简单的方案。保留集 Q+ 为 {hold['qplus']}，每个方案生成 210 次。"))
    table([tr("Серия","数据集"),tr("Метод","方案"),"Hit@3","Recall@3","MRR@3","Recall@20"],
          [[split,NAMES[m]]+[fmt(sub["retrieval"][m][k]) for k in ("hit@3","recall@3","mrr@3","recall@20")]
           for split,sub in (("dev",dev),("holdout",hold)) for m in ("dense",alt)])
    table(columns, generated(hold))
    v = hold["transitions"][alt]
    counts = hold["generation"]["dense"]["counts"]
    p(tr(f"Holdout: исправлено {v['fixed']} и испорчено {v['broken']} из 210. S/C/N = {counts['SUPPORT']}/{counts['CONTRADICT']}/{counts['NOINFO']} против 30/30/30 на dev.",
         f"保留集 210 条中修正 {v['fixed']} 条、由对变错 {v['broken']} 条。S/C/N 样本总数为 {counts['SUPPORT']}/{counts['CONTRADICT']}/{counts['NOINFO']}，开发集为 30/30/30。"))

    doc.add_page_break()
    h("Четыре трассы", "四条检索与回答轨迹")
    tags = {"bm25_only":("Находка BM25","仅 BM25 命中"),"dense_only":("Находка Dense","仅 Dense 命中"),
            "rrf_change":("Изменение top-3 при RRF","RRF 改变前三名命中"),"ce_answer":("Изменение ответа после CE","CE 改变回答")}
    for trace in traces:
        i = trace["id"]
        doc.add_heading(f"{tr(*tags[trace['kind']])} · {i}", 2)
        p(trace["claim"])
        all_ranks = {m: {r["doc_id"]: r["rank"] for r in rs} for m,rs in trace["rankings"].items()}
        focus = sorted(trace["evidence"], key=lambda d: (min(r.get(d, 21) for r in all_ranks.values()), int(d)))[:2]
        if not focus:
            focus = [trace["rankings"]["ce"][0]["doc_id"]]
        def ranks(d):
            rows = []
            for mode in NAMES:
                r = next((r for r in trace["rankings"][mode] if r["doc_id"] == d), {})
                rows.append(str(r.get("rank", "–")))
            return "/".join(rows)
        p(tr("Документы и ранги Dense/BM25/RRF/CE. ", "文档排名顺序为 Dense/BM25/RRF/CE。") +
          "  ".join(f"{d}: {ranks(d)}" for d in focus[:4]))
        contrib = []
        for d in focus[:4]:
            pair = [next((x["rank"] for x in trace["rankings"][m] if x["doc_id"] == d),None) for m in ("dense","bm25")]
            contrib.append(f"{d}: "+ " + ".join(f"1/{60+r}" if r else "0" for r in pair))
        p("RRF  " + "  ".join(contrib))
        parts = []
        for m, row in trace["answers"].items():
            raw = row["verdict"]
            passed = ", ".join(f"{d}[{','.join(map(str,row['passed'][d]))}]" for d in focus[:4] if d in row["passed"]) or "–"
            verdict = raw["label"] if raw else row["status"]
            parts.append(f"{NAMES[m]} {passed} → {verdict}")
        p(tr("Номера переданных предложений. ", "列出这些文档实际传入的句子编号。") + "  ".join(parts) +
          tr(f"  Эталон {trace['gold']}.",f"  标准标签 {trace['gold']}。"))
        hit3 = {m: bool(set(trace["evidence"]) & {r["doc_id"] for r in trace["rankings"][m][:3]})
                for m in NAMES}
        stages = [NAMES[m] for m, flag in hit3.items() if flag]
        present = [NAMES[m] for m,row in trace["answers"].items() if row["evidence_present"]]
        missing = tr("ни одного", "无")
        p(tr("Свидетельство в top-3 у ", "前 3 名命中 ") + (", ".join(stages) or missing) +
          tr(". Полное обоснование в контексте у ", "。完整证据进入上下文 ") +
          (", ".join(present) or missing) + tr(".", "。"))
    if ru:
        cover(doc)
    doc.save(OUT / tr("LR3_俄文报告.docx","LR3_中文报告.docx"))


if __name__ == "__main__":
    for language in ("ru","zh"):
        report(language, read("summary.json"), read("traces.json"))




