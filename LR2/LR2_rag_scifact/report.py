import json
from pathlib import Path
from docx import Document
from docx.shared import Mm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"runs/dev"
def load(p):return json.loads(p.read_text(encoding="utf-8"))
s=load(RUN/"summary.json");m=s["metrics"];idx=load(ROOT/"index/passport.json")
rows=[json.loads(x) for x in (RUN/"results.jsonl").read_text(encoding="utf-8").splitlines()]
cases=load(RUN/"error_cases.json")
if cases["context_loss"] is None:
    cases["context_loss"]=next((r for r in rows if r["mode"]=="rag" and r["gold"]!="NOINFO" and r["retrieval"][0] and not r["evidence_present"]),None)
corp={str(r["doc_id"]):r for r in (json.loads(x) for x in (ROOT/"data/corpus.jsonl").read_text(encoding="utf-8").splitlines())}
p=s["transitions"]["only_to_rag3"]
def build(zh):
    d=Document();sec=d.sections[0]
    sec.page_width=Mm(210);sec.page_height=Mm(297)
    sec.top_margin=sec.bottom_margin=Mm(17)
    sec.left_margin=sec.right_margin=Mm(17)
    for name,size,line in [("Title",20,26),("Heading 1",14,18),("Heading 2",11,14),("Normal",10,13)]:
        t=d.styles[name];t.font.name="Arial";t.font.size=Pt(size);t.font.color.rgb=RGBColor(0,0,0)
        t.font.bold=name!="Normal"
        t.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"),"Microsoft YaHei")
        t.paragraph_format.space_before=Pt(0);t.paragraph_format.space_after=Pt(6)
        t.paragraph_format.line_spacing=Pt(line)
    for b in list(d.styles.element.iter(qn("w:pBdr"))):b.getparent().remove(b)
    def add(a,b,sty=None):return d.add_paragraph(a if zh else b,sty)
    def h(a,b,new=False):
        q=add(a,b,"Heading 1");q.paragraph_format.page_break_before=new
    def table(head, body):
        t=d.add_table(rows=0,cols=len(head));t.style="Table Grid"
        for i,row in enumerate([head,*body]):
            rr=t.add_row()
            pr=rr._tr.get_or_add_trPr();pr.append(OxmlElement("w:cantSplit"))
            if i==0:pr.append(OxmlElement("w:tblHeader"))
            for c,v in zip(rr.cells,row):
                c.text=str(v);q=c.paragraphs[0];q.paragraph_format.space_after=Pt(3);q.paragraph_format.space_before=Pt(3)
                q.paragraph_format.line_spacing=Pt(11)
                for r in q.runs:r.bold=i==0;r.font.size=Pt(8.5)
                if i==0:
                    sh=OxmlElement("w:shd");sh.set(qn("w:fill"),"E7EFF4");c._tc.get_or_add_tcPr().append(sh)
        borders=OxmlElement("w:tblBorders")
        for e in ["top","bottom","left","right","insideH","insideV"]:
            el=OxmlElement("w:"+e);el.set(qn("w:val"),"single");el.set(qn("w:sz"),"4");el.set(qn("w:color"),"D9D9D9");borders.append(el)
        t._tbl.tblPr.append(borders)
        d.add_paragraph().paragraph_format.space_after=Pt(0)
    add("实验 2 科学断言核验与检索增强","Лабораторная работа 2 Проверка научных утверждений","Title")
    add(f"在 SciFact 的 90 条开发样本上，无检索正确 {m['only']['correct']} 条，RAG@3 正确 {m['rag3']['correct']} 条。配对比较中，RAG@3 修正了 {p['fixed']} 条，同时使 {p['broken']} 条由对变错。",
        f"На 90 утверждениях SciFact модель без поиска правильно решила {m['only']['correct']} случаев, RAG@3 — {m['rag3']['correct']}. В попарном сравнении RAG@3 исправил {p['fixed']} ответов и ухудшил {p['broken']}.")
    h("数据 方法与预测","Данные метод и прогноз")
    add(f"对 5,183 篇摘要用 all-MiniLM-L6-v2 做均值池化和 L2 归一化，按余弦相似度排序。256-token 编码上限截断了 {idx['truncated']}/5183 篇摘要。生成沿用 LR1 的 Qwen3-4B Q4_K_M，温度 0，输入预算 2,048、输出上限 128，按句编号引用。每类 30 条，210 条保留样本留给 LR3。",
        f"5183 аннотации закодированы all-MiniLM-L6-v2 со средним пулингом и L2-нормализацией. Ранжирование по косинусной близости. Предел 256 токенов затронул {idx['truncated']} аннотацию из 5183. Генерация Qwen3-4B Q4_K_M из ЛР1, температура 0, бюджет входа 2048, выхода 128 токенов. По 30 утверждений каждого класса. Отложенные 210 утверждений оставлены для ЛР3.")
    add("预注册预测：无检索时 CONTRADICT 最难；RAG@3 修正 15–30 条并新增 5–15 条错误；检索指标随 k 不下降，但准确率可能在 k=3 最好。oracle 预期在相同 60 条上更好。",
        "Прогноз: без поиска труднее всего CONTRADICT. RAG@3 исправит 15–30 ответов и ухудшит 5–15. Метрики поиска не снизятся при росте k, но точность может быть максимальной при k=3. Для oracle ожидается преимущество на тех же 60 утверждениях.")
    h("实验一 无检索 RAG 与 oracle","Опыт 1 Без поиска RAG и oracle")
    table(["Mode","Correct / n","SUPPORT","CONTRADICT","NOINFO","Citation / 60"],[
       [n,f"{m[n]['correct']}/{m[n]['n']}",*[f"{m[n]['recall'][l]:.1%}" if l in m[n]["recall"] else "—" for l in ["SUPPORT","CONTRADICT","NOINFO"]],m[n].get("citation_correct","—")] for n in ["only","rag3","oracle"]])
    add("固定在同一批 60 条有证据断言上比较，正确数分别为 " + ", ".join(f"{n} {m[n]['positive_correct']}/60" for n in ["only","rag3","oracle"])+ "。oracle 使用一条金标准证据，仅作诊断。格式不合规的回答计为错误。",
        "На одних и тех же 60 утверждениях с доказательствами: "+", ".join(f"{n} {m[n]['positive_correct']}/60" for n in ["only","rag3","oracle"])+". Oracle получает одно эталонное обоснование в диагностических целях. Некорректный формат считается ошибкой.")
    h("实验二 检索文档数","Опыт 2 Число документов",True)
    table(["k","Hit","Recall","MRR","Evidence / 60","Acc. / 90","Cite / 60","Docs / tokens"],[
       [n[-1],*[f"{m[n][x]:.3f}" for x in ["hit","recall_at_k","mrr"]],m[n]["evidence_present"],m[n]["correct"],m[n]["citation_correct"],f"{m[n]['mean_docs']:.2f} / {m[n]['mean_tokens']:.0f}"] for n in ["rag1","rag3","rag5"]])
    def transitions(field):return "; ".join(f"{k}: +{v['fixed']}/−{v['broken']}" for k,v in s[field].items() if k.startswith("rag3"))
    add("从 k=3 切换的配对变化（修正/退化）。标签：" + transitions("transitions") + "。引用：" + transitions("citation_transitions")+"。",
        "Попарные переходы от k=3 (исправлено/ухудшено). Метки: "+transitions("transitions")+". Ссылки: "+transitions("citation_transitions")+".")
    best=max(["rag1","rag3","rag5"],key=lambda n:m[n]["correct"])
    worst=min(m["only"]["recall"],key=m["only"]["recall"].get)
    add(f"无检索召回最低的类别为 {worst}。实测正确数最高的 RAG 档位是 {best}。Hit 只检查检索列表，引用命中还要求证据句进入上下文、标签正确且模型选中完整依据。",
        f"Минимальная полнота без поиска у класса {worst}. Больше всего правильных ответов у {best}. Hit относится к выдаче поиска. Верная ссылка требует наличия обоснования в контексте, правильной метки и выбора моделью полного набора предложений.")
    h("错误案例与可复核记录","Ошибки и материалы для проверки",True)
    labels=[("retrieval_miss","检索未命中","Промах поиска"),("context_loss","证据未完整进入上下文","Обоснование вне контекста"),("generation_error","证据已传入但回答有误","Ошибка при доступном обосновании"),("noinfo_error","证据不足样本判错","Ошибка на NOINFO")]
    for key,a,b in labels:
        add(a,b,"Heading 2")
        r=cases[key]
        if not r:
            add("本次未观察到此类案例。","В измеренных запусках такой случай не наблюдался.")
            r=next(r for r in rows if r["mode"]=="rag" and r["k"]==3 and r["gold"]!="NOINFO"
                   and r["correct"] and r["evidence_present"] and not r["citation_correct"])
            add("替代案例：标签正确，但引用遗漏了完整的金标准证据。","Дополнительный случай: метка верна, но ссылка не содержит полного эталонного обоснования.")
        found=", ".join(f"{v['doc_id']} ({v['score']:.3f})" for v in r["found"])
        passed="; ".join(f"{doc}: {','.join(map(str,nums))}" for doc,nums in r["passed"].items())
        d.add_paragraph(f"#{r['id']} | k={r['k']} | {r['gold']} | {r['claim']}")
        add(f"检索：{found}。传入句号：{passed}。原始回答：{r.get('raw',r['status']).replace(chr(10),' ').strip()}",
            f"Выдача: {found}. Переданные номера: {passed}. Ответ: {r.get('raw',r['status']).replace(chr(10),' ').strip()}")
    add("完整提示、文档和原始答复见 results.jsonl。索引哈希及自相似检查见 index/passport.json。代码由 Codex 协助编写；验证了指标算术、证据子集、引用解析及请求 token 计数。",
        "Полные промпты, документы и ответы в results.jsonl. Хеши индекса и самопроверка в index/passport.json. Код подготовлен при помощи Codex. Проверены арифметика метрик, наличие обоснования, разбор ссылок и подсчёт токенов.")
    out=ROOT/"reports";out.mkdir(exist_ok=True)
    d.save(out/("LR2_中文报告.docx" if zh else "LR2_俄文报告.docx"))
build(True);build(False)

