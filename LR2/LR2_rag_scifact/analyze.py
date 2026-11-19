import argparse, collections, json, pathlib, statistics as st
import mlflow
ROOT=pathlib.Path(__file__).resolve().parent
def main():
    d=ROOT/"runs/dev"
    load=lambda p:json.loads(p.read_text(encoding="utf-8"))
    rows=[json.loads(x) for x in (d/"results.jsonl").read_text(encoding="utf-8").splitlines()]
    ids=set(load(ROOT/"data/manifest.json")["dev_ids"])
    labels=["SUPPORT","CONTRADICT","NOINFO"]
    groups={n:{r["id"]:r for r in rows if (r["mode"],r["k"])==mk}
            for n,mk in {"only":("only",0),"rag1":("rag",1),"rag3":("rag",3),"rag5":("rag",5),"oracle":("oracle",0)}.items()}
    positives={i for i,r in groups["only"].items() if r["gold"]!="NOINFO"}
    assert len(rows)==420 and len(positives)==60
    assert all(set(g)==(positives if n=="oracle" else ids) for n,g in groups.items())
    metrics={}
    for n,g in groups.items():
        rs=list(g.values()); pos=[r for r in rs if r["gold"]!="NOINFO"]
        by_label={l:[r for r in rs if r["gold"]==l] for l in labels}
        m=dict(n=len(rs),correct=sum(r["correct"] for r in rs),accuracy=st.mean(r["correct"] for r in rs),
               recall={l:st.mean(r["correct"] for r in v) for l,v in by_label.items() if v},
               positive_correct=sum(g[i]["correct"] for i in positives),status=dict(collections.Counter(r["status"] for r in rs)),
               mean_tokens=st.mean(r["prompt_tokens"] for r in rs),mean_docs=st.mean(len(r["passed"]) for r in rs))
        if n!="only":
            m.update(citation_correct=sum(r["citation_correct"] for r in pos),evidence_present=sum(r["evidence_present"] for r in pos))
        if n.startswith("rag"):
            m.update(dict(zip(("hit","recall_at_k","mrr"),map(st.mean,zip(*(r["retrieval"] for r in pos))))))
        metrics[n]=m
    def pair(a,b,key,subset):
        g,h=groups[a],groups[b]
        return dict(fixed=sum(not g[i][key] and h[i][key] for i in subset),
                    broken=sum(g[i][key] and not h[i][key] for i in subset),
                    n=len(subset))
    transitions={f"{a}_to_{b}":pair(a,b,"correct",ids) for a,b in (("only","rag3"),("rag3","rag1"),("rag3","rag5"))}
    cites={f"rag3_to_{b}":pair("rag3",b,"citation_correct",positives) for b in ["rag1","rag5"]}
    r3=list(groups["rag3"].values())
    predicates={
       "retrieval_miss":lambda r:r["gold"]!="NOINFO" and r["retrieval"][0]==0 and not r["correct"],
       "context_loss":lambda r:r["gold"]!="NOINFO" and r["retrieval"][0]==1 and not r["evidence_present"] and not r["correct"],
       "generation_error":lambda r:r["gold"]!="NOINFO" and r["evidence_present"] and not r["correct"],
       "noinfo_error":lambda r:r["gold"]=="NOINFO" and not r["correct"]}
    cases={k:next((r for r in r3 if f(r)),None) for k,f in predicates.items()}
    summary=dict(total=len(rows),metrics=metrics,transitions=transitions,citation_transitions=cites,
                 case_ids={k:r["id"] if r else None for k,r in cases.items()})
    (d/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (d/"error_cases.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding="utf-8")
    mlflow.set_tracking_uri("sqlite:///"+(ROOT/"mlflow.db").as_posix())
    exp="LR2_rag_scifact"
    client=mlflow.tracking.MlflowClient()
    if not client.get_experiment_by_name(exp):client.create_experiment(exp,artifact_location=(ROOT/"mlartifacts").as_uri())
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name="SciFact dev 90"):
        mlflow.log_params(dict(encoder="all-MiniLM-L6-v2",model="Qwen3-4B-Instruct-2507-Q4_K_M",input_budget=2048,
                              dev=90,holdout=210,total_calls=420))
        mlflow.log_metrics({n+"_accuracy":m["accuracy"] for n,m in metrics.items()})
        mlflow.log_artifacts(str(d),"evaluation")
        mlflow.log_artifact(str(ROOT/"index/passport.json"),"index")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

