import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
MODES = ("react", "plan")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")


def category(r):
    if r["success"]:
        return "success"
    if r["stop"] != "final_answer":
        return r["stop"]
    if any(e.get("error") for e in r["trace"] if e["kind"]=="tool"):
        return "tool_error"
    if any(e["kind"]=="step_error" for e in r["trace"]):
        return "plan_step_error"
    return "wrong_answer_or_state"


def measure(row):
    stats = {"input_tokens":0,"output_tokens":0}
    for step in row["trace"]:
        usage = step.get("usage",{})
        stats["input_tokens"] += usage.get("prompt_tokens",0)
        stats["output_tokens"] += usage.get("completion_tokens",0)
    return {**stats,**{k:row[k] for k in ("model_calls","tool_calls","seconds")}}


def bounds(records):
    if not records:
        return {}
    values = [measure(r) for r in records]
    return {name: dict(median=median(x),min=min(x),max=max(x))
            for name,x in ((k,[r[k] for r in values]) for k in values[0])}


def analyze(split):
    path = ROOT / "runs" / split
    plan = read(ROOT / "data/manifest.json")[split]
    records = rows(path/"results.jsonl")
    assert len(records) == 2*len(plan)
    by = {m:{r["task"]:r for r in records if r["mode"]==m} for m in MODES}
    assert all(set(group)==set(plan) for group in by.values())
    buckets = {
        "all":lambda r: True,
        "one":lambda r:r["reference_calls"]==1, "several":lambda r:r["reference_calls"]>1,
        "read":lambda r:not r["writes"], "write":lambda r:r["writes"],
    }
    result = {"n":len(plan),"modes":{}}
    for m in MODES:
        all_rows = list(by[m].values())
        groups = {}
        for name, use in buckets.items():
            chosen = [r for r in all_rows if use(r)]
            groups[name] = dict(n=len(chosen),success=sum(r["success"] for r in chosen))
        result["modes"][m] = dict(groups=groups, fail=dict(Counter(category(r) for r in all_rows if not r["success"])),
                                 cost={k:bounds(v) for k,v in (("all",all_rows),
                                  ("success",[r for r in all_rows if r["success"]]),
                                  ("failure",[r for r in all_rows if not r["success"]]))})
    labels = {(True,True):"both",(True,False):"react_only",(False,True):"plan_only",(False,False):"neither"}
    result["paired"] = dict.fromkeys(labels.values(),0)
    for task in plan:
        pair = labels[by["react"][task]["success"], by["plan"][task]["success"]]
        result["paired"][pair] += 1
    return result,by


def pick_traces(by):
    chosen={}
    def keys(predicate):
        return [k for k,r in by["react"].items() if predicate(r,by["plan"][k])]
    groups=[
        ("both",keys(lambda r,p:r["success"] and p["success"])),
        ("react_only",keys(lambda r,p:r["success"] and not p["success"])),
        ("react_detour",keys(lambda r,p:not r["success"] and
                (r["stop"] in ("max_calls","max_tools") or r["tool_calls"]>=4))),
        ("context",keys(lambda r,p:r["stop"]=="context_limit" or p["stop"]=="context_limit")),
    ]
    for key, options in groups:
        if key=="react_only":
            options.sort(key=lambda k:(by["react"][k]["reference_calls"],by["react"][k]["tool_calls"]),reverse=True)
        if key=="react_detour":
            options.sort(key=lambda k:by["react"][k]["tool_calls"],reverse=True)
        target=next((k for k in options if k not in {x["task"] for x in chosen.values()}),None)
        if target:
            chosen[key]={"task":target,**{r:by[r][target] for r in MODES}}
    return chosen


def condensed(row):
    return [{k:v for k,v in step.items() if k in (
                "phase","kind","name","arguments","error","usage","steps","tool")}
            for step in row["trace"]]


def log_mlflow(data):
    import mlflow
    mlflow.set_tracking_uri("sqlite:///" + (ROOT/"mlflow.db").as_posix())
    mlflow.set_experiment("LR4_tool_agent")
    for split in data:
        if split not in ("dev","holdout"):
            continue
        meta=read(ROOT/"runs"/split/"provenance.json")
        for mode in MODES:
            with mlflow.start_run(run_name=f"Workspace {split} {mode}"):
                mlflow.set_tags({"work":"LR4","series":split,"mode":mode,"commit":meta["commit"],
                                 "model":meta["config"]["model"]})
                mlflow.log_params({k:v for k,v in meta["config"].items() if k!="base_url"})
                stats=data[split]["modes"][mode]
                mlflow.log_metrics({"success_rate":stats["groups"]["all"]["success"]/data[split]["n"],
                                    **{f"median_{k}":v["median"] for k,v in stats["cost"]["all"].items()}})
                mlflow.log_artifacts(str(ROOT/"runs"/split),artifact_path=split)
                mlflow.log_artifact(str(ROOT/"reports"/"summary.json"))
                if split=="holdout":
                    mlflow.log_artifact(str(ROOT/"reports"/"traces.json"))


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--log-mlflow",action="store_true")
    args.add_argument("--dev-only",action="store_true")
    cfg=args.parse_args()
    out=ROOT/"reports"
    out.mkdir(exist_ok=True)
    data={}
    for split in ("dev",) if cfg.dev_only else ("dev","holdout"):
        data[split],by=analyze(split)
    traces=pick_traces(by)
    dump(out/"summary.json",data)
    with (out/"success.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(("series","mode","group","success","n","accuracy"))
        for split, summary in data.items():
            for mode, values in summary["modes"].items():
                for group,v in values["groups"].items():
                    writer.writerow((split,mode,group,v["success"],v["n"],v["success"]/v["n"] if v["n"] else ""))
    with (out/"cost.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(("series","mode","outcome","metric","median","min","max"))
        for split, summary in data.items():
            for mode, values in summary["modes"].items():
                for outcome, metrics in values["cost"].items():
                    for metric, limits in metrics.items():
                        writer.writerow((split,mode,outcome,metric,limits["median"],limits["min"],limits["max"]))
    dump(out/"traces.json",traces)
    dump(out/"trace_index.json",{key:{m:{"success":t[m]["success"],"task":t["task"],
            "query":t[m]["query"],"stop":t[m]["stop"],"answer":t[m]["answer"],"trace":condensed(t[m])}
            for m in MODES} for key,t in traces.items()})
    for split in data:
        print(split, data[split]["paired"], flush=True)
    print("Traces", {k:v["task"] for k,v in traces.items()})
    if cfg.log_mlflow:
        log_mlflow(data)

