import argparse, csv, json, pathlib, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mlflow

ROOT=pathlib.Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=pathlib.Path);a=p.parse_args()
    d=a.run.resolve()
    raw=[json.loads(l) for l in (d/"results.jsonl").read_text(encoding="utf-8").splitlines()]
    selected=[r for r in raw if not r["warmup"] and r["series"]!="checks"]
    assert len(selected)==96 and all(r["status"]=="ok" for r in selected)
    def group(series,**kwargs):
        return [r for r in selected if r["series"]==series and all(r[k]==v for k,v in kwargs.items())]
    def median(rs,k):return st.median(r[k] for r in rs)
    plen=[]
    for n in [128,512,1024,2048,4096]:
        rs=group("prompt_length",prompt_tokens=n);assert len(rs)==5
        vals=[r["ttft_s"] for r in rs]
        plen.append(dict(tokens=n,ttft_s=st.median(vals),low=min(vals),high=max(vals),prefill_proxy_tok_s=n/st.median(vals)))
    olen=[]
    for k in [16,64,256,1024]:
        rs=group("output_length",completion_tokens=k);assert len(rs)==5
        olen.append(dict(tokens=k,e2e_s=median(rs,"e2e_s"),ttft_s=median(rs,"ttft_s"),tpot_s=median(rs,"tpot_s")))
    slope,intercept=np.polyfit([r["tokens"] for r in olen],[r["e2e_s"] for r in olen],1)
    load=[]
    for c in [1,2,4]:
        rs=group("load",clients=c);assert len(rs)==16
        wall=max(r["t0"]+r["e2e_s"] for r in rs)-min(r["t0"] for r in rs)
        passing=sum(r["ttft_s"]<=2 and r["tpot_s"]<=.1 for r in rs)
        load.append(dict(clients=c,n=len(rs),wall_s=wall,tokens_s=sum(r["completion_tokens"] for r in rs)/wall,
            requests_s=len(rs)/wall,ttft_median_s=median(rs,"ttft_s"),ttft_p90_s=float(np.percentile([r["ttft_s"] for r in rs],90)),
            tpot_median_s=median(rs,"tpot_s"),goodput_requests_s=passing/wall,passing=passing))
    cache=[r["ttft_s"] for r in group("prefix_cache")];assert len(cache)==3
    passport=json.loads((d/"passport.json").read_text(encoding="utf-8"))
    log=(d/"server-4.log").read_text(encoding="utf-8",errors="replace")
    kv=[line.strip() for line in log.splitlines() if "KV buffer size" in line or "n_ctx_slot" in line or "n_ctx_per_seq" in line]
    summary=dict(prompt=plen,output=olen,load=load,cache_ttft_s=cache,cache_speedup=cache[0]/cache[1],
        slope_s_per_token=float(slope),intercept_s=float(intercept),n=len(selected),
        max_passing_clients=max((r["clients"] for r in load if r["ttft_p90_s"]<=2 and r["tpot_median_s"]<=.1),default=0),
        kv_log=kv,model_bytes=passport["model_bytes"],bandwidth_assumption_gb_s=256,
        weight_only_lower_bound_ms=1000*passport["model_bytes"]/256e9)
    (d/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    for key in ["prompt","output","load"]:
        rs=summary[key]
        with (d/f"{key}.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rs[0]);w.writeheader();w.writerows(rs)
    plt.rcParams.update({"figure.dpi":165,"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    def chart(name,x,y,xlabel,ylabel,fit=None,yerr=None,labels=None):
        fig,ax=plt.subplots(figsize=(6.7,2.55))
        ax.errorbar(x,y,yerr=yerr,fmt="o-",color="#146D85",label="Measured median",capsize=4)
        if fit is not None:ax.plot(x,fit,"--",color="#A15F11",label="Linear fit");ax.legend(frameon=False)
        if labels:
            for xx,yy,l in zip(x,y,labels):ax.annotate(l,(xx,yy),xytext=(4,5),textcoords="offset points")
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(d/f"{name}.png");plt.close(fig)
    x=[r["tokens"] for r in plen];y=[r["ttft_s"] for r in plen]
    chart("ttft_prompt",x,y,"Input tokens","TTFT (s)",yerr=[[r["ttft_s"]-r["low"] for r in plen],[r["high"]-r["ttft_s"] for r in plen]])
    x=[r["tokens"] for r in olen];y=[r["e2e_s"] for r in olen]
    chart("latency_output",x,y,"Output tokens","End-to-end latency (s)",fit=[intercept+slope*k for k in x])
    chart("throughput_latency",[r["tokens_s"] for r in load],[r["ttft_p90_s"] for r in load],
          "Aggregate output tokens/s","p90 TTFT (s)",labels=[f'{r["clients"]} clients' for r in load])
    mlflow.set_tracking_uri("sqlite:///"+(ROOT/"mlflow.db").as_posix())
    client=mlflow.tracking.MlflowClient()
    if not client.get_experiment_by_name("LR1_llm_inference_serving"):
        client.create_experiment("LR1_llm_inference_serving",artifact_location=(ROOT/"mlartifacts").as_uri())
    mlflow.set_experiment("LR1_llm_inference_serving")
    with mlflow.start_run(run_name=d.name):
        mlflow.log_params({k:passport[k] for k in ("model","server_release","git_commit","ngl","context")})
        mlflow.log_metrics(dict(measured_requests=len(selected),cache_speedup=summary["cache_speedup"],
                                max_passing_clients=summary["max_passing_clients"]))
        for row in load:
            mlflow.log_metrics({k:v for k,v in row.items() if k!="clients"},step=row["clients"])
        mlflow.log_artifacts(str(d),"benchmark")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
