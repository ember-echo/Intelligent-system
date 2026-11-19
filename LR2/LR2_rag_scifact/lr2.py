import argparse, collections, hashlib, json, os, pathlib, subprocess, tarfile, time
import numpy as np
import requests

ROOT=pathlib.Path(__file__).resolve().parent
DATA=ROOT/"data"
LABELS=("SUPPORT","CONTRADICT","NOINFO")
URL="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DIGEST="11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
MODEL="Qwen3-4B-Instruct-2507-Q4_K_M"
BASE="http://127.0.0.1:8022"
BUDGET=2048

def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def put(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def lines(p):return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
def label(c):
    labs={e["label"] for es in c["evidence"].values() for e in es}
    return "NOINFO" if not c["evidence"] else next(iter(labs)) if len(labs)==1 else None

def prepare(archive):
    assert sha(archive)==DIGEST
    DATA.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for name in ("corpus.jsonl","claims_train.jsonl","claims_dev.jsonl"):
            member=next(m for m in tar.getmembers() if m.isfile() and pathlib.PurePosixPath(m.name).name==name)
            (DATA/name).write_bytes(tar.extractfile(member).read())
    claims=lines(DATA/"claims_dev.jsonl")
    dev=[];holdout=[]
    for lab in LABELS:
        ids=sorted((c["id"] for c in claims if label(c)==lab),key=lambda i:hashlib.sha256(f"scifact:{i}".encode()).hexdigest())
        dev+=ids[:30];holdout+=ids[30:]
    assert len(dev)==90 and len(holdout)==210 and len(lines(DATA/"corpus.jsonl"))==5183
    put(DATA/"manifest.json",dict(source=URL,archive_sha256=DIGEST,files={p.name:sha(p) for p in DATA.glob("*.jsonl")},
        dev_ids=dev,holdout_ids=holdout,dev_counts=dict.fromkeys(LABELS,30),
        holdout_counts=dict(collections.Counter(label(c) for c in claims if c["id"] in holdout)),
        excluded_ids=[c["id"] for c in claims if label(c) is None]))

def encode(modeldir):
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
    model=AutoModel.from_pretrained(modeldir,local_files_only=True).eval()
    corpus=sorted(lines(DATA/"corpus.jsonl"),key=lambda r:r["doc_id"])
    manifest=read(DATA/"manifest.json")
    claims={r["id"]:r["claim"] for r in lines(DATA/"claims_dev.jsonl")}
    texts=[r["title"]+"\n"+" ".join(r["abstract"]) for r in corpus]
    all_text=texts+[claims[i] for i in manifest["dev_ids"]]
    out=[];truncated=0
    for i in range(0,len(all_text),48):
        batch=all_text[i:i+48]
        lengths=tok(batch,truncation=False,return_length=True,verbose=False)["length"]
        truncated+=sum(n>256 for j,n in enumerate(lengths) if i+j<len(texts))
        x=tok(batch,padding=True,truncation=True,max_length=256,return_tensors="pt")
        with torch.inference_mode():
            h=model(**x).last_hidden_state
            mask=x["attention_mask"].unsqueeze(-1)
            vec=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=1)
        out.append(vec.numpy())
        if i%480==0:print("Embedding",i,len(all_text),flush=True)
    mat=np.concatenate(out).astype("float32")
    idx=ROOT/"index";idx.mkdir(exist_ok=True)
    np.save(idx/"matrix.npy",mat[:len(corpus)])
    np.save(idx/"queries.npy",mat[len(corpus):])
    put(idx/"doc_ids.json",[str(r["doc_id"]) for r in corpus])
    check=tok([all_text[0],all_text[0]],padding=True,truncation=True,max_length=256,return_tensors="pt")
    with torch.inference_mode():
        h=model(**check).last_hidden_state
        m=check["attention_mask"].unsqueeze(-1)
        v=torch.nn.functional.normalize((h*m).sum(1)/m.sum(1),dim=1)
    self_cosine=float(v[0]@v[1])
    assert abs(self_cosine-1)<1e-5
    put(idx/"passport.json",dict(model="sentence-transformers/all-MiniLM-L6-v2",
        revision=(modeldir/"revision.txt").read_text().strip(),limit=256,dimension=384,prefixes=None,
        pooling="attention-masked mean pooling, L2 normalization",truncated=truncated,total=len(corpus),
        matrix_sha256=sha(idx/"matrix.npy"),query_matrix_sha256=sha(idx/"queries.npy"),
        self_cosine=self_cosine,weights_sha256=sha(modeldir/"model.safetensors")))

http=requests.Session()
def post(path,body):
    r=http.post(BASE+path,json=body,timeout=(8,90))
    r.raise_for_status()
    return r.json()

def metric(found,rel):
    hits=[i for i,d in enumerate(found,1) if d in rel]
    return int(bool(hits)),len(hits)/len(rel),1/hits[0] if hits else 0

def rationale(c,doc,nums):
    return any(set(e["sentences"])<=set(nums) for e in c["evidence"].get(doc,[]))

def parse(raw,passed,mode):
    try:
        v=json.loads(raw)
        if not isinstance(v,dict) or set(v)!={"label","doc_id","sentences"} or v["label"] not in LABELS:raise ValueError()
        doc,n=v["doc_id"],v["sentences"]
        if not isinstance(n,list) or any(type(i) is not int or i<0 for i in n):raise ValueError()
        if v["label"]=="NOINFO" or mode=="only":
            if doc is not None or n:raise ValueError()
        elif not isinstance(doc,str) or doc not in passed or not n or not set(n)<=set(passed[doc]):raise ValueError()
        return v
    except (ValueError,TypeError):return None

def sanity():
    assert metric(["a","b","c"],{"c","d"})==(1,.5,1/3)
    assert metric(["a"],{"c","d"})==(0,0,0)
    c={"evidence":{"7":[{"sentences":[1,2]}]}}
    assert rationale(c,"7",[1,2,3]) and not rationale(c,"7",[1])
    good={"label":"SUPPORT","doc_id":"7","sentences":[1]}
    assert parse(json.dumps(good),{"7":[1]},"rag")==good
    for v in (dict(good,label="UNKNOWN"),dict(good,extra=1),dict(good,sentences=[2])):
        assert parse(json.dumps(v),{"7":[1]},"rag") is None

def build(claim,docs,mode):
    system=(ROOT/"prompts"/("only.txt" if mode=="only" else "evidence.txt")).read_text(encoding="utf-8").strip()
    def messages(blocks):return [{"role":"system","content":system},{"role":"user","content":"Claim: "+claim+("\n\nDocuments:\n"+"\n\n".join(blocks) if mode!="only" else "")}]
    def size(msg):
        rendered=post("/apply-template",{"messages":msg})["prompt"]
        return len(post("/tokenize",{"content":rendered,"add_special":False,"parse_special":True})["tokens"])
    blocks=[];passed={}
    for d,nums in docs:
        def render(ns):return f'Document {d["doc_id"]}: {d["title"]}\n'+"\n".join(f'[{j}] {d["abstract"][j]}' for j in ns)
        if size(messages(blocks+[render(nums)]))<=BUDGET:
            blocks.append(render(nums));passed[str(d["doc_id"])]=nums;continue
        lo,hi=0,len(nums)
        while hi-lo>1:
            mid=(lo+hi)//2
            if size(messages(blocks+[render(nums[:mid])]))<=BUDGET:lo=mid
            else:hi=mid
        if lo:
            blocks.append(render(nums[:lo]));passed[str(d["doc_id"])]=nums[:lo]
        break
    msg=messages(blocks);n=size(msg);assert n<=BUDGET
    return msg,passed,n

def evaluate(exe,weights,smoke):
    sanity()
    d=ROOT/"runs"/("smoke" if smoke else "dev")
    d.mkdir(parents=True,exist_ok=True)
    corpus={str(r["doc_id"]):r for r in lines(DATA/"corpus.jsonl")}
    manifest=read(DATA/"manifest.json")
    # Gold annotations are confined to scoring and the diagnostic oracle.
    gold={c["id"]:c for c in lines(DATA/"claims_dev.jsonl") if c["id"] in set(manifest["dev_ids"])}
    idx=ROOT/"index";docs=read(idx/"doc_ids.json")
    vec=np.load(idx/"matrix.npy");qs=np.load(idx/"queries.npy")
    rankings={}
    for id,q in zip(manifest["dev_ids"],qs):
        start=time.perf_counter();s=vec@q
        top=np.argsort(-s,kind="stable")[:5]
        rankings[id]=([dict(doc_id=docs[i],score=float(s[i])) for i in top],time.perf_counter()-start)
    put(d/"rankings.json",rankings)
    schema=read(ROOT/"response_format.json")
    records=d/"results.jsonl"
    done={(r["mode"],r["k"],r["id"]) for r in lines(records)} if records.exists() else set()
    cmd=[str(exe.resolve()),"-m",str(weights.resolve()),"--alias",MODEL,"--host","127.0.0.1","--port","8022",
         "-ngl","99","-c","4096","-np","1","-t","8","-tb","8","--jinja"]
    put(d/"server.json",dict(command=cmd,model_sha256=sha(weights),server_sha256=sha(exe),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
    with (d/"server.log").open("a",encoding="utf-8") as log,records.open("a",encoding="utf-8") as output:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            for _ in range(120):
                if proc.poll() is not None:raise RuntimeError("Server exited; inspect server.log")
                try:
                    if http.get(BASE+"/health",timeout=2).ok:break
                except requests.RequestException:pass
                time.sleep(.5)
            else:raise RuntimeError("Server startup timeout")
            def one(c,mode,k):
                id=c["id"];found,search=rankings.get(id,([],0))
                evidence=[]
                if mode=="rag":evidence=[(corpus[r["doc_id"]],list(range(len(corpus[r["doc_id"]]["abstract"])))) for r in found[:k]]
                if mode=="oracle":
                    doc=sorted(c["evidence"])[0]
                    evidence=[(corpus[doc],sorted(c["evidence"][doc][0]["sentences"]))]
                msg,passed,n=build(c["claim"],evidence,mode)
                start=time.perf_counter()
                row=dict(id=id,claim=c["claim"],mode=mode,k=k,found=found[:k] if mode=="rag" else [],passed=passed,
                         messages=msg,prompt_tokens=n,search_s=search if mode=="rag" else 0,gold=label(c),
                         evidence_present=any(rationale(c,doc,nums) for doc,nums in passed.items()))
                try:
                    resp=post("/v1/chat/completions",dict(model=MODEL,messages=msg,temperature=0,seed=42,max_tokens=128,
                        response_format=schema,cache_prompt=False))
                    raw=resp["choices"][0]["message"]["content"];usage=resp["usage"]
                    v=parse(raw,passed,mode)
                    status="ok" if v is not None else "invalid"
                    if usage["prompt_tokens"]!=n or n>BUDGET:status="token_mismatch";v=None
                    row.update(raw=raw,verdict=v,status=status,usage=usage)
                except Exception as exc:row.update(status="error",error=repr(exc),verdict=None)
                v=row["verdict"]
                correct=v is not None and v["label"]==row["gold"]
                row.update(generation_s=time.perf_counter()-start,correct=correct,
                    citation_correct=bool(correct and v["doc_id"] and rationale(c,v["doc_id"],v["sentences"])))
                if row["gold"] in LABELS[:2] and mode=="rag":
                    row["retrieval"]=metric([r["doc_id"] for r in found[:k]],set(c["evidence"]))
                return row
            warm=next(c for c in lines(DATA/"claims_train.jsonl") if label(c) in LABELS[:2])
            w=one(warm,"oracle",0);put(d/"warmup.json",w)
            if w["status"]!="ok":raise RuntimeError(str(w))
            if smoke:return
            count=len(done)
            for mode,k in [("only",0),("rag",3),("oracle",0),("rag",1),("rag",5)]:
                for id in manifest["dev_ids"]:
                    c=gold[id]
                    if (mode,k,id) in done or (mode=="oracle" and label(c)=="NOINFO"):continue
                    r=one(c,mode,k);output.write(json.dumps(r,ensure_ascii=False)+"\n");output.flush()
                    count+=1
                    if count%10==0 or r["status"] not in {"ok","invalid"}:print(count,mode,k,r["status"],flush=True)
        finally:
            proc.terminate();proc.wait(timeout=20)

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest="task",required=True)
    sp.add_parser("prepare").add_argument("archive",type=pathlib.Path)
    sp.add_parser("index").add_argument("encoder",type=pathlib.Path)
    e=sp.add_parser("run");e.add_argument("server",type=pathlib.Path);e.add_argument("weights",type=pathlib.Path);e.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.task=="prepare":prepare(a.archive)
    elif a.task=="index":encode(a.encoder)
    else:evaluate(a.server,a.weights,a.smoke)
if __name__=="__main__":main()
