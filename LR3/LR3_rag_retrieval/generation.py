# Context assembly and response validation reused verbatim from LR2.
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
BASE = "http://127.0.0.1:8023"
MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M"
BUDGET = 2048
http = requests.Session()

def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def put(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def read(p):return json.loads(p.read_text(encoding="utf-8"))

def lines(p):return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]

def label(c):
    labs={e["label"] for es in c["evidence"].values() for e in es}
    return "NOINFO" if not c["evidence"] else next(iter(labs)) if len(labs)==1 else None

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
