import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import time
from datetime import datetime
import requests

ROOT = pathlib.Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8011"
MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M"

def post(path, payload):
    r = requests.post(BASE + path, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()

def tokenize(text):
    return post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

def prompt(uid, target, tokens):
    k = target - 30
    for _ in range(12):
        excerpt = post("/detokenize", {"tokens": tokens[:k]})["content"]
        msgs = [{"role": "user", "content": uid + "\nSummarize the following text in one sentence.\n" + excerpt}]
        count = len(tokenize(post("/apply-template", {"messages": msgs})["prompt"]))
        if count == target:
            return msgs
        k += target - count
    raise ValueError(f"Cannot build {target}-token prompt")

def timing(t0, t1, last, n):
    return t1-t0, (last-t1)/(n-1) if n>1 else None, last-t0

def measure(series, uid, msgs, n, k, clients=1, cache=False, warm=False):
    start = time.perf_counter()
    row = dict(series=series, id=uid, clients=clients, warmup=warm, cache_prompt=cache,
               target_prompt=n, target_output=k, t0=start)
    chunks, offsets, usage = [], [], {}
    finish = None
    try:
        with requests.post(BASE + "/v1/chat/completions", json=dict(
            model=MODEL, messages=msgs, temperature=0, seed=42, max_tokens=k,
            ignore_eos=True, cache_prompt=cache, stream=True,
            stream_options={"include_usage": True}), stream=True, timeout=(10,180)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data:"): continue
                item = line[5:].strip()
                if item == b"[DONE]": break
                data = json.loads(item)
                if data.get("usage"): usage = data["usage"]
                if data.get("choices"):
                    choice = data["choices"][0]
                    content = choice.get("delta", {}).get("content")
                    if content:
                        offsets.append(time.perf_counter()-start)
                        chunks.append(content)
                    if choice.get("finish_reason"): finish = choice["finish_reason"]
        p, out = usage["prompt_tokens"], usage["completion_tokens"]
        ttft, tpot, e2e = timing(0, offsets[0], offsets[-1], out)
        row.update(status="ok" if (p,out)==(n,k) else "length_mismatch", prompt_tokens=p,
                   completion_tokens=out, ttft_s=ttft, tpot_s=tpot, e2e_s=e2e,
                   chunk_offsets_s=offsets, finish_reason=finish, output="".join(chunks), usage=usage)
    except Exception as exc:
        row.update(status="error",error=repr(exc),chunk_offsets_s=offsets)
    return row

class Server:
    def __init__(self, exe, model, slots, out, ngl):
        self.log = (out/f"server-{slots}.log").open("w", encoding="utf-8")
        self.cmd = [str(exe), "-m", str(model), "--alias", MODEL, "--port", "8011",
                    "--host", "127.0.0.1", "-c", "16384", "-np", str(slots), "-ngl", str(ngl),
                    "-t", "8", "-tb", "8", "--metrics", "--jinja"]
    def __enter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        deadline = time.monotonic()+120
        while time.monotonic()<deadline and self.proc.poll() is None:
            try:
                if requests.get(BASE+"/health", timeout=2).ok: return self
            except requests.RequestException: pass
            time.sleep(.5)
        self.__exit__()
        raise RuntimeError("Server startup failed")
    def __exit__(self,*args):
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=15)
        self.log.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--server",type=pathlib.Path,required=True)
    parser.add_argument("--model",type=pathlib.Path,required=True)
    parser.add_argument("--ngl",type=int,default=99)
    parser.add_argument("--smoke",action="store_true")
    a=parser.parse_args()
    assert all(abs(x-y)<1e-9 for x,y in zip(timing(0,.4,2.4,101),(.4,.02,2.4)))
    def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    if not a.smoke and git("status","--porcelain"): raise RuntimeError("Commit code and forecast before measuring")
    out=ROOT/"runs"/(("smoke-" if a.smoke else "lr1-")+datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    def save(name,obj): (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    def sha(path):
        with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
    a.server=a.server.resolve();a.model=a.model.resolve()
    book=ROOT/"data/pg1342.txt"
    corpus=" ".join(book.read_text(encoding="utf-8-sig").split("*** START OF THE PROJECT GUTENBERG EBOOK",1)[-1].split())
    passport=dict(model=MODEL, model_bytes=a.model.stat().st_size, model_sha256=sha(a.model),
                  server_sha256=sha(a.server), server_release="b12284", corpus_sha256=sha(book),
                  model_revision="ae44f08e1392f39c0e474af10c3ff8355c8b6688",
                  git_commit=git("rev-parse","HEAD"),context=16384,ngl=a.ngl,
                  gpu=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.free",
                      "--format=csv"],capture_output=True,text=True).stdout if a.ngl else "CPU",
                  ttft_limit_s=2,tpot_limit_s=.1)
    rows=[]; prompts={}
    def write(r):
        rows.append(r)
        with (out/"results.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        print(r["series"],r["id"],r["status"],r.get("prompt_tokens"),r.get("completion_tokens"),round(r.get("e2e_s",0),3),flush=True)
    def build(uid,n,offset=0):
        m=prompt(uid,n,tokens[offset:]);prompts[uid]=m
        return m
    def warm(s,c=1):
        uid=f"warm-{s}-{c}";write(measure(s,uid,build(uid,128),128,16,c,warm=True))
    try:
        with Server(a.server,a.model,1,out,a.ngl) as srv:
            passport["command_single"]=srv.cmd;save("passport.json",passport)
            tokens=tokenize(corpus[:105000])
            m=build("sanity",128)
            sanity=[measure("checks",f"check-{i}",m,128,64) for i in range(1 if a.smoke else 2)]
            for r in sanity: write(r)
            if any(r["status"]!="ok" for r in sanity): raise RuntimeError("Sanity request failed")
            if a.smoke:return
            save("checks.json",dict(example_ttft_ms=400,example_tpot_ms=20,
                temperature_zero_same_text=sanity[0]["output"]==sanity[1]["output"], exact_64=True))
            warm("prompt_length")
            for n in [128,512,1024,2048,4096]:
                for i in range(5):
                    uid=f"P{n}-{i}";write(measure("prompt_length",uid,build(uid,n),n,16))
            warm("prefix_cache")
            m=build("cache-identical",2048,5000)
            for i in range(3):write(measure("prefix_cache",f"cache-{i}",m,2048,16,cache=True))
            warm("output_length")
            for k in [16,64,256,1024]:
                for i in range(5):
                    uid=f"O{k}-{i}";write(measure("output_length",uid,build(uid,1024),1024,k))
        with Server(a.server,a.model,4,out,a.ngl) as srv:
            passport["command_load"]=srv.cmd;save("passport.json",passport)
            for c in [1,2,4]:
                warm("load",c)
                tasks=[(f"L{c}-{i}",build(f"L{c}-{i}",1024,i*83)) for i in range(16)]
                def snapshot(label):
                    (out/f"metrics-{label}-{c}.txt").write_text(requests.get(BASE+"/metrics",timeout=10).text)
                snapshot("before")
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    jobs=[pool.submit(measure,"load",uid,m,1024,128,c) for uid,m in tasks]
                    for job in concurrent.futures.as_completed(jobs):write(job.result())
                snapshot("after")
    finally: save("prompts.json",prompts)
    if any(r["status"]!="ok" for r in rows if not r["warmup"]): raise RuntimeError("Invalid requests recorded")
    print("RESULT_DIRECTORY",out)

if __name__=="__main__": main()
