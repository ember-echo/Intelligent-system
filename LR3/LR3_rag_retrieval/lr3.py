import argparse
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import generation as gen
from retrieval import BM25, checks, fuse, ranked

ROOT = Path(__file__).resolve().parent
LR2 = ROOT.parent.parent / "LR2/LR2_rag_scifact"
ENCODER = LR2.parent / "encoder"
CE = ROOT.parent / "cross_encoder"
SERVER = ROOT.parent.parent / "LR1/runtime/bin/llama-server.exe"
WEIGHTS = SERVER.parent.parent / "model.gguf"
CONFIG = ROOT / "config.json"
MODES = ("dense", "bm25", "rrf", "ce")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def provenance():
    if git("status", "--porcelain"):
        raise RuntimeError("Commit inputs and code before running.")
    source = [CONFIG, ROOT / "data/manifest.json", ROOT / "index/dense_lr2.json",
              ROOT / "prompts/evidence.txt", ROOT / "response_format.json",
              WEIGHTS, SERVER, LR2 / "runs/dev/server.json", LR2 / "runs/dev/results.jsonl",
              LR2 / "runs/dev/rankings.json", LR2 / "index/matrix.npy", LR2 / "index/doc_ids.json"]
    source += sorted(LR2.joinpath("data").glob("*.jsonl"))
    source += sorted(x for directory in (ENCODER, CE) for x in directory.iterdir() if x.is_file())
    hashes = {os.path.relpath(p, ROOT): gen.sha(p) for p in source}
    cfg = gen.read(CONFIG)
    assert CE.joinpath("revision.txt").read_text().strip() == cfg["ce_revision"]
    assert all(gen.sha(LR2 / "data" / path) == h
               for path, h in gen.read(ROOT / "data/manifest.json")["files"].items())
    passport = gen.read(ROOT / "index/dense_lr2.json")
    assert gen.sha(LR2 / "index/matrix.npy") == passport["matrix_sha256"]
    assert gen.sha(ENCODER / "model.safetensors") == passport["weights_sha256"]
    assert all(gen.sha(ROOT / p) == gen.sha(LR2 / p) for p in ("prompts/evidence.txt", "response_format.json"))
    server_src = gen.read(LR2 / "runs/dev/server.json")
    assert gen.sha(WEIGHTS) == server_src["model_sha256"] and gen.sha(SERVER) == server_src["server_sha256"]
    return {"commit": git("rev-parse", "HEAD"), "sha256": hashes, "ce_device": "cpu",
            "python": os.sys.version, "torch": torch.__version__}


class Search:
    def __init__(self, corpus, cfg):
        self.cfg = cfg
        self.ids = gen.read(LR2 / "index/doc_ids.json")
        self.matrix = np.load(LR2 / "index/matrix.npy")
        assert len(self.ids) == len(self.matrix) == len(corpus) == 5183
        assert len(set(self.ids)) == len(corpus) and set(self.ids) == corpus.keys()
        self.docs = {key: row["title"] + "\n" + " ".join(row["abstract"])
                     for key, row in corpus.items()}
        self.lexical = BM25(self.docs, cfg["k1"], cfg["b"])
        self.dense_tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        self.dense_model = AutoModel.from_pretrained(ENCODER, local_files_only=True).eval()
        self.ce_tokenizer = AutoTokenizer.from_pretrained(CE, local_files_only=True)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(CE, local_files_only=True).eval()

    @torch.inference_mode()
    def run(self, claim):
        cfg = self.cfg
        depth = cfg["depth"]
        time0 = time.perf_counter()
        batch = self.dense_tokenizer([claim], truncation=True, max_length=256, return_tensors="pt")
        h = self.dense_model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        emb = torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=1)[0].numpy()
        dense = ranked(dict(zip(self.ids, self.matrix @ emb)), depth)
        time1 = time.perf_counter()
        bm25 = self.lexical.search(claim, depth)
        time2 = time.perf_counter()
        rrf = fuse(dense, bm25, cfg["rrf_constant"], depth)
        time3 = time.perf_counter()
        ids = [r["doc_id"] for r in rrf]
        pairs = self.ce_tokenizer([claim] * len(ids), [self.docs[d] for d in ids],
                                  truncation="only_second", max_length=cfg["pair_limit"],
                                  padding=True, return_tensors="pt")
        logits = self.ce_model(**pairs).logits[:, 0].tolist()
        ce = ranked(dict(zip(ids, logits)), depth)
        time4 = time.perf_counter()
        assert {r["doc_id"] for r in ce} == set(ids)
        return dict(zip(MODES, (dense, bm25, rrf, ce))), dict(
            zip(MODES, (time1-time0, time2-time1, time3-time2, time4-time3)))


def setup(split):
    manifest = gen.read(ROOT / "data/manifest.json")
    assert len(manifest["dev_ids"]) == len(set(manifest["dev_ids"])) == 90
    assert len(manifest["holdout_ids"]) == len(set(manifest["holdout_ids"])) == 210
    assert not set(manifest["dev_ids"]) & set(manifest["holdout_ids"])
    ids = manifest[split + "_ids"]
    corpus = {str(row["doc_id"]): row for row in gen.lines(LR2 / "data/corpus.jsonl")}
    gold = {c["id"]: c for c in gen.lines(LR2 / "data/claims_dev.jsonl") if c["id"] in ids}
    assert len(gold) == len(ids) and sum(bool(c["evidence"]) for c in gold.values()) == {"dev": 60, "holdout": 128}[split]
    warm = gen.lines(LR2 / "data/claims_train.jsonl")[0]
    return ids, corpus, gold, warm


def scores(rows, gold):
    eligible = [row for row in rows if gold[row["id"]]["evidence"]]
    result = {}
    for mode in MODES:
        result[mode] = {}
        for depth in (3, 20):
            values = [gen.metric([v["doc_id"] for v in r["lists"][mode][:depth]],
                                 set(gold[r["id"]]["evidence"])) for r in eligible]
            result[mode].update({f"{metric}@{depth}": mean(column)
                                 for metric, column in zip(("hit", "recall", "mrr"), zip(*values))})
    assert result["rrf"]["recall@20"] == result["ce"]["recall@20"]
    return result


@contextmanager
def server(out):
    with (out / "server.log").open("a", encoding="utf-8") as log:
        cmd = [str(SERVER), "-m", str(WEIGHTS), "--alias", gen.MODEL, "--host", "127.0.0.1",
               "--port", "8023", "-ngl", "99", "-c", "4096", "-np", "1", "-t", "8", "-tb", "8", "--jinja"]
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("llama.cpp exited. See server.log.")
                try:
                    if requests.get(gen.BASE + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("llama.cpp startup timeout")
            yield
        finally:
            process.terminate()
            process.wait(timeout=20)


def generate(c, found, corpus, mode, cfg):
    docs = [(corpus[r["doc_id"]], list(range(len(corpus[r["doc_id"]]["abstract"]))))
            for r in found]
    row = dict(id=c["id"], claim=c["claim"], mode=mode, found=found, gold=gen.label(c),
               passed={}, evidence_present=False, verdict=None, correct=False, citation_correct=False)
    try:
        messages, passed, n = gen.build(c["claim"], docs, "rag")
        row.update(messages=messages, passed=passed, prompt_tokens=n,
                   evidence_present=any(gen.rationale(c, d, nums) for d, nums in passed.items()))
        start = time.perf_counter()
        try:
            response = gen.post("/v1/chat/completions", dict(
                model=gen.MODEL, messages=messages, temperature=cfg["temperature"], seed=cfg["seed"],
                max_tokens=cfg["max_tokens"], response_format=gen.read(ROOT / "response_format.json"),
                cache_prompt=False))
        finally:
            row["generation_s"] = time.perf_counter() - start
        raw = response["choices"][0]["message"]["content"]
        row.update(raw=raw, usage=response["usage"])
        if response["usage"]["prompt_tokens"] != n:
            row["status"] = "token_mismatch"
            return row
        verdict = gen.parse(raw, passed, "rag")
        correct = verdict is not None and verdict["label"] == row["gold"]
        row.update(verdict=verdict, status="ok" if verdict is not None else "invalid", correct=correct,
                   citation_correct=bool(correct and verdict["doc_id"] and
                                         gen.rationale(c, verdict["doc_id"], verdict["sentences"])))
    except Exception as exc:
        row.update(status="error", error=repr(exc))
    return row


def evaluate(split):
    checks()
    cfg = gen.read(CONFIG)
    torch.set_num_threads(cfg["threads"])
    run = provenance()
    selection = None
    if split == "holdout":
        selection = gen.read(ROOT / "selection.json")
        assert selection["config_sha256"] == gen.sha(CONFIG)
        assert git("diff", selection["dev_commit"], "HEAD", "--", ".", ":!selection.json") == ""
    out = ROOT / "runs" / split
    out.mkdir(parents=True, exist_ok=True)
    identity = out / "provenance.json"
    if identity.exists():
        assert gen.read(identity) == run, "Run inputs changed"
    else:
        gen.put(identity, run)
    ids, corpus, gold, warm = setup(split)
    retrieval_file = out / "rankings.jsonl"
    rows = gen.lines(retrieval_file) if retrieval_file.exists() else []
    assert [r["id"] for r in rows] == ids[:len(rows)]
    prior = gen.read(LR2 / "runs/dev/rankings.json") if split == "dev" else None
    if len(rows) < len(ids):
        engine = Search(corpus, cfg)
        engine.run(warm["claim"])
        with retrieval_file.open("a", encoding="utf-8") as stream:
            for id in ids[len(rows):]:
                lists, timing = engine.run(gold[id]["claim"])
                if prior is not None:
                    assert [r["doc_id"] for r in lists["dense"][:5]] == [r["doc_id"] for r in prior[str(id)][0]], id
                row = dict(id=id, lists=lists, search_s=timing)
                append(stream, row)
                rows.append(row)
                if len(rows) % 10 == 0:
                    print(split, "retrieval", len(rows), flush=True)
        del engine
    metrics = scores(rows, gold)
    gen.put(out / "retrieval_metrics.json", metrics)
    top = {r["id"]: r["lists"] for r in rows}
    modes = ("rrf", "ce") if split == "dev" else ("dense", selection["selected"])
    result_file = out / "results.jsonl"
    existing = gen.lines(result_file) if result_file.exists() else []
    completed = {(r["mode"], r["id"]) for r in existing}
    assert len(completed) == len(existing)
    if split == "dev":
        previous = [r for r in gen.lines(LR2 / "runs/dev/results.jsonl") if r["mode"] == "rag" and r["k"] == 3]
        assert {r["id"] for r in previous} == set(ids) and len(previous) == len(ids)
        for row in previous:
            assert [r["doc_id"] for r in row["found"]] == [r["doc_id"] for r in top[row["id"]]["dense"][:3]]
        if any(("dense", row["id"]) not in completed for row in previous):
            with result_file.open("a", encoding="utf-8") as stream:
                for row in previous:
                    if ("dense", row["id"]) not in completed:
                        append(stream, dict(row, mode="dense", reused_from="LR2 rag@3"))
    pending = [(mode, id) for mode in modes for id in ids if (mode, id) not in completed]
    if pending:
        with server(out), result_file.open("a", encoding="utf-8") as stream:
            warm_result = generate(warm, [], corpus, "warmup", cfg)
            gen.put(out / "warmup.json", warm_result)
            if warm_result["status"] not in ("ok", "invalid"):
                raise RuntimeError(f"Warm-up failed: {warm_result['status']} {warm_result.get('error', '')}")
            for i, (mode, id) in enumerate(pending, 1):
                row = generate(gold[id], top[id][mode][:cfg["context_k"]], corpus, mode, cfg)
                append(stream, row)
                if i % 10 == 0 or row["status"] not in ("ok", "invalid"):
                    print(split, "generation", i, "/", len(pending), mode, row["status"], flush=True)
    saved = gen.lines(result_file)
    expected = {(mode, id) for mode in (("dense", *modes) if split == "dev" else modes) for id in ids}
    assert len(saved) == len(expected) and {(r["mode"], r["id"]) for r in saved} == expected
    if split == "dev":
        order = cfg["selection_order"]
        chosen = max(order, key=lambda m: (metrics[m]["mrr@3"], metrics[m]["recall@3"], -order.index(m)))
        gen.put(out / "selection_candidate.json", dict(selected=chosen, metrics=metrics,
                rule="max MRR@3, max Recall@3, then BM25/RRF/CE", dev_commit=run["commit"],
                config_sha256=gen.sha(CONFIG), rankings_sha256=gen.sha(retrieval_file)))
        print("Selected", chosen, metrics[chosen], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["check", "dev", "holdout"])
    split = parser.parse_args().split
    if split == "check":
        checks()
    else:
        evaluate(split)

