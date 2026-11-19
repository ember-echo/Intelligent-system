import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import mlflow

from generation import label, lines, parse, put, rationale, read, sha
from lr3 import CONFIG, LR2, MODES, ROOT, git, scores


def condition(rows):
    eligible = [r for r in rows if r["gold"] != "NOINFO"]
    counts = Counter(r["gold"] for r in rows)
    return dict(n=len(rows), accuracy=mean(r["correct"] for r in rows),
                class_recall={c: sum(r["gold"] == c and r["correct"] for r in rows)/n
                              for c, n in counts.items()},
                counts=dict(counts), statuses=dict(Counter(r["status"] for r in rows)),
                grounded_n=len(eligible),
                citation=mean(r["citation_correct"] for r in eligible),
                context=mean(r["evidence_present"] for r in eligible))


def compare(base, alt):
    assert base.keys() == alt.keys()
    return dict(n=len(base), fixed=sum(not base[i]["correct"] and alt[i]["correct"] for i in base),
                broken=sum(base[i]["correct"] and not alt[i]["correct"] for i in base))


def top(row, mode, depth):
    return [hit["doc_id"] for hit in row["lists"][mode][:depth]]


def hit(row, mode, depth, evidence):
    return bool(evidence.intersection(top(row, mode, depth)))


def analyze():
    all_claims = lines(LR2 / "data/claims_dev.jsonl")
    gold = {c["id"]: c for c in all_claims}
    selection = read(ROOT / "selection.json")
    selected = selection["selected"]
    assert selection["rankings_sha256"] == sha(ROOT / "runs/dev/rankings.jsonl")
    report = {"selected": selected}
    for split, expected_modes, n in (("dev", ("dense", "rrf", "ce"), 90),
                                      ("holdout", ("dense", selected), 210)):
        folder = ROOT / "runs" / split
        move = lines(folder / "rankings.jsonl")
        result = lines(folder / "results.jsonl")
        ids = read(ROOT / "data/manifest.json")[split + "_ids"]
        assert [r["id"] for r in move] == ids
        by_mode = {m: {r["id"]: r for r in result if r["mode"] == m} for m in expected_modes}
        assert len(result) == n * len(expected_modes) and all(set(r) == set(ids) for r in by_mode.values())
        for row in result:
            claim = gold[row["id"]]
            verdict = parse(row.get("raw", ""), row["passed"], "rag") if row["status"] == "ok" else None
            assert row["verdict"] == verdict
            assert row["gold"] == label(claim)
            assert row["correct"] == bool(verdict and verdict["label"] == row["gold"])
            assert row["citation_correct"] == bool(row["correct"] and verdict["doc_id"] and
                                                   rationale(claim, verdict["doc_id"], verdict["sentences"]))
            assert row["evidence_present"] == any(rationale(claim, d, ns) for d, ns in row["passed"].items())
        retrieval = scores(move, gold)
        assert retrieval == read(folder / "retrieval_metrics.json")
        summary = dict(qplus=sum(bool(gold[i]["evidence"]) for i in ids), retrieval=retrieval,
                       generation={m: condition(list(r.values())) for m, r in by_mode.items()},
                       transitions={m: compare(by_mode["dense"], by_mode[m]) for m in expected_modes[1:]})
        report[split] = summary
        if split != "dev":
            continue
        timing = {}
        for m in ("dense", "bm25", "rrf", "ce", "rrf_total", "ce_total"):
            stages = {"rrf_total": MODES[:3], "ce_total": MODES}.get(m, (m,))
            seq = [sum(r["search_s"][s] for s in stages)*1000 for r in move]
            timing[m] = dict(n=len(seq), median=median(seq), low=min(seq), high=max(seq))
        complement = {}
        keys = {(True, True): "both", (True, False): "dense_only",
                (False, True): "bm25_only", (False, False): "neither"}
        for depth in (3, 20):
            counts = {group: dict(n=0, rrf_hit3=0) for group in keys.values()}
            for row in move:
                evidence = set(gold[row["id"]]["evidence"])
                if not evidence:
                    continue
                group = keys[hit(row, "dense", depth, evidence), hit(row, "bm25", depth, evidence)]
                counts[group]["n"] += 1
                counts[group]["rrf_hit3"] += hit(row, "rrf", 3, evidence)
            complement[depth] = counts
        summary.update(timing_ms=timing, complement=complement)
        traces = []
        evidence_of = lambda row: set(gold[row["id"]]["evidence"])
        is_hit = lambda row, mode: hit(row, mode, 3, evidence_of(row))
        def changed(row):
            i = row["id"]
            a, b = (by_mode[m][i] for m in ("rrf", "ce"))
            return (a["verdict"] or {}).get("label") != (b["verdict"] or {}).get("label")
        criteria = {
            "bm25_only": lambda r: is_hit(r, "bm25") and not is_hit(r, "dense"),
            "dense_only": lambda r: is_hit(r, "dense") and not is_hit(r, "bm25"),
            "rrf_change": lambda r: is_hit(r, "rrf") != is_hit(r, "dense"),
            "ce_answer": changed,
        }
        for kind, criterion in criteria.items():
            candidates = [r for r in move if criterion(r) and r["id"] not in {t["id"] for t in traces}]
            row = next((r for r in candidates if evidence_of(r) and
                        is_hit(r, "rrf") != is_hit(r, "ce")), candidates[0])
            i = row["id"]
            traces.append(dict(kind=kind, id=i, claim=gold[i]["claim"], gold=label(gold[i]),
                               evidence=gold[i]["evidence"], rankings=row["lists"],
                               answers={m: by_mode[m][i] for m in expected_modes}))
        report["traces"] = traces
    return report


def log(report):
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("LR3_rag_retrieval")
    for split in ("dev", "holdout"):
        path = ROOT / "runs" / split
        source = read(path / "provenance.json")
        data = report[split]
        with mlflow.start_run(run_name=f"SciFact {split}") as run:
            mlflow.set_tags(dict(lab="LR3", series=split, commit=source["commit"], model="Qwen3-4B-Q4_K_M"))
            mlflow.log_params({**read(CONFIG), "split": split, "qplus": data["qplus"],
                               "selected": report["selected"]})
            mlflow.log_metrics({f"{m}.{k.replace('@', '_at_')}": v
                                for m, metrics in data["retrieval"].items() for k, v in metrics.items()})
            mlflow.log_metrics({f"{m}.{k}": metrics[k] for m, metrics in data["generation"].items()
                                for k in ("accuracy", "citation", "context")})
            mlflow.log_artifacts(str(path), artifact_path=split)
            mlflow.log_artifact(str(ROOT / "reports" / "summary.json"))
            print(split, run.info.run_id)


if __name__ == "__main__":
    report = analyze()
    (ROOT / "reports").mkdir(exist_ok=True)
    put(ROOT / "reports" / "summary.json", {k:v for k,v in report.items() if k != "traces"})
    put(ROOT / "reports" / "traces.json", report["traces"])
    import sys
    if "--log-mlflow" in sys.argv:
        log(report)


