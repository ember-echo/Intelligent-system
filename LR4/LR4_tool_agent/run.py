import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suites
from agent import Agent, ROOT, read

WRITES = {
    "send_email", "delete_email", "send_email_reply", "mark_email_as_read", "mark_email_as_unread",
    "create_calendar_event", "cancel_calendar_event", "delete_calendar_event",
    "reschedule_calendar_event", "modify_calendar_event", "update_calendar_event", "update_calendar_event_participants",
    "append_to_file", "delete_file", "create_file", "move_file", "share_file", "rename_file",
    "update_file", "add_calendar_event_participants", "remove_calendar_event_participants",
}
MODES = ("react", "plan")
BASE = ROOT.parent.parent
MODEL = BASE / "LR1/runtime/model.gguf"
SERVER = BASE / "LR1/runtime/bin/llama-server.exe"
CONFIG = ROOT / "config.json"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tools_for(task, environment):
    calls = task.ground_truth(task.init_environment(environment.model_copy(deep=True)))
    return [call.function for call in calls]


def manifest(suite, cfg):
    order = [f"user_task_{i}" for i in range(40)]
    assert set(order) == set(suite.user_tasks)
    environment = suite.load_and_inject_default_environment({})
    data = {"source":"https://github.com/ethz-spylab/agentdojo", "package":importlib.metadata.version("agentdojo"),
            "benchmark":cfg["benchmark"], "suite":cfg["suite"],
            "dev":[id for i,id in enumerate(order) if not i%4],
            "holdout":[id for i,id in enumerate(order) if i%4],
            "task_id_sha256":hashlib.sha256("\n".join(order).encode()).hexdigest(),
            "groups":{}}
    for id in order:
        reference = tools_for(suite.user_tasks[id], environment)
        data["groups"][id] = {"reference_calls":len(reference), "writes":bool(WRITES.intersection(reference))}
    print("Groups",sum(v["reference_calls"]==1 for v in data["groups"].values()),
          sum(v["writes"] for v in data["groups"].values()),flush=True)
    assert sum(v["reference_calls"]==1 for v in data["groups"].values()) == 14
    assert sum(v["writes"] for v in data["groups"].values()) == 22
    dump(ROOT / "data/manifest.json", data)


def run(suite, cfg, split, only=None):
    data = read(ROOT / "data/manifest.json")
    if split == "holdout" and git("status", "--porcelain"):
        raise RuntimeError("Commit LR4 code, config, and forecast first.")
    out = ROOT / "runs" / split
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"commit":git("rev-parse","HEAD"), "package": data["package"],
                "python":platform.python_version(), "config":cfg,
                "sha256":{p.name:digest(p) for p in [CONFIG, MODEL, SERVER, ROOT/"data/manifest.json",
                                                       *sorted((ROOT/"prompts").glob("*.txt"))]},
                "server_flags":"--host 127.0.0.1 --port 8024 -c 16384 -np 1 -ngl 99 -t 8 -tb 8 --jinja --metrics"}
    meta = out / "provenance.json"
    if meta.exists() and read(meta) != metadata:
        raise RuntimeError("Cannot resume with changed inputs or commit.")
    dump(meta, metadata)
    results = out / "results.jsonl"
    previous = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()] if results.exists() else []
    seen = {(r["task"],r["mode"]) for r in previous}
    assert len(seen) == len(previous)
    planned = [only] if only else data[split]
    with results.open("a", encoding="utf-8") as stream:
        for i, id in enumerate(planned):
            for mode in MODES if i%2==0 else MODES[::-1]:
                if (id,mode) in seen:
                    continue
                task = suite.get_user_task_by_id(id)
                agent = Agent(cfg,mode)
                verdict = suite.run_task_with_pipeline(agent, task, None, {})
                utility = verdict[0] if isinstance(verdict, tuple) else verdict
                assert isinstance(utility,bool), (type(verdict).__name__,repr(verdict)[:160])
                row = {"task":id,"success":utility, **data["groups"][id], **agent.last}
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                stream.flush()
                print(split,id,mode,utility,row["stop"],row["model_calls"],row["tool_calls"],
                      round(row["seconds"],2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["manifest","dev","holdout"])
    parser.add_argument("--only")
    args=parser.parse_args()
    cfg=read(CONFIG)
    suite=get_suites(cfg["benchmark"])[cfg["suite"]]
    if args.action=="manifest":
        manifest(suite,cfg)
    else:
        if args.only and (args.action!="dev" or args.only not in read(ROOT/"data/manifest.json")["dev"]):
            parser.error("--only is for a dev task")
        run(suite,cfg,args.action,args.only)

