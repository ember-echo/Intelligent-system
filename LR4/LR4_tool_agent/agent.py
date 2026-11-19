import copy
import json
from pathlib import Path
from time import perf_counter

import requests
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, FunctionCall
from agentdojo.types import (
    ChatAssistantMessage, ChatSystemMessage, ChatToolResultMessage, ChatUserMessage,
    text_content_block_from_string as text,
)

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompt(name):
    return (ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()


def descriptor(function):
    return {"type": "function", "function": {
        "name": function.name, "description": function.description,
        "parameters": function.parameters.model_json_schema(),
    }}


class Halt(Exception):
    pass


class LocalModel:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        if config["base_url"] != "http://127.0.0.1:8024":
            raise ValueError("This runner connects only to its local model endpoint.")

    def post(self, endpoint, body):
        response = self.session.post(self.config["base_url"] + endpoint, json=body, timeout=(5, 180))
        response.raise_for_status()
        return response.json()

    def complete(self, messages, tools, schema, choice):
        cfg = self.config
        body = {"model": cfg["model"], "messages": messages, "temperature": cfg["temperature"],
                "seed": cfg["seed"], "max_tokens": cfg["max_tokens"], "cache_prompt": True}
        if tools:
            body.update(tools=tools, tool_choice=choice)
        if schema:
            body["response_format"] = {"type":"json_schema","json_schema":{"name":"plan","schema":schema}}
        template = self.post("/apply-template", body)["prompt"]
        input_count = len(self.post("/tokenize", {"content":template, "add_special":False,
                                                   "parse_special":True})["tokens"])
        if input_count + cfg["max_tokens"] > cfg["context"]:
            raise Halt("context_limit")
        started = perf_counter()
        result = self.post("/v1/chat/completions", body)
        choice_data = result["choices"][0]
        return choice_data["message"], {
            "seconds": perf_counter() - started, "estimated_input_tokens": input_count,
            "usage": result["usage"], "finish": choice_data.get("finish_reason"),
        }


class Episode:
    def __init__(self, cfg, model, query, runtime, env):
        self.cfg, self.model, self.runtime, self.env = cfg, model, runtime, env
        self.tools = {f.name: descriptor(f) for f in runtime.functions.values()}
        self.messages = [{"role":"system","content":prompt("system")}, {"role":"user","content":query}]
        self.benchmark = [ChatSystemMessage(role="system", content=[text(prompt("system"))]),
                          ChatUserMessage(role="user", content=[text(query)])]
        self.trace = []
        self.calls = self.attempts = 0

    def ask(self, phase, *, messages=None, tools=None, schema=None, choice="auto"):
        if self.calls >= self.cfg["max_calls"]:
            raise Halt("max_calls")
        messages = self.messages if messages is None else messages
        request = dict(kind="model", phase=phase, messages=copy.deepcopy(messages),
                       tools=[t["function"]["name"] for t in tools or []], schema=schema)
        self.trace.append(request)
        try:
            reply, stats = self.model.complete(messages, tools, schema, choice)
        except Halt as exc:
            request["error"] = str(exc)
            raise
        except requests.RequestException as exc:
            self.calls += 1
            request["error"] = str(exc)
            raise Halt("model_error") from exc
        self.calls += 1
        request.update(reply=reply, **stats)
        return reply

    def dispatch(self, tool_call, expected=None, deny=None):
        if self.attempts >= self.cfg["max_tools"]:
            raise Halt("max_tools")
        self.attempts += 1
        function = tool_call.get("function") or {}
        name = function.get("name", "")
        supplied = function.get("arguments")
        identifier = tool_call.get("id") or f"call-{self.attempts}"
        args, error = {}, None
        started = perf_counter()
        try:
            args = json.loads(supplied) if isinstance(supplied, str) else supplied
            if not isinstance(args, dict):
                raise ValueError("argument JSON must be an object")
        except (ValueError, TypeError) as exc:
            args, error = {}, str(exc)
        if deny is not None:
            error = deny
        if error is None and expected and name != expected:
            error = f"Expected only the planned tool {expected}."
        if error is None and name not in self.tools:
            error = f"Unknown workspace tool: {name}"
        if error is None:
            try:
                result, error = self.runtime.run_function(self.env, name, args)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        output = str(error) if error is not None else tool_result_to_str(result)
        call = FunctionCall(function=name, args=args, id=identifier)
        self.benchmark.extend([
            ChatAssistantMessage(role="assistant", content=None, tool_calls=[call]),
            ChatToolResultMessage(role="tool", content=[text(output)], tool_call_id=identifier,
                                  tool_call=call, error=str(error) if error is not None else None),
        ])
        self.trace.append(dict(kind="tool", name=name, arguments=args, raw_arguments=supplied, result=output,
                               error=None if error is None else str(error), seconds=perf_counter()-started,
                               id=identifier))
        return {"role":"tool","tool_call_id":identifier,"content":output}

    def react(self):
        while True:
            reply = self.ask("react", tools=list(self.tools.values()))
            requested = reply.get("tool_calls") or []
            if not requested:
                return reply.get("content") or "The model returned no final answer."
            self.messages.append(dict(role="assistant", content=reply.get("content"), tool_calls=requested))
            for call in requested:
                self.messages.append(self.dispatch(call))
                if self.attempts >= self.cfg["max_tools"]:
                    raise Halt("max_tools")

    def plan(self):
        render_tools = json.dumps(list(self.tools.values()), ensure_ascii=False)
        messages = self.messages + [{"role":"user","content":prompt("plan") + "\nAvailable tools:\n" + render_tools}]
        schema = {"type":"object","properties":{"steps":{"type":"array","minItems":1,
                 "maxItems":self.cfg["max_plan_steps"],"items":{"type":"object","properties":{
                    "tool":{"type":"string","enum":list(self.tools)},"purpose":{"type":"string"}},
                    "required":["tool","purpose"],"additionalProperties":False}}},
                 "required":["steps"],"additionalProperties":False}
        reply = self.ask("plan", messages=messages, schema=schema)
        try:
            plan = json.loads(reply.get("content") or "")
            if not isinstance(plan, dict) or set(plan) != {"steps"}:
                raise ValueError("Expected steps")
            steps = plan["steps"]
            if not isinstance(steps, list) or not 1 <= len(steps) <= self.cfg["max_plan_steps"]:
                raise ValueError("Invalid count")
            if any(not isinstance(s, dict) or set(s) != {"tool","purpose"} or
                   s["tool"] not in self.tools or not isinstance(s["purpose"], str) for s in steps):
                raise ValueError("Invalid step")
        except (ValueError, TypeError):
            raise Halt("plan_invalid")
        self.trace.append(dict(kind="plan", steps=steps))
        self.messages.append({"role":"assistant","content":json.dumps(plan, ensure_ascii=False)})
        for i, step in enumerate(steps, 1):
            name = step["tool"]
            instruction = {"role":"user","content":f"{prompt('step')}\nStep {i}: {name}, {step['purpose']}"}
            reply = self.ask("step", messages=self.messages+[instruction],
                             tools=[self.tools[name]],
                             choice={"type":"function","function":{"name":name}})
            calls = reply.get("tool_calls") or []
            if len(calls) == 1:
                result = self.dispatch(calls[0], expected=name)
                note = result["content"]
            else:
                note = "Step did not produce exactly one tool call."
                self.trace.append(dict(kind="step_error", tool=name, result=note))
                for call in calls:
                    self.dispatch(call, deny=note)
                    if self.attempts >= self.cfg["max_tools"]:
                        raise Halt("max_tools")
            self.messages.append({"role":"user","content":f"Workspace observation for step {i} ({name}):\n{note}"})
        reply = self.ask("final", messages=self.messages+[{"role":"user","content":prompt("final")}])
        return reply.get("content") or "The model returned no final answer."


class Agent(BasePipelineElement):
    def __init__(self, config, mode, model=None):
        self.name = "local-" + config["model"]
        self.cfg, self.mode = config, mode
        self.model = LocalModel(config) if model is None else model
        self.last = None

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        episode = Episode(self.cfg, self.model, query, runtime, env)
        started = perf_counter()
        reason = "final_answer"
        try:
            answer = episode.react() if self.mode == "react" else episode.plan()
        except Halt as exc:
            reason = str(exc)
            answer = f"The task stopped before completion ({reason})."
        episode.benchmark.append(ChatAssistantMessage(role="assistant", content=[text(answer)], tool_calls=None))
        self.last = {"query":query,"mode":self.mode,"stop":reason,"answer":answer,
                     "seconds":perf_counter()-started,"model_calls":episode.calls,
                     "tool_calls":episode.attempts,"trace":episode.trace}
        return query, runtime, env, episode.benchmark, extra_args or {}

