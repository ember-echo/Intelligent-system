import json
from pathlib import Path

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import ChatAssistantMessage, ChatToolResultMessage, text_content_block_from_string as text

from agent import Agent, ROOT, read


def reply(tool, args):
    return {"tool_calls":[{"id":"script","type":"function","function":{
        "name":tool,"arguments":json.dumps(args)}}]}


class Script:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, *args):
        return next(self.replies), {"usage": {"prompt_tokens":0,"completion_tokens":0},"seconds":0}


class ReferenceReplay(BasePipelineElement):
    def __init__(self, task):
        self.task = task

    def query(self, query, runtime, env, messages=(), extra_args=None):
        log, outputs = [], []
        for i, step in enumerate(self.task.ground_truth(env),1):
            call = FunctionCall(function=step.function,args=step.args,id=f"reference-{i}")
            result, error = runtime.run_function(env, call.function, call.args)
            assert error is None, error
            output = tool_result_to_str(result)
            outputs.append(output)
            log.extend([
                ChatAssistantMessage(role="assistant",content=None,tool_calls=[call]),
                ChatToolResultMessage(role="tool",content=[text(output)],tool_call=call,
                                      tool_call_id=call.id,error=None),
            ])
        log.append(ChatAssistantMessage(role="assistant",content=[text("\n".join(outputs) or "Done.")],
                                        tool_calls=None))
        return query,runtime,env,log,extra_args or {}


def check():
    cfg = read(ROOT/"config.json")
    suite = get_suites(cfg["benchmark"])[cfg["suite"]]
    runtime = FunctionsRuntime(suite.tools)
    assert len(runtime.functions) == 24
    def run(mode, replies):
        env = suite.load_and_inject_default_environment({})
        agent = Agent(cfg, mode, Script(replies))
        _,_,env,history,_ = agent.query("Simulated check",runtime,env)
        assert history[-1].get("role") == "assistant"
        assert history[-1].get("content")
        return agent.last,env

    profile,_ = run("react",[reply("missing_workspace_tool",{}),reply("search_emails",{}),{"content":"done"}])
    errors = [e["error"] for e in profile["trace"] if e["kind"]=="tool"]
    assert profile["stop"]=="final_answer" and len(errors)==2
    assert all(errors) and "required" in errors[1].lower(), errors
    profile,_ = run("react",[reply("get_current_day",{})]*15)
    assert profile["stop"]=="max_tools" and profile["tool_calls"]==15
    plan = {"content":json.dumps({"steps":[{"tool":"get_current_day","purpose":"day"}]})}
    profile,after = run("plan",[plan,reply("create_file",{"filename":"unexpected.txt","content":"example"}),{"content":"done"}])
    assert "Expected only" in next(e["error"] for e in profile["trace"] if e["kind"]=="tool")
    assert after == suite.load_and_inject_default_environment({})
    passed = sum(suite.run_task_with_pipeline(ReferenceReplay(task),task,None,{})[0]
                 for task in suite.user_tasks.values())
    print("Reference tool outputs pass",passed,"/40")
    assert passed >= 35, passed
    print("Offline harness checks passed")


if __name__=="__main__":
    check()
