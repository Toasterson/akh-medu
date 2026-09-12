#!/usr/bin/env python3
"""Deterministic subprocess peer for runtime tests; never contacts a provider."""
import json
import pathlib
import subprocess
import sys
import time
import uuid


def send(message):
    print(json.dumps(message), flush=True)


def read():
    return json.loads(sys.stdin.readline())


def response(request, result):
    send({"id": request["id"], "result": result})


def codex():
    init = read()
    assert init["method"] == "initialize"
    assert init["params"]["capabilities"]["experimentalApi"]
    response(init, {})
    assert read()["method"] == "initialized"
    start = read()
    params = start["params"]
    assert params["sandbox"] in ("read-only", "workspace-write")
    assert params["approvalsReviewer"] == "user"
    assert "developerInstructions" in params
    state_file = pathlib.Path("fake-provider-state.json")
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if start["method"] == "thread/start":
        assert params["dynamicTools"][0]["name"] == "akh"
        assert params["dynamicTools"][0]["type"] == "namespace"
        assert all(t["type"] == "function" for t in params["dynamicTools"][0]["tools"])
        assert params["allowProviderModelFallback"] is False
        thread = str(uuid.uuid4())
        state[thread] = []
    else:
        assert start["method"] == "thread/resume"
        thread = params["threadId"]
        assert thread in state
        if state[thread] == ["fail_next_resume"]:
            send({"id": start["id"], "error": {"code": -1, "message": "resume refused"}})
            return
    response(start, {"thread": {"id": thread}})
    turn = read()
    assert turn["method"] == "turn/start"
    assert turn["params"]["threadId"] == thread
    if "features.shell_tool=false" in sys.argv:
        assert turn["params"]["environments"] == []
    response(turn, {"turn": {"id": "turn-1"}})
    prompt = turn["params"]["input"][0]["text"]
    state[thread].append(prompt)
    state_file.write_text(json.dumps(state))
    answer = " | ".join(state[thread])
    if prompt == "tool":
        send({"id": 99, "method": "item/tool/call", "params": {
            "threadId": thread, "turnId": "turn-1", "namespace":"akh", "tool": "echo",
            "arguments": {"n": 7, "nested": {"flags": [True]}}}})
        result = read()
        assert result["id"] == 99
        assert result["result"]["success"]
        assert result["result"]["contentItems"][0]["type"] == "inputText"
        answer = result["result"]["contentItems"][0]["text"]
    elif prompt == "approval":
        send({"id": 98, "method": "item/commandExecution/requestApproval", "params": {
            "threadId": thread, "turnId": "turn-1", "command": "touch marker"}})
        answer = read()["result"]["decision"]
    elif prompt == "hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"])
        pathlib.Path("child.pid").write_text(str(child.pid))
        time.sleep(90)
        return
    elif prompt == "crash":
        return
    elif prompt == "failed":
        send({"method": "turn/completed", "params": {"threadId": thread,
             "turn": {"id": "turn-1", "status": "failed", "error": {"message": "provider failure"}}}})
        return
    elif prompt == "malformed":
        print("this is not JSON", flush=True)
        return
    send({"method": "item/agentMessage/delta", "params": {"threadId": thread, "itemId": "m", "delta": answer}})
    send({"method": "item/completed", "params": {"threadId": thread,
         "item": {"type": "agentMessage", "id": "m", "phase": "final_answer", "text": answer}}})
    send({"method": "turn/completed", "params": {"threadId": thread,
         "turn": {"id": "wrong-turn", "status": "failed"}}})
    send({"method": "turn/completed", "params": {"threadId": thread,
         "turn": {"id": "turn-1", "status": "completed"}}})


def claude():
    start = read()
    assert start["type"] == "start"
    assert start["instructions"]
    session = start["resume"] or str(uuid.uuid4())
    send({"type": "session", "id": session})
    send({"type": "text", "text": start["prompt"]})
    send({"type": "done", "success": start["prompt"] != "failed", "text": start["prompt"], "error": "provider failure"})


if sys.argv[1] == "app-server":
    codex()
else:
    claude()
