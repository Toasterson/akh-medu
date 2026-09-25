#!/usr/bin/env python3
"""Deterministic subprocess peer for runtime tests; never contacts a provider."""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
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
        if params["dynamicTools"]:
            assert params["dynamicTools"][0]["name"] == "akh"
            assert params["dynamicTools"][0]["type"] == "namespace"
            assert params["dynamicTools"][0]["tools"], "upstream rejects empty namespaces"
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
    send({"method": "item/reasoning/summaryTextDelta", "params": {"threadId": thread, "delta": "fixture reasoning"}})
    send({"method": "thread/tokenUsage/updated", "params": {"threadId": thread, "tokenUsage": {"last": {"inputTokens": 17, "outputTokens": 3, "reasoningOutputTokens": 2}}}})
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
    send({"type": "reasoning", "text": "fixture Claude reasoning"})
    send({"type": "usage", "usage": {"input_tokens": 11, "output_tokens": 5}})
    send({"type": "text", "text": start["prompt"]})
    send({"type": "done", "success": start["prompt"] != "failed", "text": start["prompt"], "error": "provider failure"})


def opencode():
    """ACP peer shaped like `opencode acp` 1.18, plus an MCP client for Akh tools."""
    assert sys.argv[sys.argv.index("--hostname") + 1] == "127.0.0.1"
    assert sys.argv[sys.argv.index("--port") + 1] == "0"
    assert "--mdns=false" in sys.argv
    config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
    assert os.environ["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert len(os.environ["OPENCODE_SERVER_PASSWORD"]) == 64
    assert config["default_agent"] == "akh"
    assert config["experimental"]["mcp_timeout"] == 5000
    assert config["experimental"]["continue_loop_on_deny"] is True
    agent = config["agent"]["akh"]
    assert agent["mode"] == "primary" and agent["prompt"]
    rules = agent["permission"]
    assert rules["akh_*"] == "allow"
    brokered = os.environ.get("OPENCODE_PURE") == "1"
    if brokered:
        assert all(action == "deny" for key, action in rules.items() if key != "akh_*")
        assert json.loads(os.environ["OPENCODE_PERMISSION"]) == rules
        assert config["lsp"] is False and config["formatter"] is False and config["snapshot"] is False
        assert "Native workspace tools are disabled" in agent["prompt"]
    init = read()
    assert init["jsonrpc"] == "2.0" and init["method"] == "initialize"
    assert init["params"]["protocolVersion"] == 1
    capabilities = init["params"]["clientCapabilities"]
    assert capabilities["terminal"] is False and capabilities["fs"]["writeTextFile"] is False
    send({"jsonrpc": "2.0", "id": init["id"], "result": {"protocolVersion": 1,
          "agentCapabilities": {"loadSession": True, "mcpCapabilities": {"http": True}}}})
    opened = read()
    params = opened["params"]
    assert params["cwd"] == os.getcwd()
    state_file = pathlib.Path("fake-opencode-state.json")
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    servers = params["mcpServers"]
    if servers and not pathlib.Path("skip-mcp").exists():
        # Like OpenCode's MCP.add: connect and cache the catalog before the
        # session is returned to the client.
        assert len(servers) == 1 and servers[0]["type"] == "http" and servers[0]["name"] == "akh"
        url = servers[0]["url"]
        assert url.startswith("http://127.0.0.1:")
        headers = {h["name"]: h["value"] for h in servers[0]["headers"]}

        def mcp(body, auth=True):
            request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                         **(headers if auth else {})})
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    raw = response.read()
                    return response.status, (json.loads(raw) if raw else None)
            except urllib.error.HTTPError as error:
                return error.code, None

        assert mcp({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, auth=False)[0] == 401
        status, init_result = mcp({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                   "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                              "clientInfo": {"name": "opencode", "version": "1.18"}}})
        assert status == 200 and init_result["result"]["serverInfo"]["name"] == "akh"
        assert mcp({"jsonrpc": "2.0", "method": "notifications/initialized"})[0] == 202
        _, listed = mcp({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert [t["name"] for t in listed["result"]["tools"]] == ["echo"]
    if opened["method"] == "session/new":
        session = "ses_" + uuid.uuid4().hex
        state[session] = []
        send({"jsonrpc": "2.0", "id": opened["id"], "result": {"sessionId": session}})
    else:
        assert opened["method"] == "session/load"
        session = params["sessionId"]
        assert session in state
        # History replay precedes the load response and is not this turn's output.
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session,
              "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "REPLAYED"}}}})
        send({"jsonrpc": "2.0", "id": opened["id"], "result": {}})
    mode = read()
    assert mode["method"] == "session/set_mode" and mode["params"] == {"sessionId": session, "modeId": "akh"}
    send({"jsonrpc": "2.0", "id": mode["id"], "result": {}})
    step = read()
    if step["method"] == "session/set_config_option":
        assert step["params"]["configId"] == "model" and step["params"]["value"] == "fixture/model"
        send({"jsonrpc": "2.0", "id": step["id"], "result": {"configOptions": []}})
        step = read()
    prompt = step
    assert prompt["method"] == "session/prompt" and prompt["params"]["sessionId"] == session
    text = prompt["params"]["prompt"][0]["text"]
    state[session].append(text)
    state_file.write_text(json.dumps(state))
    answer = " | ".join(state[session])

    def update(update_kind, **fields):
        send({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": session, "update": {"sessionUpdate": update_kind, **fields}}})

    if text == "tool":
        assert servers, "the Akh MCP endpoint is offered whenever tools are supplied"
        update("tool_call", toolCallId="call-1", title="akh_echo", kind="other", status="pending",
               rawInput={"n": 7})
        _, called = mcp({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                         "params": {"name": "echo", "arguments": {"n": 7, "nested": {"flags": [True]}}}})
        update("tool_call_update", toolCallId="call-1", status="completed")
        answer = called["result"]["content"][0]["text"]
        _, rogue = mcp({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "shell_exec", "arguments": {}}})
        assert rogue["result"]["isError"] is True
        assert "disabled" in rogue["result"]["content"][0]["text"]
    elif text == "approval":
        send({"jsonrpc": "2.0", "id": "fs-1", "method": "fs/write_text_file",
              "params": {"sessionId": session, "path": "/tmp/x", "content": "x"}})
        refused = read()
        assert refused["id"] == "fs-1" and refused["error"]["code"] == -32601
        send({"jsonrpc": "2.0", "id": "perm-1", "method": "session/request_permission", "params": {
            "sessionId": session,
            "toolCall": {"toolCallId": "t1", "title": "touch marker", "kind": "execute",
                         "rawInput": {"command": "touch marker"}},
            "options": [{"optionId": "once", "kind": "allow_once", "name": "Allow once"},
                        {"optionId": "always", "kind": "allow_always", "name": "Always allow"},
                        {"optionId": "reject", "kind": "reject_once", "name": "Reject"}]}})
        reply = read()
        assert reply["id"] == "perm-1"
        answer = reply["result"]["outcome"].get("optionId", "cancelled")
    elif text == "hang":
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            message = json.loads(line)
            if message.get("method") == "session/cancel":
                assert message["params"]["sessionId"] == session
                pathlib.Path("cancelled.marker").write_text(session)
                send({"jsonrpc": "2.0", "id": prompt["id"], "result": {"stopReason": "cancelled"}})
                return
    elif text == "auth":
        send({"jsonrpc": "2.0", "id": prompt["id"], "error": {"code": -32000, "message": "Authentication required"}})
        return
    elif text == "stop":
        send({"jsonrpc": "2.0", "id": prompt["id"], "result": {"stopReason": "max_tokens"}})
        return
    update("agent_thought_chunk", content={"type": "text", "text": "fixture OpenCode reasoning"})
    update("agent_message_chunk", content={"type": "text", "text": answer})
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "ses_other",
          "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "LEAK"}}}})
    send({"jsonrpc": "2.0", "id": prompt["id"], "result": {"stopReason": "end_turn",
          "usage": {"inputTokens": 13, "outputTokens": 4, "thoughtTokens": 1, "totalTokens": 18}}})


if sys.argv[1] == "app-server":
    codex()
elif sys.argv[1] == "acp":
    opencode()
else:
    claude()
