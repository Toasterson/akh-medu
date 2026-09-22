#!/usr/bin/env python3
"""Bounded export of complete, rendered sessions through transcript MCP only."""
from __future__ import annotations
import argparse, base64, json, os, subprocess, urllib.request
from pathlib import Path

def sse_json(body: str):
    for line in body.splitlines():
        if line.startswith("data: ") and line[6:].strip():
            return json.loads(line[6:])
    raise RuntimeError("MCP response contained no data event")

class Client:
    def __init__(self, url: str, token: str): self.url, self.token, self.sid, self.next_id = url, token.strip(), None, 1
    def request(self, method, params=None, notification=False):
        payload={"jsonrpc":"2.0","method":method};
        if params is not None: payload["params"]=params
        if not notification: payload["id"], self.next_id = self.next_id, self.next_id + 1
        headers={"Authorization":"Bearer "+self.token,"Content-Type":"application/json","Accept":"application/json, text/event-stream"}
        if self.sid: headers["Mcp-Session-Id"]=self.sid
        req=urllib.request.Request(self.url,data=json.dumps(payload).encode(),headers=headers,method="POST")
        with urllib.request.urlopen(req, timeout=60) as response:
            self.sid=response.headers.get("Mcp-Session-Id",self.sid); body=response.read().decode()
        return None if notification else sse_json(body)
    def initialize(self):
        self.request("initialize",{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"lilith-transcript-export","version":"1"}})
        self.request("notifications/initialized",notification=True)
    def tool(self,name,args):
        result=self.request("tools/call",{"name":name,"arguments":args})
        if "error" in result: raise RuntimeError("MCP tool failed")
        return json.loads(result["result"]["content"][0]["text"])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--url",required=True); p.add_argument("--token-command",required=True,nargs="+"); p.add_argument("--token-base64",action="store_true")
    p.add_argument("--repo",required=True); p.add_argument("--limit",type=int,default=12); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise SystemExit("refusing to overwrite export")
    token=subprocess.check_output(a.token_command,text=True).strip()
    if a.token_base64: token=base64.b64decode(token).decode().strip()
    client=Client(a.url,token)
    try: client.initialize()
    except Exception as error: raise SystemExit(f"MCP initialization failed ({type(error).__name__}; details suppressed)") from None
    listing=client.tool("list_sessions",{"repo":a.repo,"limit":a.limit,"budget_bytes":256000})
    summaries=listing.get("sessions",[]); exported=[]
    for summary in summaries[:a.limit]:
        cursor=None; events=[]
        while True:
            args={"source":summary["source"],"session_id":summary["session_id"],"order":"spliced","include_thoughts":False,"include_all":False,"budget_bytes":256000}
            if cursor: args["cursor"]=cursor
            page=client.tool("get_session",args); events.extend(page["events"]); cursor=page.get("continue_from")
            if not cursor: break
        exported.append({"source":summary["source"],"session_id":summary["session_id"],"events":events,"truncated":False,"continue_from":None})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(exported,ensure_ascii=False)); os.chmod(a.output,0o600)
    print(json.dumps({"sessions":len(exported),"events":sum(len(s["events"]) for s in exported)}))
if __name__=="__main__": main()
