#!/usr/bin/env python3
"""Read-only ACP projection of autonomous sessions, plus supported T3 API driver.
No model calls, bot dispatches, filesystem tools or T3 database edits.
"""
import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

TERMINAL = {'completed', 'failed', 'cancelled'}
ROLES = ('janus', 'minerva', 'vesta')
APP = '/Applications/T3 Code (Alpha).app/Contents'

def request(url, token, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')

def project_event(event):
    """Use durable broker receipts, never infer successful tools from prose."""
    data = event['data']; kind = event['kind']
    if kind == 'runtime':
        if data['kind'] in ('text', 'reasoning'):
            return {'sessionUpdate': 'agent_thought_chunk' if data['kind']=='reasoning' else 'agent_message_chunk', 'content': {'type':'text','text':data['text']}}
    if kind == 'runtime' and data['kind'] == 'tool_started':
        return {'sessionUpdate':'tool_call','toolCallId':data['id'],'title':data.get('name','Tool'),'kind':'other','status':'in_progress'}
    if kind == 'runtime' and data['kind'] == 'tool_completed':
        return {'sessionUpdate':'tool_call_update','toolCallId':data['id'],'status':'completed' if data.get('success') else 'failed','rawOutput':data}
    return None

def acp(root, role):
    token=(root/'secrets'/f't3-{role}.key').read_text().strip()
    base='http://127.0.0.1:18431'
    write_lock=threading.Lock(); cancelled=threading.Event(); worker=None
    def send(value):
        with write_lock:
            print(json.dumps(value), flush=True)
    def reply(id, result): send({'jsonrpc':'2.0','id':id,'result':result})
    def update(session, value): send({'jsonrpc':'2.0','method':'session/update','params':{'sessionId':session,'update':value}})
    def text(session, value): update(session, {'sessionUpdate':'agent_message_chunk','content':{'type':'text','text':value}})
    def watch(message):
        session=message['params']['sessionId']; id=message['id']
        try:
            prompt=''.join(p.get('text','') for p in message['params']['prompt'])
            match=re.search(r'Observe autonomous '+role+r' session ([0-9a-f-]{36})(?![0-9a-f-])',prompt)
            if not match:
                text(session, 'This is the read-only autonomous activity feed. Use the '+role.title()+' provider to give instructions.\n'); reply(id,{'stopReason':'end_turn'}); return
            source=match[1]; path='/v1/sessions/'+source; cursor=0; heartbeat=time.monotonic()
            record=request(base+path,token)
            if record['akh']!=role: raise ValueError('session owner mismatch')
            text(session,f'\n**{role.title()} autonomous wake** · `{source}` · model `{record["runtime"].get("model")}`\n\n')
            while not cancelled.is_set():
                events=request(base+path+'/events?after='+str(cursor),token)
                # API may return a capped page. Never skip unread events.
                for event in events:
                    if event['sequence']<=cursor: continue
                    projected=project_event(event)
                    if projected: update(session,projected)
                    cursor=event['sequence']
                record=request(base+path,token)
                if all(t['status'] in TERMINAL for t in record['turns']) and cursor>=record['next_sequence']-1:
                    receipts=request(base+path+'/tools',token)['records']
                    good=sum(bool(r.get('result',{}).get('success')) for r in receipts if r.get('result'))
                    states=', '.join(t['status'] for t in record['turns'])
                    text(session,f'\n\n**Outcome: {states}.** Broker receipts: {good}/{len(receipts)} successful.\n')
                    for turn in record['turns']:
                        if turn.get('error'): text(session,'Error: '+json.dumps(turn['error'])+'\n')
                    reply(id,{'stopReason':'end_turn'}); return
                if time.monotonic()-heartbeat >= 60:
                    states=', '.join(t['status'] for t in record['turns'])
                    text(session, f'\n[Activity feed: {states} at {now()}]\n')
                    heartbeat=time.monotonic()
                cancelled.wait(2)
            reply(id,{'stopReason':'cancelled'})
        except Exception as error:
            text(session, '\nActivity feed unavailable ('+type(error).__name__+'); bot state is unknown.\n')
            send({'jsonrpc':'2.0','id':id,'error':{'code':-32603,'message':'Activity source unavailable'}})
    for line in sys.stdin:
        m=json.loads(line); method=m.get('method'); id=m.get('id')
        if method=='initialize': reply(id,{'protocolVersion':1,'agentInfo':{'name':'Roman autonomous activity','version':'1'},'agentCapabilities':{'loadSession':True},'authMethods':[]})
        elif method=='authenticate': reply(id,{})
        elif method in ('session/new','session/load'):
            reply(id,{'sessionId':m['params'].get('sessionId',str(uuid.uuid4())),'models':{'currentModelId':'grok-build','availableModels':[{'modelId':'grok-build','name':'Autonomous activity'}]}})
        elif method=='session/prompt':
            if worker and worker.is_alive(): send({'jsonrpc':'2.0','id':id,'error':{'code':-32000,'message':'Already observing a wake'}})
            else:
                cancelled.clear(); worker=threading.Thread(target=watch,args=(m,),daemon=True); worker.start()
        elif method=='session/cancel': cancelled.set()
        elif method in ('session/set_model','session/set_mode'): reply(id,{})
        elif id is not None: send({'jsonrpc':'2.0','id':id,'error':{'code':-32601,'message':'Unsupported method'}})
    cancelled.set()

def service(root):
    db=sqlite3.connect(root/'state/t3-activity.sqlite3'); db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS submitted(session TEXT PRIMARY KEY, role TEXT, command TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS retries(session TEXT PRIMARY KEY, attempts INTEGER NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS commands(id TEXT PRIMARY KEY, body TEXT NOT NULL)')
    operator=(root/'secrets/operator.key').read_text().strip()
    token=None; expires=0
    while True:
        try:
            if time.time()>expires:
                cmd=[APP+'/MacOS/T3 Code (Alpha)',APP+'/Resources/app.asar/apps/server/dist/bin.mjs','auth','session','issue','--base-dir',str(Path.home()/'.t3'),'--ttl','2h','--label','Roman autonomous activity','--json']
                issued=json.loads(subprocess.run(cmd,env={**os.environ,'ELECTRON_RUN_AS_NODE':'1'},capture_output=True,text=True,check=True,timeout=30).stdout)
                token=issued['token']; expires=time.time()+3600
            runtime=json.loads((Path.home()/'.t3/userdata/server-runtime.json').read_text())
            base=runtime['origin']
            snapshot=request(base+'/api/orchestration/snapshot',token)
            project='roman-autonomous-activity'
            def dispatch(command):
                db.execute('INSERT OR IGNORE INTO commands VALUES(?,?)',(command['commandId'],json.dumps(command)))
                db.commit()
                durable=json.loads(db.execute('SELECT body FROM commands WHERE id=?',(command['commandId'],)).fetchone()[0])
                return request(base+'/api/orchestration/dispatch',token,durable)
            if not any(p['id']==project for p in snapshot['projects']):
                dispatch({'type':'project.create','commandId':'roman-activity-project-v1','projectId':project,'title':'Roman autonomous activity','workspaceRoot':str(root/'workspaces'),'createdAt':now()})
            sessions=request('http://127.0.0.1:18431/v1/sessions',operator)
            for role in ROLES:
                thread='roman-activity-'+role
                existing=next((t for t in snapshot['threads'] if t['id']==thread),None)
                if not existing:
                    dispatch({'type':'thread.create','commandId':thread+'-create-v1','threadId':thread,'projectId':project,'title':role.title()+' · autonomous activity','modelSelection':{'instanceId':thread,'model':'grok-build'},'runtimeMode':'full-access','interactionMode':'default','branch':None,'worktreePath':None,'createdAt':now()})
                elif existing.get('session',{} ) and existing['session'].get('status') in ('running','connecting'): continue
                # If T3/provider transport failed, replay the read-only view
                # with bounded retries. Never replay the underlying bot input.
                if existing and existing.get('session') and existing['session'].get('status')=='error':
                    previous=db.execute("SELECT session FROM submitted WHERE role=? AND command!='baseline' ORDER BY rowid DESC LIMIT 1",(role,)).fetchone()
                    if previous:
                        source=next((s for s in sessions if s['id']==previous[0] and s['principal']=='t3-'+role),None)
                        attempts=db.execute('SELECT attempts FROM retries WHERE session=?',previous).fetchone()
                        attempt=attempts[0] if attempts else 0
                        if source and attempt<3:
                            command='roman-observe-retry-'+source['id']+'-'+str(attempt)
                            dispatch({'type':'thread.session.stop','commandId':command+'-stop','threadId':thread,'createdAt':now()})
                            dispatch({'type':'thread.turn.start','commandId':command,'threadId':thread,'message':{'messageId':command,'role':'user','text':f"Observe autonomous {role} session {source['id']}",'attachments':[]},'modelSelection':{'instanceId':thread,'model':'grok-build'},'runtimeMode':'full-access','interactionMode':'default','createdAt':now()})
                            db.execute('INSERT OR REPLACE INTO retries VALUES(?,?)',(source['id'],attempt+1)); db.commit(); continue
                candidates=[s for s in sessions if s['akh']==role and s['principal']=='t3-'+role and s['turns'] and not db.execute('SELECT 1 FROM submitted WHERE session=?',(s['id'],)).fetchone()]
                # On first install show the latest wake, avoiding replaying the entire old backlog.
                if not db.execute('SELECT 1 FROM submitted WHERE role=?',(role,)).fetchone() and len(candidates)>1:
                    for old in sorted(candidates,key=lambda s:s['created_at'])[:-1]: db.execute('INSERT OR IGNORE INTO submitted VALUES(?,?,?)',(old['id'],role,'baseline'))
                    db.commit(); candidates=sorted(candidates,key=lambda s:s['created_at'])[-1:]
                if candidates:
                    source=min(candidates,key=lambda s:s['created_at']); sid=source['id']; command='roman-observe-'+sid
                    dispatch({'type':'thread.turn.start','commandId':command,'threadId':thread,'message':{'messageId':command,'role':'user','text':f'Observe autonomous {role} session {sid}','attachments':[]},'modelSelection':{'instanceId':thread,'model':'grok-build'},'runtimeMode':'full-access','interactionMode':'default','createdAt':now()})
                    db.execute('INSERT OR IGNORE INTO submitted VALUES(?,?,?)',(sid,role,command)); db.commit()
            print('Activity projection checked',now(),flush=True)
        except Exception as error:
            print('Activity projection unavailable:',type(error).__name__,getattr(error,'code',''),flush=True)
        time.sleep(10)

if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    role=Path(sys.argv[0]).name.removeprefix('roman-activity-')
    if '--service' in sys.argv: service(root)
    elif '--version' in sys.argv: print('Roman activity ACP 1.0')
    elif 'models' in sys.argv: print('You are logged in.\nDefault model: grok-build\nAvailable models:\n * grok-build (default)')
    elif role in ROLES: acp(root,role)
    else: raise SystemExit('Unknown activity role')
