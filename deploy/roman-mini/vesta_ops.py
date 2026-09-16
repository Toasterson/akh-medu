#!/usr/bin/env python3
"""Vesta-only MCP broker: host diagnostics and commands in a fixed container.

No shell runs on the host. No host credential, Docker socket, or home mount is
passed into the container. Kubernetes mutations are not exposed by this API.
"""
import hmac
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESOURCES=['pods','deployments','statefulsets','daemonsets','nodes','events','services','endpoints','persistentvolumeclaims','persistentvolumes','storageclasses','jobs','cronjobs','helmreleases','kustomizations','gitrepositories','helmrepositories','buckets']
TOOLS=[
 {'name':'gitops_exec','description':'Run a bounded shell command in Vesta\'s isolated Linux GitOps sandbox at /workspace (toasterson/talos checkout). Read/edit files, git diff/status, Python, jq, yamllint and kubectl kustomize are available. No network, host home, host credentials or Docker socket. Changes persist. Read AGENTS.md and applicable runbooks first. Do not claim deployment from local changes; share a tested patch through Sokar/Anima/Board for the GitOps publication workflow.', 'inputSchema':{'type':'object','properties':{'command':{'type':'string'},'timeout_seconds':{'type':'integer','minimum':1,'maximum':120}},'required':['command'],'additionalProperties':False}},
 {'name':'homelab_inspect','description':'Read live homelab Kubernetes status, describe resources or bounded pod logs. Read-only fixed kubectl operations; no Secrets, exec, apply, delete, or arbitrary flags. Use GitOps for desired-state fixes. Missing or truncated evidence is not health.', 'inputSchema':{'type':'object','properties':{'action':{'type':'string','enum':['get','describe','logs']},'resource':{'type':'string','enum':RESOURCES},'name':{'type':'string'},'namespace':{'type':'string'},'container':{'type':'string'}},'required':['action','resource'],'additionalProperties':False},'annotations':{'readOnlyHint':True}}
]

def bounded(argv, timeout):
    proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    buffers=[bytearray(),bytearray()]; truncated=[False,False]
    def drain(stream,index):
        while True:
            chunk=stream.read(4096)
            if not chunk:
                stream.close();break
            room=max(0,65536-len(buffers[index]));buffers[index].extend(chunk[:room])
            truncated[index] |= len(chunk)>room
    threads=[threading.Thread(target=drain,args=(stream,i),daemon=True) for i,stream in enumerate((proc.stdout,proc.stderr))]
    for thread in threads:thread.start()
    try: proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill();proc.wait();raise RuntimeError('Operation timed out; inspect state before retrying effective commands')
    for thread in threads:thread.join(timeout=2)
    return {'exit_code':proc.returncode,'stdout':buffers[0].decode(errors='replace'),'stderr':buffers[1].decode(errors='replace'),'truncated':any(truncated)}

def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,252}',value):raise ValueError('Invalid Kubernetes identifier')
    return value

def inspect_args(args):
    if set(args)-{'action','resource','name','namespace','container'}:raise ValueError('Unknown argument')
    action=args['action'];resource=args['resource']
    if resource not in RESOURCES or action not in ('get','describe','logs'):raise ValueError('Read operation not permitted')
    cmd=['/opt/homebrew/bin/kubectl','--request-timeout=20s',action]
    if action=='logs':
        if resource!='pods' or not args.get('name') or not args.get('namespace'):raise ValueError('Pod logs require a pod name and namespace')
        cmd += [identifier(args['name']),'--tail=200','--limit-bytes=65536']
        if args.get('container'):cmd += ['-c',identifier(args['container'])]
    else:
        if args.get('container'):raise ValueError('Container applies only to logs')
        cmd += [resource]
        if args.get('name'):cmd += [identifier(args['name'])]
        if action=='get':cmd += ['-o','wide']
    cmd += ['-n',identifier(args['namespace'])] if args.get('namespace') else ['--all-namespaces']
    return cmd

def serve(root):
    token=(root/'secrets/vesta-ops.key').read_text().strip()
    config=json.loads((root/'config/vesta-ops.json').read_text())
    lock=threading.Lock()
    def execute(name,args):
        if name=='homelab_inspect':return bounded(inspect_args(args),30)
        if name!='gitops_exec' or set(args)-{'command','timeout_seconds'}:raise ValueError('Unknown tool or argument')
        command=args['command'];timeout=args.get('timeout_seconds',60)
        if not isinstance(command,str) or not 0<len(command)<=16384 or '\x00' in command:raise ValueError('Invalid command')
        if not isinstance(timeout,int) or isinstance(timeout,bool) or not 1<=timeout<=120:raise ValueError('Invalid timeout')
        with lock:
            # Inspect on every request so a replaced or weakened container fails
            # closed. The allowlist is immutable operator configuration.
            raw=subprocess.check_output(['/opt/homebrew/bin/docker','inspect',config['container']],timeout=10)
            state=json.loads(raw)[0];host=state['HostConfig'];mounts=state['Mounts']
            if state['Id']!=config['container_id'] or state['Image']!=config['image_id'] or not state['State']['Running']:
                raise ValueError('Sandbox identity changed or container stopped')
            if host['NetworkMode']!='none' or not host['ReadonlyRootfs'] or host['Privileged'] or state['Config']['User']!='501:20' or 'ALL' not in (host.get('CapDrop') or []) or 'no-new-privileges' not in (host.get('SecurityOpt') or []) or len(mounts)!=1 or mounts[0]['Source']!=config['checkout'] or mounts[0]['Destination']!='/workspace':
                raise ValueError('Sandbox confinement changed')
            return bounded(['/opt/homebrew/bin/docker','exec','--workdir','/workspace','--env','GIT_CONFIG_COUNT=1','--env','GIT_CONFIG_KEY_0=safe.directory','--env','GIT_CONFIG_VALUE_0=/workspace',config['container'],'timeout','--kill-after=5s',str(timeout),'/bin/sh','-lc',command],timeout+15)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,obj=None):
            data=json.dumps(obj).encode() if obj is not None else b''
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.reply(200,{'status':'listening','sandbox':config['container']}) if self.path=='/healthz' else self.reply(404)
        def do_POST(self):
            if self.path!='/mcp':return self.reply(404)
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):return self.reply(401)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:return self.reply(413)
                q=json.loads(self.rfile.read(length));method=q.get('method');ident=q.get('id')
                if method=='notifications/initialized':return self.reply(202)
                if method=='initialize':result={'protocolVersion':'2025-03-26','capabilities':{'tools':{}},'serverInfo':{'name':'vesta-gitops','version':'1'}}
                elif method=='tools/list':result={'tools':TOOLS}
                elif method=='tools/call':
                    try:
                        data=execute(q['params']['name'],q['params'].get('arguments',{}));result={'content':[{'type':'text','text':json.dumps(data)}],'isError':data['exit_code']!=0}
                    except Exception as error:result={'content':[{'type':'text','text':str(error)}],'isError':True}
                else:return self.reply(200,{'jsonrpc':'2.0','id':ident,'error':{'code':-32601,'message':'Unsupported operation'}})
                self.reply(200,{'jsonrpc':'2.0','id':ident,'result':result})
            except Exception:self.reply(400,{'error':'Invalid request'})
    ThreadingHTTPServer(('127.0.0.1',18437),Handler).serve_forever()

if __name__=='__main__':serve(Path(sys.argv[1]).resolve())
