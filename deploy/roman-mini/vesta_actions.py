"""Credential-separated Vesta publishing, cluster mutations and Grafana MCP."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request
import uuid

REMOTE='git@github.com:toasterson/talos.git'
GRAFANA='http://127.0.0.1:18441/mcp'

def grafana_rpc(method,params):
    headers={'Content-Type':'application/json','Accept':'application/json, text/event-stream'}
    def call(method,params,ident):
        q={'jsonrpc':'2.0','method':method,'params':params}
        if ident is not None:q['id']=ident
        with urllib.request.urlopen(urllib.request.Request(GRAFANA,data=json.dumps(q).encode(),headers=headers),timeout=90) as response:
            sid=response.headers.get('Mcp-Session-Id')
            if sid:headers['Mcp-Session-Id']=sid
            raw=response.read(4*1024*1024+1).decode()
            if len(raw)>4*1024*1024:raise ValueError('Grafana response exceeds limit')
            if not raw:return {}
            result=json.loads(raw) if raw.lstrip().startswith('{') else json.loads(next(line[5:].strip() for line in raw.splitlines() if line.startswith('data:')))
            if 'error' in result:raise RuntimeError('Grafana MCP rejected request: '+json.dumps(result['error']))
            return result['result']
    call('initialize',{'protocolVersion':'2025-03-26','capabilities':{},'clientInfo':{'name':'Vesta','version':'1'}},1)
    headers['MCP-Protocol-Version']='2025-03-26'
    try:
        call('notifications/initialized',{},None)
        return call(method,params,2)
    finally:
        try:urllib.request.urlopen(urllib.request.Request(GRAFANA,headers=headers,method='DELETE'),timeout=5).close()
        except Exception:pass

def branch_name(branch):
    if not isinstance(branch,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,150}',branch) or any(x in branch for x in ('..','//','/.')) or branch.endswith(('.','/','.lock')):raise ValueError('Invalid branch name')
    return branch

def publish(config,args):
    if set(args)-{'revision','branch','dry_run'}:raise ValueError('Unknown publishing argument')
    revision=args['revision'];branch=branch_name(args['branch']);dry=args.get('dry_run',False)
    if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Use an exact full commit SHA')
    if not isinstance(dry,bool):raise ValueError('dry_run must be boolean')
    # Export objects from the sandbox; never execute host Git in a bot-writable
    # checkout or import its hooks/config/remote settings into the host process.
    docker=['/opt/homebrew/bin/docker','exec','--workdir','/workspace',config['container'],'git','-c','safe.directory=/workspace','-c','core.hooksPath=/dev/null']
    name='/tmp/vesta-export-'+uuid.uuid4().hex+'.bundle'
    def checked(argv,**kw):return subprocess.run(argv,check=True,capture_output=True,timeout=90,**kw)
    checked(docker+['update-ref','refs/vesta/export',revision])
    try:
        checked(docker+['bundle','create',name,'refs/vesta/export'])
        with tempfile.TemporaryDirectory(prefix='vesta-publish-') as directory:
            os.chmod(directory,0o700);bundle=Path(directory)/'export.bundle';repo=Path(directory)/'repo.git'
            with bundle.open('wb') as output:
                subprocess.run(['/opt/homebrew/bin/docker','exec',config['container'],'head','-c','104857600',name],stdout=output,stderr=subprocess.PIPE,timeout=90,check=True)
            env={**os.environ,'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null','GIT_TERMINAL_PROMPT':'0','GIT_SSH_COMMAND':'/usr/bin/ssh -o BatchMode=yes'}
            for key in list(env):
                if key.startswith('GIT_') and key not in ('GIT_CONFIG_NOSYSTEM','GIT_CONFIG_GLOBAL','GIT_TERMINAL_PROMPT','GIT_SSH_COMMAND'):del env[key]
            checked(['/usr/bin/git','init','--bare',str(repo)],env=env)
            git=['/usr/bin/git','--git-dir='+str(repo),'-c','core.hooksPath=/dev/null']
            checked(git+['fetch',str(bundle),'refs/vesta/export:refs/import/vesta'],env=env)
            actual=checked(git+['rev-parse','refs/import/vesta'],env=env).stdout.decode().strip()
            if actual!=revision:raise ValueError('Exported revision changed during publication')
            result=subprocess.run(git+['push','--porcelain','--no-verify']+(['--dry-run'] if dry else [])+[REMOTE,revision+':refs/heads/'+branch],env=env,capture_output=True,timeout=90)
            return {'exit_code':result.returncode,'stdout':result.stdout.decode()[-65536:],'stderr':result.stderr.decode()[-65536:],'revision':revision,'branch':branch,'dry_run':dry,'remote':REMOTE}
    finally:
        subprocess.run(['/opt/homebrew/bin/docker','exec',config['container'],'rm','-f',name],capture_output=True,timeout=10)

def write_args(args,context):
    from vesta_ops import identifier
    if set(args)-{'action','resource','name','namespace','manifest','patch','patch_type','replicas','dry_run'}:raise ValueError('Unknown mutation argument')
    action=args['action'];dry=args.get('dry_run',False)
    if not isinstance(dry,bool):raise ValueError('dry_run must be boolean')
    cmd=['/opt/homebrew/bin/kubectl','--context',context,'--request-timeout=30s'];body=None
    if action=='apply':
        manifest=args['manifest']
        if not isinstance(manifest,dict) or not manifest.get('apiVersion') or not manifest.get('kind'):raise ValueError('A Kubernetes JSON object is required')
        body=json.dumps(manifest).encode()
        if len(body)>512*1024:raise ValueError('Manifest too large')
        cmd += ['apply','--field-manager=vesta','-f','-','-o','name']
    elif action in ('patch','delete','scale','rollout_restart','cordon','uncordon'):
        resource=identifier(args['resource']);name=identifier(args['name'])
        if action=='patch':
            patch=args['patch'];kind=args.get('patch_type','merge')
            if kind not in ('merge','strategic','json') or not isinstance(patch,(dict,list)):raise ValueError('Invalid patch')
            cmd += ['patch',resource,name,'--type',kind,'--patch',json.dumps(patch),'-o','name']
        elif action=='delete':cmd += ['delete',resource,name,'--wait=false','-o','name']
        elif action=='scale':
            replicas=args['replicas']
            if type(replicas)!=int or not 0<=replicas<=1000:raise ValueError('Invalid replica count')
            cmd += ['scale',resource,name,'--replicas='+str(replicas)]
        elif action=='rollout_restart':
            if resource not in ('deployments','statefulsets','daemonsets'):raise ValueError('Invalid rollout resource')
            cmd += ['rollout','restart',resource+'/'+name]
        else:
            if resource!='nodes':raise ValueError('Cordon operations require nodes')
            cmd += [action,name]
    else:raise ValueError('Unsupported cluster mutation')
    if args.get('namespace'):cmd += ['-n',identifier(args['namespace'])]
    if dry:cmd += ['--dry-run=server']
    return cmd,body

ACTION_TOOLS=[
 {'name':'gitops_publish','description':'Push an exact sandbox commit SHA to a branch of toasterson/talos using brokered credentials. Non-force push only, remote protection applies. Prefer vesta/* fix branches, preserve review gates; main updates must meet the repository delivery policy. Run validations first. dry_run checks publication without updating refs. Inspect remote outcome before retrying uncertain writes.','inputSchema':{'type':'object','properties':{'revision':{'type':'string','pattern':'^[0-9a-f]{40}$'},'branch':{'type':'string'},'dry_run':{'type':'boolean','default':False}},'required':['revision','branch'],'additionalProperties':False}},
 {'name':'homelab_write','description':'Authorized live homelab cluster mutation: apply a JSON manifest, patch/delete a named resource, scale or restart a workload, cordon/uncordon a node. Credentials stay in the host broker. Read current evidence and runbooks, limit blast radius, respect live owners, mirror persistent desired state in GitOps and verify recovery. Use dry_run=true for server validation where supported. No shell or kubectl exec.','inputSchema':{'type':'object','properties':{'action':{'type':'string','enum':['apply','patch','delete','scale','rollout_restart','cordon','uncordon']},'resource':{'type':'string'},'name':{'type':'string'},'namespace':{'type':'string'},'manifest':{'type':'object'},'patch':{'oneOf':[{'type':'object'},{'type':'array'}]},'patch_type':{'type':'string','enum':['merge','strategic','json']},'replicas':{'type':'integer','minimum':0,'maximum':1000},'dry_run':{'type':'boolean','default':False}},'required':['action'],'additionalProperties':False}}
]
