#!/usr/bin/env python3
"""Fresh tool-free Vulkan report from guarded evidence; no old-session resumption."""
import argparse,copy,hashlib,json,os,secrets,socket,sqlite3,subprocess,time,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'training/lilith/coordination-20260922/model-run2'
OUT=SOURCE/'fresh-report-probe'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument('--no-encoder-batching',action='store_true');parser.add_argument('--capsule',action='store_true');a=parser.parse_args()
 if a.no_encoder_batching:OUT=SOURCE/'fresh-report-no-encoder'
 if a.capsule:OUT=SOURCE/'fresh-report-capsule'
 OUT.mkdir(exist_ok=False)
 db=sqlite3.connect('file:'+str(SOURCE/'coordination.sqlite')+'?mode=ro',uri=True)
 row=json.loads(db.execute('select body from assignments').fetchone()[0]);db.close()
 guards=json.loads((SOURCE/'report-recovery/guards.json').read_text())
 actual={'source_sha256':digest(row['source']),'proposal_sha256':digest(canonical(row['proposal'])),
  'verification_sha256':digest(canonical(row['verification'])),'review_sha256':digest(row['review']),
  'return_text_sha256':digest(row['return_request']['text'])}
 if actual!=guards or not row['verification']['passed']:raise ValueError('Evidence guard mismatch')
 write(OUT/'evidence.json',{'guards':guards,'proposal':row['proposal'],'verification':row['verification'],
  'review':row['review'],'request_text':row['return_request']['text']})
 request_text=row['return_request']['text']
 if a.capsule:
  v=row['verification']
  capsule={'schema':'lilith/report-capsule@1','objective':row['objective'],'target':row['target'],
   'command':v['command'],'passed':v['passed'],'exit_code':v['exit_code'],'timed_out':v['timed_out'],
   'cleanup':v['remote_cleanup_succeeded'],'guest':'OpenIndiana illumos-c4b084c090; akh-exec@127.0.0.1:10115',
   'original_modified':False,'review':row['review'],'artifacts':'./evidence.json'}
  request_text='Report this tested proposal briefly to the human. Evidence: '+canonical(capsule)
  write(OUT/'capsule.json',capsule)
  (OUT/'prompt.txt').write_text(request_text)
 old=Path.home()/'.local/share/akh-medu/lilith-dev-vulkan-20260921'
 private=Path.home()/'.local/share/akh-medu'/('lilith-report-'+uuid.uuid4().hex)
 private.mkdir(mode=0o700)
 for d in ('credentials','workspaces'):(private/d).mkdir(mode=0o700)
 names=('akhexecd','akhsbxd','akhrtd','akhsessd');configs={}
 for n in names:
  text=(old/(n+'.json')).read_text().replace(str(old),str(private))
  for source_port,target_port in ((18420,18520),(18421,18521),(18422,18522),(18423,18523)):
   text=text.replace('127.0.0.1:'+str(source_port),'127.0.0.1:'+str(target_port))
  configs[n]=json.loads(text)
 for n in ('operator',)+names:
  token=secrets.token_urlsafe(32)
  for suffix,value in [('token',token),('sha256',digest(token))]:
   p=private/'credentials'/(n+'.'+suffix);p.write_text(value+'\n');p.chmod(0o600)
 for config in configs.values():
  for p in config['principals']:p['akhs']=['lilith-report']
  host,port=config['listen'].rsplit(':',1)
  assert host=='127.0.0.1'
  with socket.socket() as s:s.bind((host,int(port)))
 profile=copy.deepcopy(configs['akhsessd']['akhs']['lilith-learner'])
 profile['tools']=[];profile['instructions']='You are Lilith. Concisely report only the supplied test evidence and reviewer opinion. No tools are available. Do not claim the original repository changed or infer broader release validation.'
 configs['akhsessd']['akhs']={'lilith-report':profile}
 runtime=configs['akhrtd']['runtimes']['lilith'];configs['akhrtd']['runtimes']={'lilith':runtime}
 runtime['timeout_secs']=180
 for p in (runtime['working_dir'],runtime['isolation']['directory']):Path(p).mkdir(parents=True,exist_ok=True)
 native=runtime['native_candle'];native.update(max_context_tokens=4096,max_output_tokens=192 if a.capsule else 768,max_iterations=1)
 for name in ('model','tokenizer'):
  if sha(native[name+'_path'])!=native[name+'_sha256']:raise ValueError('Artifact hash mismatch')
 write(OUT/'protocol.json',{'schema':'lilith/fresh-report-probe@1','driver_sha256':sha(__file__),
  'model':native,'binary_sha256':{n:sha(ROOT/'target/release'/n) for n in names},'evidence_guards':guards,
  'capsule':a.capsule,'request_text_sha256':digest(request_text),'profile':profile,'private_root':str(private),'prior_assignment_id':row['id'],
  'encoder_batching':not a.no_encoder_batching,'scope':'New tool-free session, exact saved user report payload, no prior conversation history. Original unknown turn remains unresolved. Not a driver fix or quality benchmark.'})
 for n,c in configs.items():write(private/(n+'.json'),c)
 token=(private/'credentials/operator.token').read_text().strip()
 def api(n,path,body=None):
  r=urllib.request.Request('http://'+configs[n]['listen']+path,
   data=None if body is None else json.dumps(body).encode(),
   headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
  with urllib.request.urlopen(r,timeout=15) as f:return json.load(f)
 procs=[];logs=[];result={'status':'running','original_unknown_turn_resolved':False,'truth_review':'pending'}
 write(OUT/'result.json',result)
 try:
  for n,c in configs.items():
   log=(OUT/(n+'.log')).open('w');logs.append(log)
   p=subprocess.Popen([str(ROOT/'target/release'/n),'--config',str(private/(n+'.json'))],cwd=ROOT,
    stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',
    AKH_VULKAN_PIPELINE_CACHE_ENTRIES='16384',AKH_VULKAN_BATCH_ENCODER='0' if a.no_encoder_batching else '1',VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json'))
   procs.append(p)
   for _ in range(100):
    if p.poll() is not None:raise RuntimeError(n+' exited')
    try:
     if api(n,'/healthz')['service_id']==c['service_id']:break
    except (OSError,ValueError):time.sleep(.2)
   else:raise TimeoutError(n+' readiness')
  sid=api('akhsessd','/v1/sessions',{'request_id':uuid.uuid4().hex,'akh':'lilith-report','workspace_profile':'local-dev'})['id']
  path='/v1/sessions/'+sid
  turn=api('akhsessd',path+'/inputs',{'request_id':uuid.uuid4().hex,'text':request_text})
  deadline=time.monotonic()+180
  while time.monotonic()<deadline:
   session=api('akhsessd',path);current=next(t for t in session['turns'] if t['id']==turn['id'])
   if current['status'] in ('completed','failed','cancelled','unknown'):break
   time.sleep(.4)
  else:raise TimeoutError('report deadline')
  write(OUT/'session.json',session)
  records=api('akhsessd',path+'/tools')['records'];write(OUT/'tools.json',records)
  if records:raise RuntimeError('Tool-free report used a tool')
  if current['status']!='completed' or not current.get('output'):raise RuntimeError('Report did not complete')
  result.update(status='completed',report=current['output'],session_id=sid,turn_id=turn['id'])
 except BaseException as error:
  result.update(status='interrupted',error=type(error).__name__+': '+str(error));raise
 finally:
  for p in reversed(procs):
   if p.poll() is None:p.terminate()
  for p in procs:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for log in logs:log.close()
  result['services_stopped']=all(p.poll() is not None for p in procs);write(OUT/'result.json',result)

if __name__=='__main__':main()
