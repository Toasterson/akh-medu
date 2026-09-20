"""Archive user-facing messages and primary sources; no tracker mutations."""
import concurrent.futures, datetime, hashlib, json, pathlib, shutil, subprocess, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
SESSION = pathlib.Path('/Users/toasty/.codex/sessions/2026/09/16/rollout-2026-09-16T21-37-53-01a0abb9-c2cb-71f0-afc5-2c458c585078.jsonl')
def dump(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
messages=[]; started=False
for line in SESSION.read_text().splitlines():
 o=json.loads(line); p=o.get('payload',{})
 if o.get('type')!='response_item' or p.get('type')!='message' or p.get('role') not in ('user','assistant'): continue
 if p.get('role')=='assistant' and p.get('channel') not in (None,'final','commentary'): continue
 s='\n'.join(b['text'] for b in p.get('content',[]) if b.get('type') in ('input_text','output_text'))
 if s.startswith('Today it rains papers'): started=True
 if started and s: messages.append({'role':p['role'],'channel':p.get('channel'),'text':s,'timestamp':o.get('timestamp')})
assert any('Yes—highly relevant' in m['text'] for m in messages)
dump(ROOT/'transcripts/conversation.json',messages)
md='# Dream-RSI / ADAM / Kaine — verbatim conversation\n\nSnapshot of user-facing messages through '+messages[-1]['timestamp']+'. Excludes system/developer instructions, private reasoning and tool traffic. Later ingestion receipts are separate. Historical analysis is preserved without editorial changes.\n\n'
for i,m in enumerate(messages,1): md+=f"## {i}. {m['role']} ({m.get('channel') or 'user-facing'})\n\n{m['text']}\n\n"
(ROOT/'transcripts/conversation.md').write_text(md)
assert json.loads((ROOT/'transcripts/conversation.json').read_text())==messages
assert all(m['text'] in md for m in messages)
analysis=next(m['text'] for m in messages if m['role']=='assistant' and '**Yes—highly relevant' in m['text'])
(ROOT/'ANALYSIS.md').write_text(analysis+'\n')
urls={
 'papers/dream-rsi-2609.14858v1.pdf':'https://arxiv.org/pdf/2609.14858v1',
 'papers/dream-rsi-2609.14858v1.html':'https://arxiv.org/html/2609.14858v1',
 'sources/arxiv-abstract.html':'https://arxiv.org/abs/2609.14858v1',
 'sources/project.html':'https://dream-rsi.com/',
 'sources/repository-readme.md':'https://raw.githubusercontent.com/zhengkid/Dream-RSI/main/README.md',
 'sources/tweet-fxtwitter.json':'https://api.fxtwitter.com/Dr_Singularity/status/2100204780010222076'}
def fetch(item):
 rel,url=item
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=60) as r: data=r.read(); final=r.url; mime=r.headers.get('Content-Type')
 (ROOT/rel).write_bytes(data)
 return {'path':rel,'requested_url':url,'resolved_url':final,'content_type':mime,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex: records=list(ex.map(fetch,urls.items()))
assert (ROOT/'papers/dream-rsi-2609.14858v1.pdf').read_bytes().startswith(b'%PDF')
subprocess.run(['pdftotext','-layout',str(ROOT/'papers/dream-rsi-2609.14858v1.pdf'),str(ROOT/'papers/dream-rsi-2609.14858v1.txt')],check=True)
info=subprocess.check_output(['pdfinfo',str(ROOT/'papers/dream-rsi-2609.14858v1.pdf')],text=True)
(ROOT/'papers/pdfinfo.txt').write_text(info)
for rel in ['docs/ai/decisions/110-kaine-character-guided-reward-live-environment.md','docs/ai/decisions/111-kaine-architecture-ladder.md','docs/ai/plans/2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md']:
 shutil.copyfile(REPO/rel,ROOT/'context'/pathlib.Path(rel).name)
(ROOT/'context/git-head.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True))
dump(ROOT/'sources/retrieval-manifest.json',records)
dump(ROOT/'context/transcript-provenance.json',{'session':str(SESSION),'session_id':'01a0abb9-c2cb-71f0-afc5-2c458c585078','message_count':len(messages),'cutoff':messages[-1]['timestamp'],'verification':'JSON round-trip equality and exact message inclusion in Markdown verified; Only response_item message output_text retained; reasoning items excluded. Legacy source omits channel metadata.'})
print(json.dumps({'messages':len(messages),'downloads':records,'pdfinfo':info}))
