"""Finite spatial fixture preparation, with explicitly authored interpretation."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
FONT=Path('/System/Library/Fonts/Supplemental/Arial.ttf')

def descriptors():
 return [dict(id='table',split='development',mechanism='row_column_join',columns=['ASSET','SITE','NODE'],rows=[['api','west','nodealpha'],['api','east','nodebeta']],goal='Resolve api at west from same-row column associations'),dict(id='procedure',split='validation',mechanism='vertical_precedence',rows=[['stop'],['configure'],['start']],goal='Recover observed operation precedence; executor must independently validate prerequisites')]

def render(desc,path,control='complete'):
 image=Image.new('RGB',(1100,500),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype(str(FONT),36)
 if desc['id']=='table':
  for col,text in enumerate(desc['columns']):draw.text((60+col*330,40),text,fill='black',font=font)
  for row,values in enumerate(desc['rows']):
   for col,text in enumerate(values):
    if control=='missing_structure' and row==0 and col==2:continue
    draw.text((60+col*330,160+row*100),text,fill='black',font=font)
  if control=='neighbor_distractor':draw.text((60,390),'NOTE nodebeta',fill='black',font=font)
 else:
  for row,values in enumerate(desc['rows']):
   if control=='missing_structure' and row==1:continue
   draw.text((250,60+row*110),values[0],fill='black',font=font)
  if control=='neighbor_distractor':draw.text((720,170),'start',fill='black',font=font)
 image.save(path)

def interpret(words,kind):
 """Input is observed word text/confidence/bounds only, never fixture descriptors."""
 good=[w for w in words if w['text'].strip()]
 if any(w['confidence']<80 for w in good):return dict(facts=[],error='Low-confidence observation')
 if kind=='table':
  headers={label:[w for w in good if w['text']==label] for label in ['ASSET','SITE','NODE']}
  if any(len(v)!=1 for v in headers.values()):return dict(facts=[],error='Ambiguous/missing header')
  centers=[headers[k][0]['left']+headers[k][0]['width']/2 for k in headers]
  if not centers[0]<centers[1]<centers[2]:return dict(facts=[],error='Header order unsupported')
  cells=[w for w in good if 120<=w['top']<350] # declared fixture body region, not generic layout inference
  groups={}
  for w in cells:
   row=round((w['top']-160)/100);col=min(range(3),key=lambda c:abs((w['left']+w['width']/2)-centers[c]))
   groups.setdefault(row,{}).setdefault(col,[]).append(w)
  if set(groups)!={0,1} or any(set(g)!={0,1,2} or any(len(v)!=1 for v in g.values()) for g in groups.values()):return dict(facts=[],error='Incomplete or ambiguous row')
  facts=[]
  for i,g in sorted(groups.items()):
   subject='row-'+str(i)
   facts.extend(dict(subject=subject,predicate=p,object=g[c][0]['text']) for c,p in enumerate(['asset','site','node']))
  return dict(facts=facts,assistance='Header association and fixed body region; two-row fixture schema')
 if kind=='procedure':
  body=sorted([w for w in good if 200<=w['left']<600],key=lambda w:w['top'])
  if len(body)!=3 or len({w['top'] for w in body})!=3:return dict(facts=[],error='Missing or ambiguous procedure structure')
  return dict(facts=[dict(subject='procedure',predicate='step-'+str(i),object=w['text']) for i,w in enumerate(body)],assistance='Fixed procedure column and three-step schema; observed vertical order')
 raise ValueError('Unknown layout contract')


def breadth_descriptor(split,seed):
 """Private rendering/expectation inputs only; never candidate observations."""
 import random,copy
 rng=random.Random(seed)
 if split not in ('development','validation'):raise ValueError('Unknown split')
 spans=[]
 def add(identity,text,x,y,w=150,h=40):
  spans.append(dict(id=identity,text=text,box=[x,y,w,h]));return identity
 if split=='development':
  groups=['cluster-'+str(i) for i in range(rng.randint(2,3))];rows=['asset-'+str(i) for i in range(rng.randint(3,5))];rng.shuffle(groups);rng.shuffle(rows)
  selected=rng.choice(groups);row=rng.choice(rows);initial='node-a';later='node-b'
  for i,g in enumerate(groups):
   x=200+i*340;add('group-'+g,g,x,20,320);add(g+'-node','NODE',x,80);add(g+'-state','STATE',x+170,80)
  for j,a in enumerate(rows):
   y=150+j*60;add('row-'+a,a,20,y)
   for i,g in enumerate(groups):
    x=200+i*340;add(a+'-'+g+'-node',initial if (a,g)==(row,selected) else 'other-'+str(j)+str(i),x,y);add(a+'-'+g+'-state','READY',x+170,y)
  ids=['group-'+selected,selected+'-node',selected+'-state','row-'+row,row+'-'+selected+'-node',row+'-'+selected+'-state']
  changed=copy.deepcopy(spans);next(s for s in changed if s['id']==ids[-2])['text']=later
  goal=dict(group=selected,asset=row,instruction='Use NODE under the named group for this asset only when its STATE is READY')
  missing=ids[0];conflict=dict(id='conflicting-cell',text='conflicting-node',box=next(s['box'][:] for s in spans if s['id']==ids[-2]))
  mechanism='table_multilevel_header'
 else:
  labels=['branch-a','branch-b','branch-c'][:rng.randint(2,3)];rng.shuffle(labels);initial_label,later_label=rng.sample(labels,2);targets={label:'node-'+str(i) for i,label in enumerate(labels)}
  add('procedure-title','CONSULT PANEL STATUS',20,20,450)
  for i,label in enumerate(labels):
   add('condition-'+label,'IF '+label,20,100+i*80,220);add('action-'+label,targets[label],270,100+i*80,200)
  add('panel-header','STATUS',650,20,200);add('panel-value',initial_label,650,90,200)
  initial=targets[initial_label];later=targets[later_label];changed=copy.deepcopy(spans);next(s for s in changed if s['id']=='panel-value')['text']=later_label
  ids=['procedure-title','condition-'+initial_label,'action-'+initial_label,'panel-header','panel-value'];goal=dict(instruction='Join the procedure branch to the separate STATUS panel and use its target')
  missing='panel-value';conflict=dict(id='contradictory-status',text=later_label,box=[650,150,200,40]);mechanism='procedure_branch_cross_reference'
 return dict(schema='akh-perception-breadth-descriptor-v1',split=split,seed=seed,mechanism=mechanism,public_start=goal,examiner=dict(canvas=[1280,520],initial_spans=spans,later_spans=changed,later_trigger='after_initial_effect_verified',expected_initial=initial,expected_later=later,initial_required_spans=ids,missing_region=missing,contradictory_span=conflict),assistance='Authored cell/region contract; span IDs are examiner-only; future OCR must supply actual text, confidence and image coordinates',unimplemented=['rendering new layout','actual OCR span association','native graph and isolated effect integration'],new_world_credit=0)


def descriptor_answer(desc,phase='initial',control='complete'):
 """Symbolic descriptor oracle; not an OCR/parser capability measurement."""
 import copy
 e=desc['examiner'];spans=copy.deepcopy(e['initial_spans' if phase=='initial' else 'later_spans'])
 if control=='missing_region':spans=[s for s in spans if s['id']!=e['missing_region']]
 elif control=='contradictory_status':spans.append(copy.deepcopy(e['contradictory_span']))
 elif control!='complete':raise ValueError('Unknown control')
 def one(test):
  matches=[s for s in spans if test(s)]
  if len(matches)!=1:raise ValueError('Missing or ambiguous source span')
  return matches[0]
 try:
  if desc['split']=='development':
   goal=desc['public_start'];group=one(lambda s:s['text']==goal['group']);row=one(lambda s:s['text']==goal['asset'])
   x,y,w,h=group['box'];headers=[s for s in spans if x<=s['box'][0]<x+w and s['box'][1]==80]
   node=[s for s in headers if s['text']=='NODE'];state=[s for s in headers if s['text']=='STATE']
   if len(node)!=1 or len(state)!=1:raise ValueError('Header association missing')
   cell=one(lambda s:s['box'][:2]==[node[0]['box'][0],row['box'][1]])
   ready=one(lambda s:s['box'][:2]==[state[0]['box'][0],row['box'][1]])
   if ready['text']!='READY':raise ValueError('Not ready')
   return dict(answer=cell['text'],spans=[group['id'],node[0]['id'],state[0]['id'],row['id'],cell['id'],ready['id']])
  title=one(lambda s:s['text']=='CONSULT PANEL STATUS');header=one(lambda s:s['text']=='STATUS');value=one(lambda s:s['box'][0]==header['box'][0] and s['box'][1]>header['box'][1]);condition=one(lambda s:s['text']=='IF '+value['text']);action=one(lambda s:s['box'][0]==270 and s['box'][1]==condition['box'][1])
  return dict(answer=action['text'],spans=[title['id'],condition['id'],action['id'],header['id'],value['id']])
 except ValueError as error:return dict(answer=None,spans=[],error=str(error))


def render_breadth(desc,path,phase='initial',control='complete'):
 """Render private fixture inputs; only image bytes cross into OCR."""
 import copy
 e=desc['examiner'];spans=copy.deepcopy(e['initial_spans' if phase=='initial' else 'later_spans'])
 if control=='missing_region':spans=[s for s in spans if s['id']!=e['missing_region']]
 elif control=='contradictory_status':spans.append(copy.deepcopy(e['contradictory_span']))
 elif control!='complete':raise ValueError('Unknown control')
 image=Image.new('RGB',tuple(e['canvas']),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype(str(FONT),22)
 for span in spans:draw.text(tuple(span['box'][:2]),span['text'],fill='black',font=font)
 image.save(path)


def interpret_breadth(words,goal,kind):
 """Observed OCR words only. Fixed cell bands are declared layout assistance."""
 good=[dict(w,index=i) for i,w in enumerate(words) if w['text'].strip()]
 if not good or any(w['confidence']<80 for w in good):return dict(answer=None,spans=[],error='Missing or low-confidence OCR')
 # Cells are grouped by geometry, never by private descriptor/span identity.
 def cell(x,y,width=150,height=45):
  ws=[w for w in good if x<=w['left']<x+width and y<=w['top']<y+height]
  if not ws:raise ValueError('Missing observed cell')
  return ' '.join(w['text'] for w in sorted(ws,key=lambda w:(w['top']//20,w['left']))),[w['index'] for w in ws]
 try:
  if kind=='table_multilevel_header':
   groups=[]
   for x in [200,540,880]:
    try:text,ids=cell(x,20,320)
    except ValueError:continue
    if text==goal['group']:groups.append((x,ids))
   if len(groups)!=1:raise ValueError('Missing unique grouped header')
   x,ids=groups[0];node,ni=cell(x,80);state,si=cell(x+170,80)
   if node!='NODE' or state!='STATE':raise ValueError('Unsupported observed subheader')
   rows=[]
   for y in range(150,451,60):
    try:text,ri=cell(20,y)
    except ValueError:continue
    if text==goal['asset']:rows.append((y,ri))
   if len(rows)!=1:raise ValueError('Missing unique observed row')
   y,ri=rows[0];target,ti=cell(x,y);ready,vi=cell(x+170,y)
   import re
   if ready!='READY' or not re.fullmatch(r'node-[a-z]+',target):raise ValueError('Ambiguous target or readiness')
   return dict(answer=target,spans=ids+ni+si+ri+ti+vi)
  if kind!='procedure_branch_cross_reference':raise ValueError('Unknown mechanism')
  title,ids=cell(20,20,450);header,hi=cell(650,20,200)
  if title!='CONSULT PANEL STATUS' or header!='STATUS':raise ValueError('Missing observed cross-reference')
  panel=[w for w in good if 650<=w['left']<850 and w['top']>=70]
  # Status must occupy one observed line; another line is unresolved conflict.
  if not panel or max(w['top'] for w in panel)-min(w['top'] for w in panel)>20:raise ValueError('Missing or conflicting status')
  value,vi=cell(650,90,200);matches=[]
  for y in [100,180,260]:
   try:condition,ci=cell(20,y,220)
   except ValueError:continue
   if condition=='IF '+value:
    target,ti=cell(270,y,200);matches.append((target,ci+ti))
  if len(matches)!=1:raise ValueError('Missing unique observed branch')
  target,si=matches[0]
  import re
  if not re.fullmatch(r'node-[0-9]+',target):raise ValueError('Invalid observed target')
  return dict(answer=target,spans=ids+hi+vi+si)
 except ValueError as error:return dict(answer=None,spans=[],error=str(error))


def interpret_procedure_support(words):
 """Confidence>=80 on all decision support; unused action values may be unreadable.

Declared fixed regions only. Every branch condition remains required so low
confidence cannot conceal another matching branch. No OCR text normalization.
 """
 good=[dict(w,index=i) for i,w in enumerate(words) if w['text'].strip()]
 def cell(x,y,width,height=45,required=True):
  ws=[w for w in good if x<=w['left']<x+width and y<=w['top']<y+height]
  if not ws and not required:return None,[]
  if not ws or any(w['confidence']<80 for w in ws):raise ValueError('Missing or low-confidence required support')
  return ' '.join(w['text'] for w in sorted(ws,key=lambda w:(w['top']//20,w['left']))),[w['index'] for w in ws]
 try:
  title,ids=cell(20,20,450);header,hi=cell(650,20,200)
  if title!='CONSULT PANEL STATUS' or header!='STATUS':raise ValueError('Missing cross-reference')
  panel=[w for w in good if 650<=w['left']<850 and w['top']>=70]
  if not panel or max(w['top'] for w in panel)-min(w['top'] for w in panel)>20:raise ValueError('Missing or conflicting status')
  value,vi=cell(650,90,200);matches=[];all_conditions=[]
  for y in [100,180,260]:
   condition,ci=cell(20,y,220,required=False)
   if condition is None:continue
   if not condition.startswith('IF '):raise ValueError('Unsupported condition')
   all_conditions+=ci
   if condition=='IF '+value:
    target,ti=cell(270,y,200);matches.append((target,ti))
  if len(matches)!=1:raise ValueError('Missing unique branch')
  target,ti=matches[0]
  import re
  if not re.fullmatch(r'node-[0-9]+',target):raise ValueError('Invalid target')
  return dict(answer=target,spans=ids+hi+vi+all_conditions+ti,assistance='Fixed regions; confidence80 on title/panel/all conditions/selected action; unused action values excluded')
 except ValueError as error:return dict(answer=None,spans=[],error=str(error))
