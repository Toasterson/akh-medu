#!/usr/bin/env python3
"""Install Janus-only Astra broker, preserving all existing provider grants."""
import json
from pathlib import Path
import plistlib
import secrets
import shutil
import sys
from janus_escalation import TOOLS, T3
root=Path(sys.argv[1]).resolve();source=Path(__file__).resolve().parent
# Existing registered T3 project IDs; broker never accepts arbitrary paths.
projects={p['title']:p['id'] for p in T3().call('/api/orchestration/snapshot')['projects'] if p['id']!='roman-autonomous-activity'}
p=root/'config/janus-escalation.json'
if not p.exists():p.write_text(json.dumps({'projects':projects},indent=2)+'\n');p.chmod(0o600)
key=root/'secrets/janus-escalation.key'
if not key.exists():
 with key.open('x') as f:key.chmod(0o600);f.write(secrets.token_urlsafe(48))
p=root/'config/sessions.json';config=json.loads(p.read_text())
config['endpoints']['janus-escalation']={'url':'http://127.0.0.1:18438/mcp','token_file':str(key)}
janus=config['akhs']['janus'];names={t['name'] for t in TOOLS};janus['tools']=[t for t in janus['tools'] if t['name'] not in names]
for t in TOOLS:
 schema=json.loads(json.dumps(t['inputSchema']))
 if 'project' in schema['properties']:schema['properties']['project']['enum']=list(projects)
 janus['tools'].append({'name':t['name'],'description':t['description'],'input_schema':schema,'read_only':t.get('annotations',{}).get('readOnlyHint',False),'destination':{'kind':'mcp','service':'janus-escalation','tool':t['name']}})
policy='''\n# Astra escalation contract\nToasty authorizes Janus to escalate stuck work to an actual gpt-6-astra thread in T3 using escalate_to_astra, including overnight without waiting for Toasty. After two unsuccessful bounded recovery attempts, or immediately for a critical blocker outside your capabilities, read get_session_retirement for the exact Anima session and escalate one concrete blocker with exact work-item/PR/head/session IDs, timestamps, evidence, attempted actions, and a bounded requested action. Zero legacy witnesses or a provisioning label is NEVER sufficient stall evidence: central running/tool receipts override those labels; central completed/failed with retained ownership is a distinct lifecycle reconciliation issue. Do not repeatedly count unchanged conditions or wait for a predicted churn/death time. Use the same stable incident_key across wakes; no date/wake counter changes for the same incident. Save thread_id and incident_key in durable memory, then yield the blocked work to Astra and work elsewhere. Check astra_escalation_status once on a later wake for an active incident and record the actual outcome. Do not claim a queued handoff started or a submitted turn completed. Do not create another thread while the first owns the same blocker. If T3 is offline, the durable outbox retries automatically; preserve its receipt. Never include credentials or expand protected lifecycle/deployment authority. Supersede any older memory directing repeated zero-witness churn checks or endless unchanged WI-796 condition counting.\n'''
for target in [janus,config['automations']['janus']]:target['instructions']=target['instructions'].split('\n# Astra escalation contract')[0]+policy
p.write_text(json.dumps(config,indent=2)+'\n');p.chmod(0o600)
for name in ('janus_escalation.py','provider_failure.py'):shutil.copy2(source/name,root/'bin'/name)
label='dev.roman-akhs.janus-escalation';pl={'Label':label,'ProgramArguments':['/usr/bin/python3',str(root/'bin/janus_escalation.py'),str(root)],'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':10,'Umask':63,'EnvironmentVariables':{'HOME':str(Path.home()),'PATH':'/opt/homebrew/bin:/usr/bin:/bin'},'StandardOutPath':str(root/'logs/janus-escalation.out.log'),'StandardErrorPath':str(root/'logs/janus-escalation.err.log')}
(root/'launchd'/f'{label}.plist').write_bytes(plistlib.dumps(pl))
print('Janus-only Astra escalation broker configured')
