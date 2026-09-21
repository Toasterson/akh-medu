#!/usr/bin/env python3
"""Register an already-provisioned Vesta sandbox and its authorized operations broker."""
import json
from pathlib import Path
import plistlib
import secrets
import shutil
import sys
from vesta_ops import TOOLS
from vesta_actions import ACTION_TOOLS
root=Path(sys.argv[1]).resolve();source=Path(__file__).resolve().parent
assert (root/'config/vesta-ops.json').exists(), 'Provision and verify the container first'
key=root/'secrets/vesta-ops.key'
if not key.exists():key.write_text(secrets.token_urlsafe(48));key.chmod(0o600)
p=root/'config/sessions.json';session=json.loads(p.read_text())
session['endpoints']['vesta-ops']={'url':'http://127.0.0.1:18437/mcp','token_file':str(key)}
grafana= root/'config/vesta-grafana-tools.json'
if grafana.exists():
 tools=TOOLS+ACTION_TOOLS+[dict(t,name='grafana__'+t['name']) for t in json.loads(grafana.read_text())]
else:tools=TOOLS
vesta=session['akhs']['vesta'];vesta['tools']=[t for t in vesta['tools'] if t['name'] not in {tool['name'] for tool in TOOLS+ACTION_TOOLS} and not t['name'].startswith('grafana__')]
for tool in tools:vesta['tools'].append({'name':tool['name'],'description':tool['description'],'input_schema':tool['inputSchema'],'read_only':tool.get('annotations',{}).get('readOnlyHint',False),'destination':{'kind':'mcp','service':'vesta-ops','tool':tool['name']}})
capabilities='''\nVesta has homelab_inspect for live Kubernetes get/describe/logs and gitops_exec for filesystem and commands inside a dedicated Linux sandbox. The GitOps checkout at /workspace is toasterson/talos. Read AGENTS.md and the relevant runbooks before fixes. Make narrow changes, validate with git diff --check, yamllint and kubectl kustomize where applicable, and preserve other work. No cluster credentials, host home, Docker socket or network are exposed inside this sandbox. You are authorized to take corrective action: use homelab_write for bounded live cluster mutations and gitops_publish to push tested commits to toasterson/talos. Prefer a vesta/* branch for review; respect repository protection and existing review gates. Mirror lasting live changes in GitOps. Grafana MCP tools prefixed grafana__ support queries, dashboards, alerting and operational changes. Preserve intentional silences. Before a mutation read current evidence, check ownership, record the intended action and rollback; afterwards verify the actual result and save receipts to memory. Inspect uncertain outcomes before retrying. Use Sokar/Anima/Board for durable coordination. Do not limit yourself to handoff when a justified repair is within these grants. Local files are not a merged PR or a deployed fix; verify CI, merge, Flux reconciliation and live recovery independently. Never claim host or cluster remediation from a local edit. Janus and Minerva retain their existing filesystem restrictions.\n'''
for target in [vesta,session['automations']['vesta']]:
 target['instructions']=target['instructions'].split('\nVesta has homelab_inspect')[0].replace('No filesystem or shell tools exist.','').replace('Vesta can triage and coordinate through Anima/Board but has no Kubernetes or shell remediation grant.','')+capabilities
p.write_text(json.dumps(session,indent=2)+'\n');p.chmod(0o600)
for name in ('vesta_ops.py','vesta_actions.py'):shutil.copy2(source/name,root/'bin'/name)
plist={'Label':'dev.roman-akhs.vesta-ops','ProgramArguments':['/usr/bin/python3',str(root/'bin/vesta_ops.py'),str(root)],'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':30,'Umask':63,'EnvironmentVariables':{'HOME':str(Path.home()),'PATH':'/opt/homebrew/bin:/usr/bin:/bin','KUBECONFIG':str(Path.home()/'.kube/config')},'StandardOutPath':str(root/'logs/vesta-ops.out.log'),'StandardErrorPath':str(root/'logs/vesta-ops.err.log')}
(root/'launchd/dev.roman-akhs.vesta-ops.plist').write_bytes(plistlib.dumps(plist))
print('Configured Vesta-only GitOps sandbox and homelab diagnostics')
