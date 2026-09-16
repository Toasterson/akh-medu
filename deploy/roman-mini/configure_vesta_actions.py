#!/usr/bin/env python3
"""Provision the authorized Grafana tunnel and discover Vesta action grants."""
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import time
from vesta_actions import grafana_rpc
root=Path(sys.argv[1]).resolve();source=Path(__file__).resolve().parent
label='dev.roman-akhs.grafana-tunnel';p=Path.home()/'Library/LaunchAgents'/f'{label}.plist'
plist={'Label':label,'ProgramArguments':['/opt/homebrew/bin/kubectl','-n','observability','port-forward','--address=127.0.0.1','service/mcp-grafana','18441:80'],'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':30,'Umask':63,'EnvironmentVariables':{'HOME':str(Path.home()),'KUBECONFIG':str(Path.home()/'.kube/config'),'PATH':'/opt/homebrew/bin:/usr/bin:/bin'},'StandardOutPath':str(root/'logs/grafana-tunnel.out.log'),'StandardErrorPath':str(root/'logs/grafana-tunnel.err.log')}
p.write_bytes(plistlib.dumps(plist));(root/'launchd'/p.name).write_bytes(p.read_bytes())
if subprocess.run(['launchctl','print','gui/501/'+label],capture_output=True).returncode:
 subprocess.run(['launchctl','bootstrap','gui/501',str(p)],check=True)
for attempt in range(10):
 try:catalog=grafana_rpc('tools/list',{})['tools'];break
 except Exception:
  if attempt==9:raise
  time.sleep(1)
(root/'config/vesta-grafana-tools.json').write_text(json.dumps(catalog,indent=2))
p=root/'config/vesta-ops.json';config=json.loads(p.read_text())
config.setdefault('cluster_context',subprocess.check_output(['/opt/homebrew/bin/kubectl','config','current-context'],text=True).strip())
p.write_text(json.dumps(config,indent=2))
subprocess.run([sys.executable,str(source/'configure_vesta.py'),str(root)],check=True)
print('Discovered',len(catalog),'Grafana tools; restart broker and reload sessions at a quiet boundary')
