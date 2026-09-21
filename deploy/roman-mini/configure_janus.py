#!/usr/bin/env python3
"""Apply Toasty's standing authority for gate-respecting Janus merges."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve();p=root/'config/sessions.json';config=json.loads(p.read_text())
instruction='''\n# Janus merge authority\nToasty explicitly authorizes Janus to perform routine approved merges himself through anima__approve_and_merge. You already have this write tool. Read the exact work item, current PR head, preceding working AI approval, current CI and live ownership first. The supported Anima tool enforces head-specific gates and updates lifecycle; use it rather than requesting Toasty or a worker to click merge. Preserve explicit holds, protected human gates, prerequisites and active sessions; never override failed review/CI or manufacture readiness. Confirm the actual merge and resulting lifecycle independently. If a technical boundary genuinely requires a worker, discover the appropriate active worker with board__board_agents and send an addressed coordination/delegation@v1 through board__board_send_inbox containing exact item/PR/head, blockers, requested action and evidence. Record the handoff and do not repeatedly delegate unchanged work. Vesta now has authorized cluster writes, GitOps publishing and Grafana MCP tools for infrastructure recovery.\n'''
for target in (config['akhs']['janus'],config['automations']['janus']):
 target['instructions']=target['instructions'].split('\n# Janus merge authority')[0]+instruction
for automation in config['automations'].values():
 automation['instructions']=automation['instructions'].replace('Vesta can triage and coordinate through Anima/Board but has no Kubernetes or shell remediation grant.','Vesta has authorized homelab cluster writes, GitOps publishing and Grafana MCP tools for bounded infrastructure recovery.')
assert any(t['name']=='anima__approve_and_merge' and not t['read_only'] for t in config['akhs']['janus']['tools'])
p.write_text(json.dumps(config,indent=2)+'\n');p.chmod(0o600)
print('Janus autonomous merge authority configured; lifecycle gates retained')
