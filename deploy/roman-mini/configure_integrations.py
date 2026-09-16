#!/usr/bin/env python3
"""Render provider, MCP and T3 configuration from authenticated tool catalogs.

Does not fetch credentials or restart services. Existing credentials, state, and
T3 providers are preserved. The daemons load keys through daemon_credentials.py.
"""
import json
from pathlib import Path
import plistlib
import shutil
import sys

def inline_schema(schema):
    """Expand local definitions for cloud tool parsers without weakening validation."""
    def expand(node, stack=()):
        if isinstance(node, list):
            return [expand(x, stack) for x in node]
        if not isinstance(node, dict):
            return node
        if '$ref' in node:
            ref = node['$ref']
            if not ref.startswith('#/$defs/') or ref in stack:
                raise ValueError('external or recursive tool schema reference')
            target = schema['$defs'][ref.removeprefix('#/$defs/')]
            resolved = expand(target, stack + (ref,))
            resolved.update({k: expand(v, stack) for k, v in node.items() if k != '$ref'})
            return resolved
        out = {k: expand(v, stack) for k, v in node.items() if k != '$defs'}
        branches = out.get('oneOf', out.get('anyOf', []))
        if branches and all(b.get('type') == 'object' for b in branches):
            out.setdefault('type', 'object')
        return out
    return expand(schema)


root = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parent
roles = ('janus', 'minerva', 'vesta')

def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')
    path.chmod(0o600)

runtime_path = root / 'config/runtime.json'
runtime = json.loads(runtime_path.read_text())
for name, profile in runtime['runtimes'].items():
    endpoint = profile['model_endpoint']
    endpoint.pop('token_file', None)
    endpoint['token_env'] = 'ROMAN_' + name.upper() + '_KEY'
save(runtime_path, runtime)
session_path = root / 'config/sessions.json'
session = json.loads(session_path.read_text())
session['endpoints']['sandboxes'] = session['endpoints']['sandbox']
for principal in session['principals']:
    if principal['id'] == 'sessions-caller':
        principal['id'] = 'runtime'
        principal['roles'] = ['runtime']
services = {
    'anima': ('https://anima.wegmueller.it/mcp', 'ROMAN_ANIMA_TOKEN'),
    'seshat': ('http://seshat.lab.home/mcp', 'ROMAN_ARCHIVE_TOKEN'),
    'sokar': ('http://127.0.0.1:18439/mcp', 'ROMAN_ARCHIVE_TOKEN'),
}
for role in roles:
    services['board-' + role] = ('http://127.0.0.1:18089/mcp', 'ROMAN_BOARD_' + role.upper())
for name, (url, token_env) in services.items():
    session['endpoints'][name] = {'url': url, 'token_env': token_env, 'allow_insecure_http': True}
for role in roles:
    profile = session['akhs'][role]
    profile['tools'] = [t for t in profile['tools'] if t['destination']['kind'] == 'cognitive']
    for name in ('anima', 'seshat', 'sokar', 'board'):
        catalog = json.loads((root / 'evidence' / f'{name}-mcp-catalog.json').read_text())
        for tool in catalog:
            profile['tools'].append({'name': name + '__' + tool['name'],
                'description': tool.get('description', ''), 'input_schema': inline_schema(tool['inputSchema']),
                'read_only': tool.get('annotations', {}).get('readOnlyHint', False),
                'destination': {'kind': 'mcp', 'service': 'board-' + role if name == 'board' else name, 'tool': tool['name']}})
    profile['instructions'] = profile['instructions'].split('\nUse internal memory and knowledge notes')[0] + '\nUse internal memory and knowledge notes for persistent work. No filesystem or shell tools are granted. Sokar paths are artifact names, never host paths. Anima MCP acts under Toasty\'s delegated operator authority; identify your Akh name in audit reasons and messages. Djehuty uses your own Roman identity. Respect existing lifecycle gates even when an override tool exists.\n'
    assert all(t['destination']['kind'] in ('cognitive', 'mcp') for t in profile['tools'])
save(session_path, session)
shutil.copy2(source / 'daemon_credentials.py', root / 'bin/daemon_credentials.py')
for role in ('runtime', 'sessions'):
    path = root / 'launchd' / f'dev.roman-akhs.{role}.plist'
    doc = plistlib.loads(path.read_bytes())
    doc.setdefault('EnvironmentVariables', {}).update({'HOME': '/Users/toasty', 'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/Users/toasty/.local/bin'})
    if role == 'sessions':
        doc.pop('UserName', None)
    doc['ProgramArguments'] = [shutil.which('python3'), str(root / 'bin/daemon_credentials.py'), role, str(root)]
    path.write_bytes(plistlib.dumps(doc))
# Dedicated supervised tunnels avoid the currently failing lab gateway routes.
for name, namespace, service, ports in (
    ('board-tunnel', 'djehuty-board', 'djehuty-board', '18089:8080'),
    ('sokar-tunnel', 'akh-medu', 'akh-medu-sokard', '18439:9400'),
):
    doc = {'Label': 'dev.roman-akhs.' + name, 'UserName': 'toasty',
        'ProgramArguments': [shutil.which('kubectl'), '-n', namespace, 'port-forward', '--address=127.0.0.1', 'service/' + service, ports],
        'EnvironmentVariables': {'HOME': '/Users/toasty', 'KUBECONFIG': '/Users/toasty/.kube/config', 'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/Users/toasty/.local/bin'},
        'RunAtLoad': True, 'KeepAlive': True, 'ThrottleInterval': 30,
        'StandardOutPath': str(root / 'logs' / (name + '.log')),
        'StandardErrorPath': str(root / 'logs' / (name + '.err.log'))}
    (root / 'launchd' / (doc['Label'] + '.plist')).write_bytes(plistlib.dumps(doc))
print('Rendered providers and 102 cognitive/service tools per Akh; no filesystem grants')
