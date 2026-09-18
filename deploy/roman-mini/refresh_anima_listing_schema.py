#!/usr/bin/env python3
"""Refresh the existing WI-listing grant after its server deployment.

This never grants another tool or changes running sessions. Run at a drained
session-service boundary, then restart that service to load the new profile.
The input is an authenticated MCP tools/list catalog, not agent-supplied JSON.
"""
import argparse
import copy
import json
from pathlib import Path


def refreshed_config(config, catalog):
    tool = next(t for t in catalog if t['name'] == 'list_work_items')
    if tool['inputSchema']['properties'].get('page_token', {}).get('type') != 'string':
        raise ValueError('deployed list_work_items does not advertise page_token')
    result = copy.deepcopy(config)
    changed = []
    for role, profile in result['akhs'].items():
        for grant in profile['tools']:
            destination = grant.get('destination', {})
            if destination == {'kind': 'mcp', 'service': 'anima', 'tool': 'list_work_items'}:
                grant['input_schema'] = copy.deepcopy(tool['inputSchema'])
                grant['description'] = tool['description']
                changed.append(role)
    if not {'janus', 'minerva'}.issubset(changed):
        raise ValueError('expected existing Janus and Minerva listing grants')
    return result, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('catalog', type=Path)
    args = parser.parse_args()
    path = args.root / 'config/sessions.json'
    original = path.read_text()
    result, changed = refreshed_config(json.loads(original), json.loads(args.catalog.read_text()))
    backup = path.with_name('sessions.before-wi-pagination.json')
    if not backup.exists():
        with backup.open('x') as handle:
            backup.chmod(0o600)
            handle.write(original)
    temporary = path.with_name('sessions.pagination.tmp')
    with temporary.open('w') as handle:
        temporary.chmod(0o600)
        handle.write(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
    print('Updated existing listing schema for: ' + ', '.join(changed))


if __name__ == '__main__':
    main()
