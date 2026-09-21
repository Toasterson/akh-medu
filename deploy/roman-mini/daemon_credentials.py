#!/usr/bin/env python3
"""Resolve provider/service credentials into the daemon environment, never disk.

Launchd passes ROLE and deployment ROOT. Only these operator-owned references are
resolved; no agent-supplied command, reference, or environment is accepted.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def capture(command):
    result = subprocess.run(command, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError('credential source unavailable: ' + Path(command[0]).name)
    return result.stdout.decode().strip()


def main():
    role, root_arg = sys.argv[1:]
    root = Path(root_arg)
    env = dict(os.environ)
    op = '/Users/toasty/.local/bin/op-service-account'
    if role == 'runtime':
        for var, reference in {
            'ROMAN_OLLAMA_KEY': 'op://Homelab/Ollama API Key Akhrtd Command Group/api key',
            'ROMAN_OPENROUTER_KEY': 'op://Homelab/OpenRouter API Key - Akhrtd Command Group/credential',
        }.items():
            env[var] = capture([op, 'read', reference])
        binary = 'akhrtd'
    elif role == 'sessions':
        env['ROMAN_ARCHIVE_TOKEN'] = capture([op, 'read', 'op://Homelab/sia-runtime/SIA_SESHAT__TOKEN'])
        env['ROMAN_ANIMA_TOKEN'] = json.loads(capture(['/Users/toasty/.local/bin/anima-pat', 'get-auth-header']))['Authorization'].removeprefix('Bearer ')
        for akh in ('janus', 'minerva', 'vesta'):
            env['ROMAN_BOARD_' + akh.upper()] = json.loads((root / 'secrets' / f'board-{akh}.json').read_text())['bearer_token']
        binary = 'akhsessd'
    elif role == 'events':
        for akh in ('janus', 'minerva', 'vesta'):
            env['ROMAN_BOARD_' + akh.upper()] = json.loads((root / 'secrets' / f'board-{akh}.json').read_text())['bearer_token']
        binary = 'akheventd'
    else:
        raise RuntimeError('unsupported daemon')
    os.execve(root / 'bin' / binary, [binary, '--config', str(root / 'config' / (role + '.json'))], env)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        detail = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print('Roman daemon credential initialization failed: ' + detail, file=sys.stderr)
        sys.exit(1)
