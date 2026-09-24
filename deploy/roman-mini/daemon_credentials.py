#!/usr/bin/env python3
"""Resolve provider/service credentials into the daemon environment, never disk.

Launchd passes ROLE and deployment ROOT. Only these operator-owned references are
resolved; no agent-supplied command, reference, or environment is accepted.
"""
import json
import os
import re
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
    if role in ('runtime', 'runtime-minerva', 'runtime-vesta'):
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
    elif role == 'core':
        # ADR 113 decision 2: the akh's own turn runs inside akhomed, so the core
        # process needs the same Ollama Cloud key the runtime roles use, plus the
        # MCP tokens the akh's `[[mcp.servers]]` (Phase 53i) will authenticate
        # with, following the same 1Password references and `secrets/` files the
        # sessions and events roles already read. ROMAN_ARCHIVE_TOKEN also
        # authenticates the Sokar MCP endpoint today (see the `services` map in
        # configure_integrations.py); there is no separate Sokar credential yet.
        env['ROMAN_OLLAMA_KEY'] = capture([op, 'read', 'op://Homelab/Ollama API Key Akhrtd Command Group/api key'])
        env['ROMAN_ARCHIVE_TOKEN'] = capture([op, 'read', 'op://Homelab/sia-runtime/SIA_SESHAT__TOKEN'])
        env['ROMAN_ANIMA_TOKEN'] = json.loads(capture(['/Users/toasty/.local/bin/anima-pat', 'get-auth-header']))['Authorization'].removeprefix('Bearer ')
        for akh in ('janus', 'minerva', 'vesta'):
            env['ROMAN_BOARD_' + akh.upper()] = json.loads((root / 'secrets' / f'board-{akh}.json').read_text())['bearer_token']
        # ADR 113 decision 4 / 53d: Board, Anima and alert events are admitted
        # into each akh's own reactor. `reactor_manage enable` refuses without
        # this gate. It is process-wide, so while it is set an akh's own
        # reactor_manage call can enable a schedule it staged; CUTOVER.md lists
        # this as an operator decision.
        env['AKH_REACTOR_ENABLE'] = '1'
        # akhomed reads AKH_CORE_CONFIG and friends from the environment (see
        # configure_core.py), not from a `--config` argument. The binary is the
        # provenance-named `bin/akhomed-main-<sha>` recorded by configure_core.py
        # (deploy-from-main rule); refuse anything else.
        release = json.loads((root / 'config/core-release.json').read_text())
        executable = Path(release['binary'])
        if executable.parent.resolve() != (root / 'bin').resolve() or not re.fullmatch(r'akhomed-main-[0-9a-f]{7,40}', executable.name):
            raise RuntimeError('core release binary is not bin/akhomed-main-<sha>; rerun configure_core.py')
        os.execve(executable, [str(executable)], env)
        return
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
