"""Scoped certificate-authenticated OpenBao KV v2 slot storage.

Values stay in memory. Updates use the version observed by this operation and
fail on concurrent changes; there is no fallback to a different authority.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import ssl
import stat
import urllib.error
import urllib.parse
import urllib.request

from slots import SlotError


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise SlotError('OpenBao redirect refused', help='check the configured private endpoint')


def _private(path):
    path = Path(path).expanduser()
    s = path.lstat()
    if not path.is_absolute() or not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o077:
        raise SlotError('OpenBao credential file must be owner-only', help='use absolute paths to regular mode 0600 files')
    return path


class OpenBao:
    def __init__(self, config_file, allowed_items):
        config = json.loads(_private(config_file).read_text())
        parsed = urllib.parse.urlsplit(config['url'])
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise SlotError('OpenBao endpoint must use HTTPS', help='configure the verified private endpoint')
        self.base = config['url'].rstrip('/') + '/v1/'
        self.paths = config['items']
        if set(self.paths) != set(allowed_items) or any(not isinstance(p, str) or not p.startswith('secret/data/') or '..' in p or '?' in p or '#' in p for p in self.paths.values()):
            raise SlotError('OpenBao item mapping does not match the slot configuration', help='enroll exactly the hub, state and configured slot items')
        context = ssl.create_default_context(cafile=str(_private(config['ca_file'])))
        context.load_cert_chain(str(_private(config['cert_file'])), str(_private(config['key_file'])))
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context), NoRedirect())
        self.token = None
        self.observed = {}
        auth = self._request('auth/cert/login', {'name': config['role']})['auth']
        if not 0 < auth.get('lease_duration', 0) <= 3600 or 'root' in auth.get('policies', []):
            raise SlotError('OpenBao returned an unsafe token scope or lifetime', help='enroll a dedicated non-root role with at most one-hour tokens')
        self.token = auth['client_token']

    def _request(self, path, data=None):
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['X-Vault-Token'] = self.token
        req = urllib.request.Request(self.base + path, headers=headers, data=None if data is None else json.dumps(data).encode())
        try:
            with self.opener.open(req, timeout=30) as response:
                body = response.read()
                return json.loads(body) if body else None
        except (urllib.error.URLError, ValueError, OSError) as exc:
            # Never expose an upstream body, token or key in exception output.
            code = getattr(exc, 'code', None)
            raise SlotError('OpenBao request failed' + (f' (HTTP {code})' if code else ''),
                            help='check connectivity and scoped authorization; a concurrent update requires re-running the operation') from None

    def fields(self, item):
        if item not in self.paths:
            raise SlotError('unassigned OpenBao item', help='use an enrolled slot item')
        doc = self._request(self.paths[item])['data']
        values, version = doc['data'], doc['metadata']['version']
        if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()) or not isinstance(version, int) or version < 1:
            raise SlotError('invalid OpenBao slot document', help='repair the scoped KV entry')
        if item in self.observed and self.observed[item][1] != version:
            raise SlotError('OpenBao slot changed during this operation', help='re-run the complete operation against fresh state')
        self.observed[item] = (dict(values), version)
        return dict(values)

    def read(self, item, field):
        fields = self.fields(item)
        if field not in fields:
            raise SlotError('OpenBao slot field missing', help='repair the scoped KV entry')
        return fields[field]

    def set_fields(self, item, values):
        if item not in self.observed:
            self.fields(item)
        original, version = self.observed[item]
        if not set(values).issubset(original):
            raise SlotError('OpenBao update references missing fields', help='repair the scoped KV entry')
        updated = {**original, **values}
        result = self._request(self.paths[item], {'options': {'cas': version}, 'data': updated})
        self.observed[item] = (updated, result['data']['version'])
