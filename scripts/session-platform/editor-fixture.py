#!/usr/bin/env python3
"""Test-only editor server: observes the executor's files, with its own HOME."""
import http.server
import os
import pathlib
import sys
import uuid

port=int(sys.argv[1])
home=pathlib.Path(os.environ['HOME'])
identity=home/'environment-id'
if not identity.exists():
    identity.write_text(str(uuid.uuid4()))
if '--pair' in sys.argv:
    print(f'http://127.0.0.1:{port}/pair#token=fixture-only')
else:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    http.server.ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
