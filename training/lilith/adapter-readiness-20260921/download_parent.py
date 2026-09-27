"""Download public pinned parent artifacts into a separate project cache."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path.home() / '.local/share/akh-medu/lilith-training/parent-5926caa4'
ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
metadata = json.loads(Path(__file__).with_name('parent-candidate-api.json').read_text())
receipts = []
for item in metadata['siblings']:
    name = item['rfilename']
    if name not in ['config.json', 'generation_config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'model.safetensors']:
        continue
    target = ROOT / name
    expected = item.get('lfs', {}).get('sha256')
    if not target.exists():
        url = f"https://huggingface.co/{metadata['id']}/resolve/{metadata['sha']}/{name}"
        tmp = target.with_suffix(target.suffix + '.partial')
        digest = hashlib.sha256()
        size = tmp.stat().st_size if tmp.exists() else 0
        if size > item['size']:
            raise ValueError(f'Oversized partial file: {name}')
        if size:
            with tmp.open('rb') as prior:
                while block := prior.read(4 * 1024 * 1024):
                    digest.update(block)
        request = urllib.request.Request(url, headers={'Range': f'bytes={size}-'} if size else {})
        source = urllib.request.urlopen(request, timeout=120)
        if size and source.status != 206:
            # Server ignored Range: restart safely, never append a whole file.
            size, digest = 0, hashlib.sha256()
        elif size and not source.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
            raise ValueError('Unexpected range response')
        with source, tmp.open('ab' if size else 'wb') as out:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
                out.write(block)
                size += len(block)
                if size % (512 * 1024 * 1024) == 0:
                    print(name, size, flush=True)
            out.flush()
            os.fsync(out.fileno())
        if size != item['size'] or (expected and digest.hexdigest() != expected):
            raise ValueError(f'Artifact mismatch: {name}')
        os.replace(tmp, target)
    digest = hashlib.sha256()
    with target.open('rb') as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    assert target.stat().st_size == item['size']
    assert expected is None or digest.hexdigest() == expected
    receipts.append({'name': name, 'bytes': target.stat().st_size, 'sha256': digest.hexdigest(), 'expected_sha256': expected})
    print('verified', name, flush=True)
Path(__file__).with_name('parent-download.json').write_text(json.dumps({'repo':metadata['id'], 'revision':metadata['sha'], 'cache':str(ROOT), 'artifacts':receipts},indent=2)+'\n')
