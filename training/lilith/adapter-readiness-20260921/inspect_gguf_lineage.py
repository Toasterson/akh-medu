"""Read GGUF metadata only; do not load tensors or initialize an accelerator."""
import hashlib
import json
import struct
from pathlib import Path

model = Path.home() / '.local/share/akh-medu/models/llm/gemma4-12b.gguf'
FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
with model.open('rb') as f:
    def read(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
    def string():
        size = read('Q')
        if size > 100_000_000:
            raise ValueError('oversized metadata string')
        return f.read(size).decode('utf8')
    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            subtype, count = read('I'), read('Q')
            if count > 2_000_000:
                raise ValueError('oversized metadata array')
            vals = [value(subtype) for _ in range(count)]
            return {'array_length': count, 'element_type': subtype} if count > 256 else vals
        return read(FORMATS[kind])
    assert f.read(4) == b'GGUF'
    version, tensors, entries = read('I'), read('Q'), read('Q')
    assert version in (2, 3)
    metadata = {}
    for _ in range(entries):
        key, kind = string(), read('I')
        val = value(kind)
        if key == 'tokenizer.chat_template':
            val = {'utf8_sha256': hashlib.sha256(val.encode()).hexdigest(), 'characters': len(val)}
        metadata[key] = val
    report = {'schema': 'lilith/gguf-lineage-metadata@1', 'gguf_version': version,
              'tensor_count': tensors, 'metadata_count': entries, 'metadata_end_offset': f.tell(),
              'model': str(model), 'metadata': metadata}
    Path(__file__).with_name('gguf-metadata.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in metadata.items() if k.startswith('general.')}, indent=2))
