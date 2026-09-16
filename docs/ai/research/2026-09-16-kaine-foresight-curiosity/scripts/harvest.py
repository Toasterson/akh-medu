"""Read-only source harvesting; preserves conversational text without rewriting it."""
import hashlib
import json
import pathlib
import shutil
import sqlite3

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
THREAD = '7e5cfbbd-084f-4cf8-96ab-16fcf9f39733'
CODEX = pathlib.Path('/Users/toasty/.codex/sessions/2026/09/16/rollout-2026-09-16T20-34-10-01a0ab7f-6f83-7ea2-8cba-8c938422c48c.jsonl')
CLAUDE = pathlib.Path('/Users/toasty/.claude/projects/-Users-toasty-ws-akh-medu/bfb0ce41-a104-407c-a0cc-6f5c801cc003.jsonl')

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def transcript(name, messages, note):
    write_json(ROOT / 'transcripts' / (name + '.json'), messages)
    rendered = '# ' + name + '\n\n' + note + '\n\n'
    for i, m in enumerate(messages, 1):
        rendered += f"## {i}. {m['role']}\n\n{m['text']}\n\n"
    (ROOT / 'transcripts' / (name + '.md')).write_text(rendered)
    recovered = json.loads((ROOT / 'transcripts' / (name + '.json')).read_text())
    assert recovered == messages
    assert all(m['text'] in rendered for m in messages)

c = sqlite3.connect('file:/Users/toasty/.t3/userdata/state.sqlite?mode=ro', uri=True)
c.row_factory = sqlite3.Row
thread = dict(c.execute('select * from projection_threads where thread_id=?', (THREAD,)).fetchone())
write_json(ROOT / 'context/t3-thread.json', thread)
messages = [dict(r) for r in c.execute('select message_id,role,text,created_at,attachments_json from projection_thread_messages where thread_id=? order by created_at', (THREAD,))]
transcript('claude-t3-conversation', messages, 'Exact stored message text from t3code. Historical claims are preserved, not endorsed. Thread remains snoozed.')
c.close()

codex = []
started = False
for line in CODEX.open():
    o = json.loads(line)
    p = o.get('payload', {})
    if o.get('type') != 'response_item' or p.get('type') != 'message' or p.get('role') not in ('user', 'assistant'):
        continue
    text = '\n'.join(b['text'] for b in p.get('content', []) if b.get('type') in ('input_text', 'output_text'))
    if text.startswith('Oh, another paper for Kaine Tweet:'):
        started = True
    if started and text:
        codex.append({'role': p['role'], 'text': text, 'timestamp': o.get('timestamp')})
transcript('codex-conversation', codex, 'Exact user-facing messages through the harvest snapshot; excludes system/developer instructions, private reasoning, and raw tool traffic. Original citations and caveats are retained.')

evidence = []
for line in CLAUDE.open():
    o = json.loads(line)
    m = o.get('message', {})
    for b in m.get('content', []) if isinstance(m.get('content'), list) else []:
        if b.get('type') == 'tool_use' and b.get('name', '').lower().find('agent') < 0:
            evidence.append({'kind': 'tool_use', 'name': b.get('name'), 'input': b.get('input'), 'id': b.get('id')})
        elif b.get('type') == 'tool_result':
            content = b.get('content')
            # Keep source-retrieval evidence, not internal agent-launch metadata.
            if 'Async agent launched' not in str(content):
                evidence.append({'kind': 'tool_result', 'tool_use_id': b.get('tool_use_id'), 'content': content})
write_json(ROOT / 'context/claude-retrieval-evidence.json', evidence)

for rel in ['docs/ai/decisions/110-kaine-character-guided-reward-live-environment.md', 'docs/ai/decisions/111-kaine-architecture-ladder.md', 'docs/ai/plans/2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md']:
    shutil.copyfile(REPO / rel, ROOT / 'context' / pathlib.Path(rel).name)
write_json(ROOT / 'context/transcript-provenance.json', {
    't3_thread_id': THREAD, 'claude_session_path': str(CLAUDE), 'codex_session_path': str(CODEX),
    'claude_messages': len(messages), 'codex_messages': len(codex),
    'fidelity': 'Message strings are unchanged; JSON escaping and Markdown role wrappers are archival formatting.',
    'scope': 'Research conversation and retrieval evidence, no private reasoning or credentials.'})
print(json.dumps({'claude_messages': len(messages), 'codex_messages': len(codex), 'root': str(ROOT)}))
