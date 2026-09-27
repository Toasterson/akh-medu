"""Verified local lesson admission and retrieval; frozen weights, no automatic SFT."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import uuid
from lilith_contracts import CONTRACTS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


# Operator-curated principles. Teacher prose is retained as opinion, never used
# as an independently verified target. This first slice admits only these domains.
PRINCIPLES = {
    'integer-clamp/1': dict(family='saturation',
        text='For an integer saturation contract, check both bounds and preserve in-range values. Test values immediately below, at, and above each bound. Do not confuse saturation with rejecting invalid input.',
        features=[['domain','integer'],['operation','saturate'],['boundary','inclusive'],['policy','preserve_in_range'],['evidence','host_python']]),
    'u64-range/1': dict(family='bounded_range',
        text='For bounded unsigned arithmetic, reject out-of-domain inputs first. Check the starting quantity is within the capacity before comparing the requested amount with capacity minus starting quantity. An empty request at the boundary needs an explicit contract. Python reference tests do not establish C overflow or native platform correctness.',
        features=[['domain','integer'],['operation','capacity_check'],['boundary','empty_allowed'],['policy','reject_invalid'],['evidence','host_python']]),
}


class Lessons:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path)
        path.chmod(0o600)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS lessons (id TEXT PRIMARY KEY, body TEXT NOT NULL, sha256 TEXT NOT NULL, active INTEGER NOT NULL, reason TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', ('store_id', uuid.uuid4().hex))
        self.store_id = self.db.execute("SELECT value FROM metadata WHERE name='store_id'").fetchone()[0]
        self.db.commit()

    def close(self):
        self.db.close()

    def admit(self, assignment):
        if assignment.get('status') != 'completed' or assignment.get('cancellation'):
            raise ValueError('only completed, non-cancelled assignments can teach')
        for key in ('id','parent','parent_turn','child','review_turn','reviewer','review','report'):
            if not assignment.get(key):
                raise ValueError('missing teaching provenance: '+key)
        path = assignment['intent']['path']
        matches = [c for c in CONTRACTS.values() if c.artifact == path]
        if len(matches) != 1:
            raise ValueError('unsupported lesson contract')
        contract = matches[0]
        prior_grant = assignment.get('contract')
        if prior_grant is not None and prior_grant != contract.grant():
            raise ValueError('historical contract differs; do not silently reinterpret it')
        source = assignment['source']
        expected = digest(source)
        original = assignment['evidence']
        if (assignment['source_sha256'] != expected or original.get('source_sha256') != expected
                or original.get('passed') is not True):
            raise ValueError('artifact and verification evidence disagree')
        replay = contract.verifier(source)
        if replay['cases'] != original.get('cases') or replay['source_sha256'] != expected:
            raise ValueError('verification replay differs from historical evidence')
        principle = PRINCIPLES[contract.id]
        lesson_id = digest(canonical([contract.id, expected]))
        envelope = dict(schema='lilith/lesson@1', id=lesson_id, contract=contract.grant(),
            family=principle['family'], text=principle['text'], features=principle['features'],
            source=source, source_sha256=expected, replay=replay,
            preconditions=['integer inputs', 'local Python reference semantics'],
            limitations=['not C/ABI/kernel validation', 'not native illumos validation', 'not a quality parity result'],
            provenance=dict(assignment=assignment['id'], parent=assignment['parent'],
                author_turn=assignment['parent_turn'], teacher=assignment['reviewer'],
                teacher_session=assignment['child'], teacher_turn=assignment['review_turn'],
                teacher_opinion=assignment['review'], original_evidence=original,
                historical_contract_pinned=prior_grant is not None,
                assignment_sha256=digest(canonical(assignment)),
                principle_origin='operator-curated; source replayed; teacher opinion retained separately'))
        body = canonical(envelope)
        with self.db:
            # Repeated source/contract observations do not multiply evidence or
            # reactivate a revoked lesson. Original attribution is immutable.
            self.db.execute('INSERT OR IGNORE INTO lessons VALUES (?, ?, ?, 1, NULL)',
                            (lesson_id, body, digest(body)))
        return lesson_id

    def state(self, lesson_id):
        row = self.db.execute('SELECT active, reason FROM lessons WHERE id=?', (lesson_id,)).fetchone()
        if row is None:
            raise ValueError('unknown lesson')
        return dict(active=bool(row[0]), reason=row[1])

    def revoke(self, lesson_id, reason):
        if not reason.strip() or len(reason) > 2000:
            raise ValueError('revocation requires a bounded reason')
        with self.db:
            row = self.db.execute('SELECT active, reason FROM lessons WHERE id=?', (lesson_id,)).fetchone()
            if row is None:
                raise ValueError('unknown lesson')
            if not row[0] and row[1] != reason:
                raise ValueError('revocation reason is immutable')
            self.db.execute('UPDATE lessons SET active=0, reason=? WHERE id=?', (reason, lesson_id))

    def active(self, excluded_families=()):
        items = []
        for body, checksum in self.db.execute('SELECT body, sha256 FROM lessons WHERE active=1 ORDER BY id'):
            if digest(body) != checksum:
                raise ValueError('lesson envelope checksum mismatch')
            item = json.loads(body)
            if item['family'] in excluded_families:
                continue
            contract = CONTRACTS[item['contract']['id']]
            if item['contract'] != contract.grant():
                raise ValueError('lesson verifier changed; explicit re-admission is required')
            replay = contract.verifier(item['source'])
            if replay != item['replay'] or digest(item['source']) != item['source_sha256']:
                raise ValueError('lesson replay no longer matches')
            items.append(item)
        return items

    def retrieve(self, mode, query, features, binary, excluded_families=(), max_bytes=1400):
        if mode == 'none':
            return dict(mode=mode, hits=[], context='')
        if mode not in ('text', 'vsa'):
            raise ValueError('unknown retrieval mode')
        items = self.active(excluded_families)
        by_id = {item['id']:item for item in items}
        space = None
        if mode == 'text':
            tokens = set(re.findall(r'[a-z0-9_]+', (query+' '+canonical(features)).lower()))
            def score(item):
                words = set(re.findall(r'[a-z0-9_]+', (item['text']+' '+canonical(item['features'])).lower()))
                return len(tokens & words) / max(1, len(tokens | words))
            hits = sorted([(item['id'],score(item)) for item in items], key=lambda x:(-x[1],x[0]))
        else:
            request = dict(query=features, items=[dict(id=i['id'],features=i['features']) for i in items])
            result = subprocess.run([str(binary)], input=canonical(request), text=True,
                                    capture_output=True, check=True, timeout=20)
            result = json.loads(result.stdout)
            space, hits = result['space'], result['hits']
        selected = []
        context = ''
        if hits:
            item = by_id[hits[0][0]]
            required = {tuple(f) for f in features if f[0] in ('domain', 'operation')}
            if not required.issubset({tuple(f) for f in item['features']}):
                return dict(mode=mode, hits=[], context='', space=space, reason='top candidate is outside symbolic applicability')
            context = ('Verified prior lesson (reference scope only): '+item['text']+
                       '\nLesson '+item['id']+'; evidence source SHA-256 '+item['source_sha256'])
            if len(context.encode()) > max_bytes:
                context = ''  # Never silently truncate a principle or its provenance.
            else:
                selected = [dict(id=item['id'], score=hits[0][1], family=item['family'])]
        return dict(mode=mode, hits=selected, context=context, space=space)

    def export(self, path, excluded_families):
        # Candidate development SFT only. Full replay precedes opening the output.
        items = self.active(excluded_families)
        with Path(path).open('x') as stream:
            for item in items:
                stream.write(canonical(dict(split='development_candidate', family=item['family'],
                    lesson_id=item['id'], input=item['contract']['specification'],
                    output=item['source'], source_sha256=item['source_sha256']))+'\n')
        return len(items)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True, type=Path)
    sub = p.add_subparsers(dest='command', required=True)
    a=sub.add_parser('admit');a.add_argument('assignments',type=Path)
    r=sub.add_parser('revoke');r.add_argument('id');r.add_argument('--reason',required=True)
    e=sub.add_parser('export');e.add_argument('output',type=Path);e.add_argument('--exclude-family',action='append',default=[])
    q=sub.add_parser('retrieve');q.add_argument('--mode',choices=['none','text','vsa'],required=True);q.add_argument('--query',required=True)
    q.add_argument('--features',type=json.loads,default=[]);q.add_argument('--vsa-binary',type=Path,default=Path('target/release/examples/lilith_lesson_vsa'))
    q.add_argument('--exclude-family',action='append',default=[])
    args=p.parse_args(); store=Lessons(args.db)
    try:
        if args.command=='admit':
            print(json.dumps([store.admit(row) for row in json.loads(args.assignments.read_text())]))
        elif args.command=='revoke':
            store.revoke(args.id,args.reason);print(json.dumps({'revoked':args.id}))
        elif args.command=='export':print(json.dumps({'candidate_examples':store.export(args.output,args.exclude_family),'weights_updated':False}))
        else:print(json.dumps(store.retrieve(args.mode,args.query,args.features,args.vsa_binary,args.exclude_family)))
    finally:store.close()


if __name__=='__main__':main()
