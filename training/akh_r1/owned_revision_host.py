"""Private single-host journal for isolated native revision; no inference here."""
import copy
import fcntl
import json
import os
from pathlib import Path
from sourced_native_rule_contract import Contract


def key(labels):
    return tuple(labels[x] for x in ('subject', 'predicate', 'object'))


class Host:
    def __init__(self, graph, directory, graph_identity, contract_factory=Contract):
        self.graph = graph
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.directory / 'journal.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(
            graph_identity=graph_identity, pending=False, current=None, used=[], observations={}, operations=[])
        if self.state['graph_identity'] != graph_identity:
            self.close()
            raise ValueError('Graph identity mismatch')
        self.contract = contract_factory()
        for receipt, document in self.state['observations'].items():
            self.contract.observe(receipt, document)

    def close(self):
        self.lock.close()

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump(self.state, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def observe(self, receipt, document):
        # Only trusted host observation ingress may call this method.
        self.contract.observe(receipt, document)
        self.state['observations'][receipt] = copy.deepcopy(document)
        self.save()

    def call(self, tool, args):
        record = dict(tool=tool, arguments=copy.deepcopy(args), response=None)
        self.state['operations'].append(record)
        self.save()
        response = self.graph.call(tool, args)
        record['response'] = copy.deepcopy(response)
        self.save()
        if not isinstance(response, dict) or response.get('success') is not True:
            raise RuntimeError('Native operation failed; preserve pending journal')
        return response

    def snapshot(self):
        rows = self.call('exam_owned_snapshot', {})['rows']
        mapped = {key(row['labels']): row['edge'] for row in rows}
        if len(mapped) != len(rows):
            raise ValueError('Ambiguous label snapshot')
        return mapped

    def apply(self, receipt, source_hash, previous):
        if self.state['pending']:
            raise RuntimeError('Interrupted update requires recovery; no retry')
        plan = self.contract.request(receipt, source_hash)
        current = self.state['current']
        expected = None if current is None else dict(receipt=current['receipt'], sha256=current['sha256'])
        if previous != expected or receipt in self.state['used']:
            raise ValueError('Current receipt/hash mismatch or replay')
        before = self.snapshot()
        old = [] if current is None else current['sources'] + current['derived']
        for row in old:
            if before.get(key(row['labels'])) != row['edge']:
                raise ValueError('Persisted ownership does not match live graph')
        owned_keys = {key(row['labels']) for row in old}
        for fact in plan['observed_facts']:
            if key(fact) in before and key(fact) not in owned_keys:
                raise ValueError('Cannot adopt pre-existing source edge')
        self.state['pending'] = True
        self.state['used'].append(receipt)
        self.save()
        if current:
            self.call('exam_retract_owned', dict(sources=[r['edge'] for r in current['sources']],
                                                derived=[r['edge'] for r in current['derived']]))
            remaining = self.snapshot()
            if any(key(r['labels']) in remaining for r in old):
                raise RuntimeError('Invalidated edge remains live')
        self.call('memory_write', dict(summary='Owned observed source '+source_hash,
                                      source=receipt, facts=plan['observed_facts']))
        written = self.snapshot()
        sources = [dict(labels=f, edge=written[key(f)]) for f in plan['observed_facts']]
        inference = self.call('exam_run_rules', dict(rules=plan['native_rules']))
        after = self.snapshot()
        # Claim only newly committed admitted conclusions, never existing facts.
        # Capture native-reported committed derivations, including intermediate
        # variable-subject joins. Never claim old asserted results as owned.
        reported = {key(d['triple']) for d in inference.get('derived', [])}
        derived = [dict(labels=dict(zip(('subject','predicate','object'), k)), edge=v)
                   for k,v in after.items() if k not in written and k in reported]
        if any(not row['edge'].get('provenance_id') for row in derived):
            raise RuntimeError('Derived edge lacks committed provenance')
        self.state['current'] = dict(receipt=receipt, sha256=source_hash, sources=sources, derived=derived)
        self.state['pending'] = False
        self.save()
        return copy.deepcopy(self.state['current'])
