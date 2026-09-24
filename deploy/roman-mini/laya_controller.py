#!/usr/bin/env python3
"""Durable event-to-scoped-fixer controller; all Anima access is through MCP."""
import fcntl
from datetime import datetime, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

from laya_service import CHOICES, digest, encoded

ROLE = {'product': 'minerva', 'execution': 'janus', 'infrastructure': 'vesta',
        'human': 'janus', 'wait': 'janus', 'investigate': 'janus'}
SCOPE = {
    'minerva': 'Investigate and repair requirements, acceptance criteria or decomposition within existing product authority. Preserve accepted scope and explicit human decisions.',
    'janus': 'Investigate and repair the execution or workflow blocker using supported Anima tools. Preserve exact-head review, CI, dependencies and live ownership.',
    'vesta': 'Investigate and repair the evidenced infrastructure or provider fault within existing operator authority. Preserve workload ownership and verify service and tool execution separately.',
}


def enabled(root):
    path = Path(root)/'config/laya.json'
    return path.exists() and json.loads(path.read_text()).get('event_only') is True


class Queue:
    def __init__(self, root):
        self.path = Path(root)/'state/laya-decisions.sqlite3'
        self.db = sqlite3.connect(self.path, timeout=10)
        self.path.chmod(0o600)
        self.db.execute('pragma journal_mode=WAL')
        self.db.execute('pragma synchronous=FULL')
        self.db.executescript('''
          create table if not exists queue(entity text primary key, change_json text not null, revision text not null, queued_at real not null);
          create table if not exists decisions(id text primary key, entity text not null, snapshot text not null, proposal text not null, action text not null, role text, state text not null, event_id text, created_at real not null);
          create table if not exists observations(entity text primary key, revision text not null, observed_at real not null);
          create table if not exists intervention_budget(episode text primary key, attempts integer not null);
          create table if not exists metadata(key text primary key, value text not null);
          create table if not exists fixer_results(decision_id text primary key, role text not null, result text not null, created_at real not null);
        ''')

    def observe(self, change, force=False):
        entity = change['kind']+':'+change['id']
        revision = digest(change['current'])
        old = self.db.execute('select revision,observed_at from observations where entity=?', (entity,)).fetchone()
        # Listing rows omit park reasons and thread revisions. Rehydrate old
        # observations as deterministic catch-up; only changed dossiers infer.
        stale_park = change.get('current', {}).get('current_phase') == 'parked' and old and time.time()-old[1] > 300
        if force or not old or old[0] != revision or stale_park:
            with self.db:
                self.db.execute('insert into queue values (?,?,?,?) on conflict(entity) do update set change_json=excluded.change_json, revision=excluded.revision',
                                (entity, encoded(change), revision, time.time()))
                self.db.execute('insert or replace into observations values (?,?,?)', (entity, revision, time.time()))

    def close(self): self.db.close()

    def record_result(self, role, args):
        required = {'decision_id', 'outcome', 'evidence_refs', 'detail', 'next_event'}
        if set(args) != required or args['outcome'] not in ('fix_applied', 'human_required', 'waiting', 'blocked'):
            raise ValueError('invalid typed fixer result')
        if args['next_event'] not in ('human_reply', 'dependency_changed', 'lane_recovered', 'ci_completed', 'review_completed', 'work_item_changed', 'none'):
            raise ValueError('unknown next event')
        if not isinstance(args['detail'], str) or not 1 <= len(args['detail']) <= 4000:
            raise ValueError('detail must describe evidence and the exact remaining question or blocker')
        refs = args['evidence_refs']
        if not isinstance(refs, list) or not 1 <= len(refs) <= 16 or any(not isinstance(x, str) or not 1 <= len(x) <= 512 for x in refs):
            raise ValueError('provide bounded source references')
        if args['outcome'] == 'human_required' and args['next_event'] != 'human_reply':
            raise ValueError('human question requires human_reply')
        if args['outcome'] == 'waiting' and args['next_event'] == 'none':
            raise ValueError('waiting requires a named next event')
        row = self.db.execute('select role,state from decisions where id=?', (args['decision_id'],)).fetchone()
        if not row or row[0] != role: raise PermissionError('result does not belong to this role')
        value = encoded(args)
        old = self.db.execute('select result from fixer_results where decision_id=?', (args['decision_id'],)).fetchone()
        if old and old[0] != value: raise ValueError('a different result is already recorded')
        if not old and row[1] not in ('queued', 'running'): raise ValueError('intervention is not active')
        with self.db:
            self.db.execute('insert or ignore into fixer_results values (?,?,?,?)', (args['decision_id'], role, value, time.time()))
        return {'decision_id': args['decision_id'], 'recorded': True, 'verified_fix': False}


def host_action(snapshot, proposal):
    """The proposal selects a scoped investigator; never authorizes WI mutation."""
    if snapshot['kind'] == 'plan': return 'product', 'minerva', 'pending'
    wi = snapshot['detail']['work_item']
    phase, reason = wi.get('current_phase'), wi.get('parked_reason') or ''
    if phase in ('done', 'cancelled', 'draft'): return 'no_intervention', None, 'observed'
    if reason in ('operator-hold', 'human-parked'): return 'preserve_hold', None, 'held'
    if reason.startswith('gate-needs-human'): return 'human_required', None, 'human_required'
    # Restoring infrastructure need not replace a task owner. Vesta retains
    # the explicit ownership constraint in its scoped brief and tool grants.
    if phase == 'parked' and (reason in ('ci-infra-failing', 'ci-toolchain-broken') or reason.startswith('lane-circuit-breaker')):
        return 'infrastructure', 'vesta', 'pending'
    if snapshot.get('live_sessions'): return 'wait_for_owner', None, 'waiting'
    if wi.get('blocked_by') and phase != 'parked': return 'wait_for_dependency', None, 'waiting'
    if phase != 'parked': return 'scheduler_owned', None, 'observed'
    # These are host-owned typed classes. Laya cannot reclassify a hard ceiling
    # as permission to retry, nor classify a broken runner as a scope defect.
    if reason == 'too-big': return 'product', 'minerva', 'pending'
    if snapshot.get('missing'): return 'investigate', 'janus', 'pending'
    choice = proposal.get('choice', 'investigate')
    if choice not in CHOICES: choice = 'investigate'
    # Human/wait suggestions require a scoped evidence check. Only host facts
    # above establish an authoritative hold/wait; scores cannot silence work.
    if choice in ('human', 'wait'): choice = 'investigate'
    return choice, ROLE[choice], 'pending'


class Controller:
    def __init__(self, root):
        from observer import Observer
        self.root = Path(root)
        self.config = json.loads((self.root/'config/laya.json').read_text())
        self.queue = Queue(root)
        self.observer = Observer(root)

    def hydrate(self, change):
        ident = change['id']
        if change['kind'] == 'plan':
            return {'kind': 'plan', 'detail': self.observer.mcp('get_plan', {'id': ident}), 'missing': []}
        snapshot = {'kind': 'work_item', 'missing': []}
        reads = [('detail', 'get_work_item_detail', {'id': ident, 'message_page_size': 100, 'change_page_size': 100}),
                 ('forensics', 'get_work_item_forensics', {'work_item_id': ident, 'max_events': 500}),
                 ('loop', 'get_wi_loop_state', {'work_item_id': ident}),
                 ('scheduling', 'get_scheduling_state', {'work_item_id': ident}),
                 ('sessions', 'list_sessions', {'work_item_id': ident})]
        for key, tool, args in reads:
            try: snapshot[key] = self.observer.mcp(tool, args)
            except Exception as error: snapshot['missing'].append({'source': key, 'error': type(error).__name__})
        if 'detail' not in snapshot: raise RuntimeError('current WI state unavailable')
        detail = snapshot['detail']
        # Aggregate cost statistics are unrelated mutable telemetry, not WI evidence.
        detail.pop('ai_review_token_cost', None)
        detail.pop('session_costs', None)
        if detail.get('thread_messages_elided') or len(detail.get('thread_messages', [])) >= 100:
            snapshot['missing'].append({'source': 'thread', 'error': 'bounded_history'})
        if len(detail.get('change_history', [])) >= 100:
            snapshot['missing'].append({'source': 'changes', 'error': 'bounded_history'})
        if snapshot.get('forensics', {}).get('events_truncated'):
            snapshot['missing'].append({'source': 'forensics', 'error': 'bounded_history'})
        if any(e.get('witnesses_truncated') for e in snapshot.get('forensics', {}).get('events', [])):
            snapshot['missing'].append({'source': 'witnesses', 'error': 'bounded_history'})
        sessions = snapshot.get('sessions', {})
        sessions = sessions.get('sessions', []) if isinstance(sessions, dict) else sessions
        snapshot['live_sessions'] = [s['id'] for s in sessions if s.get('status') in ('provisioning', 'running', 'blocked')]
        plan = detail['work_item'].get('plan_id')
        if plan:
            try: snapshot['plan'] = self.observer.mcp('get_plan', {'id': plan})
            except Exception as error: snapshot['missing'].append({'source': 'plan', 'error': type(error).__name__})
        return snapshot

    def classify(self, snapshot):
        from observer import fetch
        token = (self.root/'secrets/laya.key').read_text().strip()
        sid = digest(snapshot)
        try:
            result = fetch('http://127.0.0.1:18446/decide', {'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'},
                           {'schema_version': 1, 'snapshot_id': sid, 'evidence': snapshot}, timeout=45)
            if result.get('snapshot_id') != sid or result.get('schema_version') != 1 or result.get('choice') not in CHOICES or result.get('authorizes_mutation') is not False:
                raise ValueError('invalid or stale inference result')
            return result
        except Exception as error:
            return {'choice': 'investigate', 'error': type(error).__name__, 'snapshot_id': sid, 'authorizes_mutation': False}

    def evaluate_one(self):
        row = self.queue.db.execute('select entity,change_json,revision from queue where queued_at<=? order by queued_at limit 1', (time.time(),)).fetchone()
        if not row: return False
        entity, raw, revision = row
        change = json.loads(raw)
        try: snapshot = self.hydrate(change)
        except Exception:
            with self.queue.db:
                self.queue.db.execute('update queue set queued_at=? where entity=?', (time.time()+60, entity))
            raise
        sid = digest(snapshot)
        waiting = self.queue.db.execute('select value from metadata where key=?', ('wait:'+entity,)).fetchone()
        if waiting:
            guard = json.loads(waiting[0])
            wi = snapshot.get('detail', {}).get('work_item', {})
            unchanged = sid == guard['snapshot_id']
            if guard['outcome'] == 'human_required' and wi.get('current_phase') == 'parked':
                new_reply = any(x.get('sender_type') == 'human' and x.get('created_at', '') > guard['recorded_at'] for x in snapshot['detail'].get('thread_messages', []))
                unchanged = not new_reply
            if unchanged:
                with self.queue.db: self.queue.db.execute('delete from queue where entity=? and revision=?', (entity, revision))
                return True
            with self.queue.db: self.queue.db.execute('delete from metadata where key=?', ('wait:'+entity,))
        ident = digest({'entity': entity, 'snapshot_id': sid, 'policy': 1, 'model': self.config['model_revision']})
        prior = self.queue.db.execute('select state from decisions where id=?', (ident,)).fetchone()
        if prior and prior[0] == 'turn_completed':
            with self.queue.db:
                self.queue.db.execute("update decisions set state='human_required',action='no_verified_progress' where id=?", (ident,))
        if not prior:
            proposal = self.classify(snapshot)
            if proposal.get('input_complete') is False:
                with self.queue.db:
                    self.queue.db.execute('update queue set queued_at=? where entity=?', (time.time()+5, entity))
                return True
            action, role, state = host_action(snapshot, proposal)
            with self.queue.db:
                self.queue.db.execute('insert or ignore into decisions values (?,?,?,?,?,?,?,?,?)',
                                      (ident, entity, encoded(snapshot), encoded(proposal), action, role, state, None, time.time()))
        with self.queue.db:
            self.queue.db.execute('delete from queue where entity=? and revision=?', (entity, revision))
        return True

    def dispatch_one(self):
        if not enabled(self.root): return
        if json.loads((self.root/'config/laya.json').read_text()).get('admit_interventions') is not True: return
        activity = self.observer.get('role_activity', {})
        if not activity.get('ok') or time.time()-activity.get('at', 0) > 120: return
        for row in self.queue.db.execute("select id,entity,snapshot,action,role from decisions where state='pending' order by created_at").fetchall():
            ident, entity, raw, action, role = row
            if self.queue.db.execute("select 1 from decisions where (role=? or entity=?) and state in ('queued','running','unknown')", (role,entity)).fetchone(): continue
            if not self.observer.role_ready(role, rest=0): continue
            snapshot = json.loads(raw)
            # Re-read immediately before admission: an old park decision cannot
            # dispatch a fixer against a released hold, moved head or live owner.
            change = {'kind': snapshot['kind'], 'id': entity.split(':', 1)[1]}
            fresh = self.hydrate(change)
            if digest(fresh) != digest(snapshot):
                with self.queue.db:
                    self.queue.db.execute("update decisions set state='stale' where id=?", (ident,))
                self.queue.observe(dict(change, current=fresh), force=True)
                continue
            wi = snapshot.get('detail', {}).get('work_item', {})
            parks = [e for e in snapshot.get('forensics', {}).get('events', []) if e.get('to_phase') == 'parked']
            episode = digest({'entity': entity, 'park': parks[-1] if parks else wi.get('parked_reason', 'plan')})
            budget = self.queue.db.execute('select attempts from intervention_budget where episode=?', (episode,)).fetchone()
            if budget and budget[0] >= 2:
                with self.queue.db: self.queue.db.execute("update decisions set state='human_required',action='intervention_budget' where id=?", (ident,))
                continue
            generation = self.queue.db.execute('select value from metadata where key=?', ('event_generation:'+ident,)).fetchone()
            event_id = 'laya-'+ident+('-g'+generation[0] if generation else '')
            data = {'decision_id': ident, 'snapshot_id': digest(snapshot), 'work_item_id': wi.get('id'),
                    'entity': entity,
                    'project_id': wi.get('project_id'), 'action': action, 'evidence_complete': not snapshot.get('missing'),
                    'evidence_sources': list(snapshot), 'missing': snapshot.get('missing', []),
                    'instruction': SCOPE[role]+' Read the full current WI detail, forensics, thread and referenced artifacts before acting. This is a bounded intervention, not a general project scan. If this is a legitimate human request, preserve the park and record the exact question and owner. Do not release explicit holds, reset budgets, bypass gates, or infer success from occupancy. Record evidence in the WI thread, then call record_fixer_result with this decision_id and typed outcome, source references, detail and next event. A fix_applied report is verified separately. Do not repeat a prior unsuccessful action without new evidence.'}
            # Observer's existing immutable, idempotent outbox is the delivery
            # boundary. A crash before our local receipt safely reuses event_id.
            kind = 'laya.'+role+'.intervention'
            source = 'roman-alerts' if role == 'vesta' else 'roman-anima'
            with self.observer.db: self.observer.event(source, kind, data, event_id)
            with self.queue.db:
                self.queue.db.execute("update decisions set state='queued',event_id=? where id=?", (event_id, ident))
                self.queue.db.execute('insert into intervention_budget values (?,1) on conflict(episode) do update set attempts=attempts+1', (episode,))
            break

    def reconcile_deliveries(self):
        for ident, entity, event_id in self.queue.db.execute("select id,entity,event_id from decisions where state in ('queued','running')").fetchall():
            deliveries = self.observer.get('event_deliveries:'+event_id, [])
            if not deliveries: continue
            rows = [self.observer.db.execute('select state,detail from deliveries where id=?', (d,)).fetchone() for d in deliveries]
            if any(not row or row[0] in ('pending','running') for row in rows): continue
            states = {row[0] for row in rows}
            state = 'turn_completed' if states == {'completed'} else 'unknown' if 'unknown' in states else 'intervention_failed'
            report = self.queue.db.execute('select result,created_at from fixer_results where decision_id=?', (ident,)).fetchone()
            if state == 'turn_completed' and report:
                outcome = json.loads(report[0])['outcome']
                state = {'human_required': 'human_required', 'waiting': 'waiting', 'blocked': 'intervention_failed', 'fix_applied': 'turn_completed'}[outcome]
                if outcome in ('human_required', 'waiting'):
                    kind, wid = entity.split(':', 1)
                    current = self.hydrate({'kind': kind, 'id': wid})
                    guard = {'snapshot_id': digest(current), 'outcome': outcome,
                             'recorded_at': datetime.fromtimestamp(report[1], timezone.utc).isoformat(), 'result': json.loads(report[0])}
                    with self.queue.db: self.queue.db.execute('insert or replace into metadata values (?,?)', ('wait:'+entity, encoded(guard)))
            with self.queue.db: self.queue.db.execute('update decisions set state=? where id=?', (state, ident))
            # A completed agent turn is not proof of a fix. Fresh evidence is
            # evaluated after a terminal receipt; no past tool action is replayed.
            if state == 'turn_completed':
                kind, wid = entity.split(':', 1)
                self.queue.observe({'kind': kind, 'id': wid, 'current': {'after_intervention': ident}}, force=True)

    def backfill(self):
        # Deterministic reconciliation only; this does not wake a model on a
        # timer. It detects missed/new parks and material evidence changes.
        cursor = None
        for _ in range(32):
            args = {'current_phase': 'parked', 'page_size': 100}
            if cursor: args['page_token'] = cursor
            page = self.observer.mcp('list_work_items', args)
            for item in page['work_items']:
                self.queue.observe({'kind': 'work_item', 'id': item['id'], 'current': item})
            cursor = page.get('next_page_token')
            if not cursor: return
        raise RuntimeError('park reconciliation exceeded page budget')

    def tick(self):
        row = self.queue.db.execute("select value from metadata where key='backfill_at'").fetchone()
        if not row or time.time()-float(row[0]) > 60:
            self.backfill()
            with self.queue.db: self.queue.db.execute("insert or replace into metadata values ('backfill_at',?)", (str(time.time()),))
        self.reconcile_deliveries()
        self.evaluate_one()
        self.dispatch_one()
        with self.queue.db: self.queue.db.execute("insert or replace into metadata values ('heartbeat',?)", (str(time.time()),))
        counts = dict(self.queue.db.execute('select state,count(*) from decisions group by state'))
        attention = [{'entity': e, 'action': a, 'report': json.loads(r) if r else None} for e,a,r in self.queue.db.execute("select d.entity,d.action,r.result from decisions d left join fixer_results r on r.decision_id=d.id where d.state in ('human_required','unknown','intervention_failed') order by d.created_at desc limit 40")]
        with self.observer.db:
            self.observer.put('laya_status', {'ok': True, 'at': int(time.time()), 'event_only': enabled(self.root),
                'model_revision': self.config['model_revision'], 'counts': counts, 'attention': attention,
                'pending_observations': self.queue.db.execute('select count(*) from queue').fetchone()[0]})


RESULT_TOOL = {'name': 'record_fixer_result', 'description': 'Record the typed outcome of your current Laya-scoped intervention. This records evidence; it does not change Anima phase, clear a gate, verify a fix, or grant authority. For human_required, detail must contain the exact question and owner. Retry an identical report with the same decision_id.',
 'inputSchema': {'type': 'object', 'additionalProperties': False, 'properties': {
     'decision_id': {'type': 'string'},
     'outcome': {'type': 'string', 'enum': ['fix_applied', 'human_required', 'waiting', 'blocked']},
     'evidence_refs': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {'type': 'string'}},
     'detail': {'type': 'string', 'minLength': 1, 'maxLength': 4000},
     'next_event': {'type': 'string', 'enum': ['human_reply', 'dependency_changed', 'lane_recovered', 'ci_completed', 'review_completed', 'work_item_changed', 'none']}},
     'required': ['decision_id', 'outcome', 'evidence_refs', 'detail', 'next_event']}}


def serve_results(root):
    tokens = {r: (root/'secrets'/('laya-'+r+'.key')).read_text().strip() for r in SCOPE}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, status, data):
            body = encoded(data).encode(); self.send_response(status)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            self.reply(200, {'status': 'listening'}) if self.path == '/healthz' else self.reply(404, {})
        def do_POST(self):
            if self.path != '/mcp': return self.reply(404, {})
            role = next((r for r,t in tokens.items() if hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+t)), None)
            if not role: return self.reply(401, {})
            ident = None
            try:
                n = int(self.headers.get('Content-Length', '0'))
                if not 0 < n <= 32768: return self.reply(413, {})
                q = json.loads(self.rfile.read(n)); ident = q.get('id'); method = q.get('method')
                if method == 'notifications/initialized': return self.reply(202, {})
                if method == 'initialize': result = {'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'laya-fixer-results', 'version': '1'}}
                elif method == 'tools/list': result = {'tools': [RESULT_TOOL]}
                elif method == 'tools/call' and q['params']['name'] == RESULT_TOOL['name']:
                    queue = Queue(root)
                    try: data = queue.record_result(role, q['params']['arguments'])
                    finally: queue.close()
                    result = {'content': [{'type': 'text', 'text': encoded(data)}], 'isError': False}
                else: raise ValueError('unsupported method')
                self.reply(200, {'jsonrpc': '2.0', 'id': ident, 'result': result})
            except Exception as error:
                self.reply(200, {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32602, 'message': type(error).__name__}})
    ThreadingHTTPServer(('127.0.0.1', 18447), Handler).serve_forever()


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    lock = (root/'state/laya-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = Controller(root)
    threading.Thread(target=serve_results, args=(root,), daemon=True).start()
    while True:
        try: controller.tick()
        except Exception as error: print('Laya controller deferred: '+type(error).__name__, flush=True)
        time.sleep(2)
