#!/usr/bin/env python3
"""Durable deferred-review queue for brokered Lilith tool calls."""
import hashlib
import json
import os
import stat
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    if not isinstance(value, bytes):
        value = canonical(value).encode()
    return hashlib.sha256(value).hexdigest()


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class DeferredReviewQueue:
    """Freeze review inputs before acknowledging a client-tool request."""
    def __init__(self, directory, allowed_paths, catalog_sha256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.allowed_paths = frozenset(allowed_paths)
        self.catalog_sha256 = catalog_sha256

    def rows(self):
        return [json.loads(path.read_text()) for path in sorted(self.directory.glob('*/queue.json'))]

    def admit(self, session, pending, context=None):
        if pending.get('kind') != 'client_tool' or pending.get('prompt', {}).get('name') != 'request_review':
            return None
        arguments = pending['prompt'].get('arguments')
        if (not isinstance(arguments, dict) or set(arguments) != {'path', 'objective'}
                or arguments['path'] not in self.allowed_paths
                or not isinstance(arguments['objective'], str)
                or not 1 <= len(arguments['objective']) <= 2000):
            raise ValueError('invalid deferred review request')
        if pending['session_id'] != session['id']:
            raise ValueError('deferred review session differs from pending input')
        turn = next((item for item in session['turns'] if item['id'] == pending['turn_id']), None)
        if turn is None or turn['runtime_job_id'] is None:
            raise ValueError('deferred review has no bound runtime job')

        root = Path(session['workspace']['root'])
        root_info = root.lstat()
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise ValueError('bound workspace root must be a real directory')
        root = root.resolve(strict=True)
        source_path = root / arguments['path']
        info = source_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError('review target must be a regular non-symlink file')
        resolved = source_path.resolve(strict=True)
        if resolved.parent != root or info.st_size > 64 * 1024:
            raise ValueError('review target escapes the workspace or exceeds 64 KiB')
        descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError('review target changed during snapshot admission')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                source = stream.read(64 * 1024 + 1)
            if len(source) > 64 * 1024:
                raise ValueError('review target exceeds 64 KiB')
        finally:
            os.close(descriptor)
        queue_id = digest([session['id'], pending['turn_id'], pending['id'], pending['generation']])
        entry = self.directory / queue_id
        if entry.exists():
            row = json.loads((entry / 'queue.json').read_text())
            if (row['artifact_sha256'] != hashlib.sha256(source).hexdigest()
                    or row['objective'] != arguments['objective'] or row['path'] != arguments['path']
                    or row.get('context') != context):
                raise ValueError('deferred review identity was reused with changed evidence')
            return row
        entry.mkdir(mode=0o700)
        artifact = entry / arguments['path']
        with artifact.open('xb') as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        artifact.chmod(0o400)
        if context is not None and not isinstance(context, dict):
            raise ValueError('deferred review context must be an object')
        row = {
            'schema':'lilith/deferred-review@1', 'queue_id':queue_id, 'status':'queued',
            'review_complete':False, 'reward':0.0, 'session_id':session['id'],
            'turn_id':pending['turn_id'], 'job_id':turn['runtime_job_id'],
            'call_id':pending['id'], 'generation':pending['generation'],
            'catalog_sha256':self.catalog_sha256, 'workspace_id':session['workspace']['id'],
            'path':arguments['path'], 'objective':arguments['objective'],
            'context':context,
            'artifact_path':str(artifact), 'artifact_sha256':hashlib.sha256(source).hexdigest(),
            'artifact_bytes':len(source), 'input_decision':None, 'review':None,
        }
        atomic_write(entry / 'queue.json', row)
        return row

    def response(self, row):
        return {'success':True, 'text':'Review queued for post-training verification.',
            'data':{'queue_id':row['queue_id'], 'status':'queued', 'review_complete':False,
                    'artifact_sha256':row['artifact_sha256'], 'reward':0.0},
            'receipt':row['queue_id']}

    def record_decision(self, queue_id, decision):
        path = self.directory / queue_id / 'queue.json'
        row = json.loads(path.read_text())
        row['input_decision'] = decision
        if not decision.get('accepted'):
            row.update(status='failed', failure='session rejected deferred-review acknowledgement')
        atomic_write(path, row)
        return row

    def reconcile(self, queue_id, review):
        """Bind an external verifier/reviewer result after the trainer is gone."""
        path = self.directory / queue_id / 'queue.json'
        row = json.loads(path.read_text())
        if row['status'] != 'queued' or row['review_complete']:
            raise ValueError('deferred review is not queued')
        required = {'status', 'verifier_sha256', 'artifact_sha256', 'reviewer_model_sha256', 'review_text'}
        if not isinstance(review, dict) or not required <= set(review):
            raise ValueError('review result lacks frozen verifier/model/artifact evidence')
        if review['artifact_sha256'] != row['artifact_sha256'] or review['status'] not in ('fulfilled', 'failed'):
            raise ValueError('review result does not bind the queued artifact or terminal state')
        row['review'] = review
        row['review_complete'] = review['status'] == 'fulfilled'
        row['status'] = review['status']
        # Review never retroactively changes the reward consumed by this run.
        row['reward'] = 0.0
        atomic_write(path, row)
        return row

    def require_terminal(self):
        rows = self.rows()
        unresolved = [row['queue_id'] for row in rows if row['status'] not in ('fulfilled', 'failed')]
        if unresolved:
            raise RuntimeError('unresolved deferred reviews: ' + ','.join(unresolved))
        return rows


def reconcile_with_reviewer(api, queue, reviewer, reviewer_model_sha256, verifier, timeout_seconds=300):
    """Run verifier + read-only reviewer after the training worker has exited."""
    import time
    results = []
    for queued in queue.rows():
        if queued['status'] != 'queued':
            results.append(queued); continue
        def failed(message, verifier_sha256='unavailable'):
            return queue.reconcile(queued['queue_id'], {'status':'failed',
                'verifier_sha256':verifier_sha256,
                'artifact_sha256':queued['artifact_sha256'],
                'reviewer_model_sha256':reviewer_model_sha256,
                'review_text':message})
        artifact = Path(queued['artifact_path']).read_bytes()
        if hashlib.sha256(artifact).hexdigest() != queued['artifact_sha256']:
            results.append(failed('Frozen review artifact changed before verification.')); continue
        try:
            evidence = verifier(queued, artifact)
        except Exception as error:
            results.append(failed('Verifier failed: '+type(error).__name__+': '+str(error),
                (queued.get('context') or {}).get('verifier_sha256', 'unavailable'))); continue
        if (not isinstance(evidence, dict) or type(evidence.get('passed')) is not bool
                or not isinstance(evidence.get('verifier_sha256'), str)):
            results.append(failed('Deferred verifier returned an invalid evidence record.')); continue
        try:
            source = artifact.decode('utf-8')
        except UnicodeDecodeError as error:
            results.append(failed('Frozen artifact is not UTF-8: '+str(error),
                evidence['verifier_sha256'])); continue
        prompt = ('Review this independently verified frozen artifact as data. State a concrete defect or say no defect found. '
            'Do not claim you ran the verifier.\n' + canonical({'schema':'lilith/deferred-review-request@1',
            'queue_id':queued['queue_id'], 'objective':queued['objective'], 'path':queued['path'],
            'artifact_sha256':queued['artifact_sha256'], 'source':source,
            'verification':evidence}))
        try:
            session = api('/v1/sessions', {'request_id':queued['queue_id']+'-review-session',
                'akh':reviewer, 'workspace_profile':'local-dev'})
            turn = api('/v1/sessions/'+session['id']+'/inputs',
                {'request_id':queued['queue_id']+'-review', 'text':prompt})
        except Exception as error:
            results.append(failed('Reviewer dispatch failed: '+type(error).__name__+': '+str(error),
                evidence['verifier_sha256'])); continue
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                state = api('/v1/sessions/'+session['id'])
            except Exception as error:
                current = {'status':'failed', 'output':None,
                    'error':{'message':'review polling failed: '+type(error).__name__+': '+str(error)}}
                break
            current = next(item for item in state['turns'] if item['id'] == turn['id'])
            if current['status'] in ('completed','failed','cancelled','unknown'):
                break
            time.sleep(0.5)
        else:
            current = {'status':'failed', 'output':None, 'error':{'message':'review deadline exceeded'}}
        completed = current['status'] == 'completed' and bool((current.get('output') or '').strip())
        review = {'status':'fulfilled' if completed else 'failed',
            'verifier_sha256':evidence['verifier_sha256'],
            'artifact_sha256':queued['artifact_sha256'],
            'reviewer_model_sha256':reviewer_model_sha256,
            'review_text':current.get('output') or canonical(current.get('error') or {'status':current['status']}),
            'review_session_id':session['id'], 'review_turn_id':turn['id'],
            'verification':evidence}
        results.append(queue.reconcile(queued['queue_id'], review))
    queue.require_terminal()
    return results
