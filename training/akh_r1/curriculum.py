"""Whole-episode validation and target-only chat materialization.

Annotations and examiner outcomes never enter model messages. Each decision sees
only earlier events. Splits are assigned to entire command/workflow families.
"""
import hashlib
import json
from pathlib import Path
from protocol import parse_action

SCHEMA = 'akh-r1-episode-v1'
SPLITS = {'train', 'validation', 'test'}
KATA = {'Yes-And', 'Mirror', 'Redirect-Refusal', 'Stubborn Affirmation',
        'Double Diamond', 'Clarify-Before-Answer', 'Silence', 'Naming-the-Move'}
FORMS = {'Working Agreement', 'Observe and Inquire', 'Competing Explanations',
         'Small Intervention', 'Response and Repair', 'Return Ownership and Learning'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def validate_episode(episode, family_splits):
    required = {'schema', 'id', 'family', 'split', 'provenance', 'initial_messages',
                'events', 'annotations', 'examiner'}
    if set(episode) != required or episode['schema'] != SCHEMA:
        raise ValueError('Episode fields or schema are invalid')
    if episode['split'] not in SPLITS or family_splits.get(episode['family']) != episode['split']:
        raise ValueError('Family split mismatch')
    if not isinstance(episode['id'], str) or not episode['id'] or not episode['provenance'].get('kind'):
        raise ValueError('Missing identity or provenance')
    for message in episode['initial_messages']:
        if (set(message) != {'role', 'content'} or message['role'] not in {'system', 'user'}
            or not isinstance(message['content'], str)):
            raise ValueError('Initial messages may contain only system/user text')
    seen = set()
    receipts = {}
    previous = None
    for event in episode['events']:
        if set(event) - {'id', 'role', 'content', 'evidence_refs', 'observes', 'train_target'}:
            raise ValueError('Undeclared event fields could leak examiner data')
        if not isinstance(event.get('content'), str) or event.get('role') not in {'assistant', 'user', 'tool'}:
            raise ValueError('Invalid event role/content')
        if not isinstance(event.get('id'), str) or not event['id'] or event['id'] in seen:
            raise ValueError('Missing or duplicate event ID')
        if 'train_target' in event and (event['role'] != 'assistant'
                                       or not isinstance(event['train_target'], bool)):
            raise ValueError('Only assistant events may declare a boolean train_target')
        if not isinstance(event.get('evidence_refs', []), list) or not set(event.get('evidence_refs', [])).issubset(seen):
            raise ValueError('Evidence must precede the decision that cites it')
        if event['role'] == 'tool':
            if previous is None or previous['role'] != 'assistant' or event.get('observes') != previous['id']:
                raise ValueError('Tool observation must follow its actual action')
            try:
                receipt = json.loads(event['content'])
                if isinstance(receipt, dict) and isinstance(receipt.get('evidence_id'), str):
                    receipts[event['id']] = receipt['evidence_id']
            except json.JSONDecodeError:
                pass
        elif 'observes' in event:
            raise ValueError('Only a tool observation may bind to an action')
        if event['role'] == 'assistant' and event['content'].startswith('<tool_call>'):
            action = parse_action(event['content'], 'native')
            if action['tool'] == 'remember':
                visible = {receipts[ref] for ref in event.get('evidence_refs', []) if ref in receipts}
                if action['arguments'].get('source') not in visible:
                    raise ValueError('A teaching citation must use a visible earlier receipt ID')
        seen.add(event['id'])
        previous = event
    for annotation in episode['annotations']:
        if annotation['at_event'] not in seen:
            raise ValueError('Annotation refers to an absent event')
        if annotation.get('kata') not in KATA | {None} or annotation.get('form') not in FORMS | {None}:
            raise ValueError('Unknown curriculum label')
    if episode['examiner'].get('effect_success') is not True:
        raise ValueError('Only verified positive episodes enter this compiler; retain failures separately')


def decision_rows(episode, family_splits, tools):
    validate_episode(episode, family_splits)
    history = [dict(m) for m in episode['initial_messages']]
    for event in episode['events']:
        message = dict(role=event['role'], content=event['content'])
        if event['role'] == 'assistant' and event.get('train_target', True):
            yield dict(messages=[*history, message], tools=tools,
                       episode_id=episode['id'], decision_id=event['id'],
                       family=episode['family'], split=episode['split'],
                       episode_sha256=canonical_hash(episode))
        history.append(message)


def compile_corpus(episodes, family_splits, tools_by_family, output):
    output = Path(output)
    episodes = list(episodes)
    counts = {split: dict(episodes=0, decisions=0) for split in sorted(SPLITS)}
    identities = set()
    rows = {split: [] for split in SPLITS}
    by_hash = {canonical_hash(episode): episode for episode in episodes}
    for episode in episodes:
        if episode['id'] in identities:
            raise ValueError('Duplicate episode identity')
        identities.add(episode['id'])
        parent_hash = episode['provenance'].get('acquisition_episode_sha256')
        if parent_hash:
            parent = by_hash.get(parent_hash)
            if (parent is None or parent['family'] != episode['family']
                or parent['split'] != episode['split']
                or parent['provenance'].get('phase') != 'acquire'):
                raise ValueError('Cold graph lineage must remain within its acquisition family and split')
        split = episode['split']
        rendered = list(decision_rows(episode, family_splits, tools_by_family[episode['family']]))
        counts[split]['episodes'] += 1
        counts[split]['decisions'] += len(rendered)
        rows[split].extend(rendered)
    output.mkdir(parents=True, exist_ok=False)
    # MLX-LM expects valid.jsonl, while the manifest retains the full split name.
    for split, data in rows.items():
        name = 'valid' if split == 'validation' else split
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data))
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in episodes))
    manifest = dict(schema='akh-r1-curriculum-v1', family_splits=family_splits, counts=counts,
                    families_sha256=canonical_hash(family_splits), episodes_sha256=canonical_hash(episodes),
                    tools_sha256=canonical_hash(tools_by_family),
                    label_visibility='annotations and examiner values are never rendered',
                    split_unit='entire command contract family, including cold and language variants',
                    release_certification=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
