"""Shared, explicitly assisted temporal profile for the public R0 interface.

Native event operations stay host-only. PublicEpisode enforces exact observed
source receipts before this adapter sees a candidate fact.
"""
import json
from public_episode import PublicEpisode

PROFILE_ID = 'akh-temporal-assistance-v1'

class TemporalEpisode(PublicEpisode):
    def checked_write(self,facts,source):
        super().checked_write(facts,source)
        for fact in facts:
            if fact['subject']=='operator' and fact['predicate']=='permission-event':
                event=json.loads(fact['object'])
                if not self.graph.call('exam_temporal_event',event).get('success'):raise RuntimeError('Native event rejected')
    def validate_fact_write(self,fact):
        if fact['subject']=='operator' and fact['predicate']=='permission-event':
            event=json.loads(fact['object'])
            if set(event)!={'id','timestamp','granted'} or not isinstance(event['id'],str) or not event['id'] or len(event['id'])>128 or type(event['timestamp']) is not int or not 0<=event['timestamp']<2**64 or type(event['granted']) is not bool:raise ValueError('Invalid temporal event')

class TemporalGuard:
    """Permit a configured effect only with an affirmative native state receipt."""
    def __init__(self, graph, effect_stages, terminal):
        self.graph = graph
        self.effect_stages = [list(stage) for stage in effect_stages]
        self.terminal = terminal
        self.receipts = []

    def __call__(self, stages):
        if stages == self.effect_stages:
            state = self.graph.call('exam_temporal_state')
            self.receipts.append(state)
            if state.get('success') is not True or state.get('holding') is not True:
                return dict(success=False, error='Native temporal permission absent')
        return self.terminal(stages)
