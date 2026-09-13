"""Reference validation only: a resolvable receipt does not prove a fact true."""
from flat_protocol import project_fact


class UnresolvedSource(ValueError):
    pass


class ReceiptLinks:
    def __init__(self):
        self.visible = set()

    def observe(self, response):
        identity = response.get('evidence_id')
        if isinstance(identity, str) and identity:
            self.visible.add(identity)

    def resolves(self, source):
        return isinstance(source, str) and source in self.visible

    def require(self, source):
        if not self.resolves(source):
            raise UnresolvedSource('source must exactly match evidence_id from an earlier tool response in this episode')


def write_memory(bridge, tool, arguments, links=None):
    projected = project_fact(arguments) if tool == 'remember' else arguments
    if links is not None:
        links.require(projected.get('source'))
    return projected, bridge.call('memory_write', projected)


def memory_metrics(turns):
    links = ReceiptLinks()
    counts = dict(memory_write_attempts=0, accepted_memory_writes=0,
                  accepted_writes_with_resolved_receipts=0, rejected_source_links=0,
                  graph_queries=0, successful_graph_queries=0,
                  graph_queries_returning_triples=0, empty_graph_queries=0,
                  unclassified_successful_graph_queries=0)
    for row in turns:
        request, response = row.get('request', {}), row.get('response', {})
        tool = request.get('tool')
        if tool in {'remember', 'memory_write'}:
            counts['memory_write_attempts'] += 1
            if response.get('success'):
                counts['accepted_memory_writes'] += 1
                counts['accepted_writes_with_resolved_receipts'] += links.resolves(request['arguments'].get('source'))
        counts['rejected_source_links'] += response.get('error_code') == 'unresolved_source_reference'
        if tool == 'kg_query':
            counts['graph_queries'] += 1
            if response.get('success'):
                counts['successful_graph_queries'] += 1
                result = response.get('result', '')
                if result.startswith('No triples found'):
                    counts['empty_graph_queries'] += 1
                elif result.startswith('Found ') and ' triple(s)' in result:
                    counts['graph_queries_returning_triples'] += 1
                else:
                    counts['unclassified_successful_graph_queries'] += 1
        links.observe(response)
    return counts
