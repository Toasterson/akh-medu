"""Development mechanisms; identities vary, topology determines world identity."""

def generate():
    return [
      dict(id='dev-acquisition',family='acquisition',mechanism='direct_attribute',
           clauses=[['sensor-a','location','?answer']],
           facts=[['sensor-a','location','rack-west']],replacement=['sensor-a','location','rack-east']),
      dict(id='dev-reasoning',family='grounded_reasoning',mechanism='three_edge_forward_chain',
           clauses=[['alarm-a','service','?service'],['?service','owner','?owner'],['?owner','channel','?answer']],
           facts=[['alarm-a','service','svc-a'],['svc-a','owner','team-a'],['team-a','channel','queue-a']],replacement=['team-a','channel','queue-b']),
      dict(id='dev-revision',family='knowledge_revision',mechanism='guarded_scalar_revision',
           clauses=[['host-a','mode','?mode'],['policy-a','requires','?mode'],['host-a','route','?answer']],
           facts=[['host-a','mode','online'],['policy-a','requires','online'],['host-a','route','path-old']],replacement=['host-a','route','path-new']),
    ]
