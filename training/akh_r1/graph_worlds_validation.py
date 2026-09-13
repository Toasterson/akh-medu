"""Validation mechanisms frozen separately; no seed-only development copies."""

def generate():
    return [
      dict(id='val-acquisition',family='acquisition',mechanism='two_source_intersection',
           clauses=[['switch-a','candidate_zone','?answer'],['policy-b','approved_zone','?answer']],
           facts=[['switch-a','candidate_zone','zone-a'],['switch-a','candidate_zone','zone-b'],['policy-b','approved_zone','zone-a']],replacement=['policy-b','approved_zone','zone-b']),
      dict(id='val-reasoning',family='grounded_reasoning',mechanism='diamond_join_then_projection',
           clauses=[['service-b','owner','?team'],['schedule-b','on_call','?team'],['?team','queue','?answer']],
           facts=[['service-b','owner','team-b'],['service-b','owner','team-c'],['schedule-b','on_call','team-b'],['team-b','queue','channel-b'],['team-c','queue','channel-c']],replacement=['schedule-b','on_call','team-c']),
      dict(id='val-revision',family='knowledge_revision',mechanism='intermediate_edge_reroute',
           clauses=[['job-b','assigned','?worker'],['?worker','destination','?answer']],
           facts=[['job-b','assigned','worker-a'],['worker-a','destination','store-a'],['worker-b','destination','store-b']],replacement=['job-b','assigned','worker-b']),
    ]
