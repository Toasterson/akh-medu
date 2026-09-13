"""Closed stdin-only Unix grammar; no file operands or executable expressions."""
import re


def validate_pipeline(stages, command, max_source_args=4):
    if not isinstance(stages,list) or not 1 <= len(stages) <= 6:
        raise ValueError('Expected one to six stages')
    if any(not isinstance(s,list) or not s or not all(isinstance(x,str) and len(x)<=256 and '\x00' not in x for x in s) for s in stages):
        raise ValueError('Invalid argv')
    if stages[0][0]!=command or not 1<=len(stages[0])<=max_source_args:
        raise ValueError('Source executable not exposed')
    # Composed reads only. No mutator may be hidden ahead of a filter.
    if len(stages)>1 and stages[0][1:]!=['records']:
        raise ValueError('Only records can feed pipelines')
    for s in stages[1:]:
        allowed=s in (['sort'],['sort','-u'],['uniq'],['wc','-l'])
        allowed=allowed or (len(s)==3 and s[:2]==['cut','-f'] and re.fullmatch('[1-9][0-9]?',s[2]) is not None)
        if not allowed:raise ValueError('Filter must use the closed stdin-only grammar')
    return stages
