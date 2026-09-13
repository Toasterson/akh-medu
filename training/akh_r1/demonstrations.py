"""Authored interface lessons about a different command; never fixture truth.

These are explicitly labeled synthetic examples. They teach discovery, argument
binding and storing a sourced command contract. They do not populate the KG.
The optional arm measures in-context scaffolding, not unaided or trained ability.
"""
import json


def call(name, arguments):
    return dict(role='assistant', content='<tool_call>\n'+json.dumps(dict(name=name, arguments=arguments))+'\n</tool_call>')


def messages():
    return [
        dict(role='user', content='Synthetic interface lesson, not the current lab. A new utility called daven is available. Discover its interface; do not guess its flags.'),
        call('terminal', dict(command='daven --help')),
        dict(role='tool', content=json.dumps(dict(success=True, stdout='daven v7. daven inspect --record KEY: read a record. daven mark --record KEY --label TEXT: attach a label without deleting the record. daven --version: report version.'))),
        call('remember', dict(subject='daven-v7', relation='inspect_syntax', value='daven inspect --record KEY', source='daven --help v7')),
        dict(role='tool', content='{"success":true,"result":"One sourced fact stored."}'),
        dict(role='user', content='Synthetic continuation: inspect the record whose key is oak-93. Substitute that actual identifier for the KEY placeholder.'),
        call('terminal', dict(command='daven inspect --record oak-93')),
        dict(role='tool', content='{"success":true,"stdout":"record=oak-93 label=pending"}'),
        call('report', dict(message='The observed record oak-93 has label pending.')),
        dict(role='tool', content='{"success":true,"result":"Message recorded."}'),
        dict(role='user', content='End of synthetic lesson. Its command and record do not exist in the next lab. Use the same discovery and evidence discipline with the next task, and use its actual identifiers.'),
    ]
