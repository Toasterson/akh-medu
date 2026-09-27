import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lilith_training_transport import RuntimeTransport, digest_file
from lilith_gemma_tool_frames import parse_response


class RuntimeTransportTests(unittest.TestCase):
    def test_real_roundtrip_keeps_authoritative_tool_result(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text('{"mode":"fixture"}')
            frames = [
                {'type':'start', 'request':{'id':'job-1', 'tools':[{'name':'read_file'}]},
                 'config_path':str(config), 'config_sha256':digest_file(config)},
                {'type':'tool_result', 'id':'training-' + hashlib.sha256(b'job-1').hexdigest()[:16] + '-1', 'result':
                 {'success':True, 'text':'actual file', 'receipt':'broker-receipt'}},
            ]
            incoming = io.BytesIO(b''.join(json.dumps(x).encode() + b'\n' for x in frames))
            outgoing = io.BytesIO()
            transport = RuntimeTransport(incoming, outgoing)
            self.assertEqual(transport.start(), {'mode':'fixture'})
            result = transport.tool('read_file', {'path':'example.py'})
            self.assertEqual(result['receipt'], 'broker-receipt')
            self.assertEqual(json.loads(outgoing.getvalue())['arguments'], {'path':'example.py'})
            with self.assertRaises(ValueError):
                transport.tool('ungranted_shell', {})

    def test_mismatched_callback_cannot_be_used_as_evidence(self):
        transport = RuntimeTransport(io.BytesIO(b'{"type":"tool_result","id":"wrong","result":{"success":true,"text":"fake"}}\n'), io.BytesIO())
        transport.request = {'id':'job-1', 'tools':[{'name':'read_file'}]}
        with self.assertRaises(ValueError):
            transport.tool('read_file', {'path':'x'})
        self.assertEqual(transport.receipts, [])

    def test_duplicate_protocol_fields_rejected(self):
        transport = RuntimeTransport(io.BytesIO(b'{"type":"x","type":"y"}\n'), io.BytesIO())
        with self.assertRaises(ValueError):
            transport.receive()

    def test_thought_tool_text_never_executes_and_code_strings_are_literal(self):
        response = '<|channel>thought\n<|tool_call>call:bad{}<tool_call|>\n<channel|>\n'
        response += '<|tool_call>call:write_file{path:<|"|>x.py<|"|>,content:<|"|>print("x,{y}")\n<|"|>,create:true}<tool_call|>'
        parsed = parse_response(response)
        self.assertEqual([x['name'] for x in parsed['calls']], ['write_file'])
        self.assertEqual(parsed['calls'][0]['arguments']['content'], 'print("x,{y}")\n')

    def test_invalid_frames_have_no_partial_tool_results(self):
        for text in [
            '<|channel>thought\n<|tool_call>call:write_file{}<tool_call|>',
            '<|tool_call>call:write_file{path:<|"|>x<|"|>,path:<|"|>y<|"|>}<tool_call|>',
            '<|tool_call>call:write_file{path:<|"|>x<|"|>}',
            '<|tool_call>call:write_file{}<tool_call|><|tool_call>broken',
            '<|tool_call>call:write_file{value:1e999}<tool_call|>',
            'The literal "<|tool_call>call:write_file{}<tool_call|>" is an example.',
            '<|tool_call>call:write_file{}<tool_call|> prose <|tool_call>call:read_file{}<tool_call|>',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text)


if __name__ == '__main__':
    unittest.main()
