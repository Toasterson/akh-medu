# Exact Ptah GGUF tokenizer for Candle

The native tokenizer was generated directly from the installed GGUF metadata,
without substituting another Qwen model's vocabulary. No weights were executed
and no inference service was started during provisioning.

## Artifact identity

- GGUF: `/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
- Architecture/name: `qwen35` / `Qwen3.8 27B Abliterated`.
- Tokenizer: `gpt2` BPE, pre-tokenizer `qwen35`, 248,320 IDs and 247,587 merges.
- Active vocabulary: 248,077 entries; another 243 entries are GGUF UNUSED padding.
- Installed output: `/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json`.
- Output SHA-256: `e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.
- Embedded template SHA-256: `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

Sibling files `ptah-qwen38.tokenizer.verification.json` and
`ptah-qwen38.tokenizer.chat_template.jinja` contain the complete verification
record/token vectors and exact template. The GGUF digest above comes from the
installed content-addressed artifact name; this tokenizer procedure does not
rehash the 16.8 GB weight file.

## Verification and the published tokenizer discrepancy

The Ollama manifest identifies `orcarouter/Qwen3.8-27B-Uncensored` as the model
source. That repository's metadata was public but file resolution returned 401.
The [published JonathanColetti artifact](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored/tree/5bb7aa90f0efef548e87005b1fb7658e522b6b7f)
at pinned revision `5bb7aa90f0efef548e87005b1fb7658e522b6b7f` independently matches
every active token ID, every merge in order, and the complete chat template.
Its tokenizer.json SHA-256 is
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

However, copying that JSON would be incorrect for GGUF-compatible tokenization:
it contains the Qwen2 splitting regex and NFC normalization. Its accompanying
tokenizer_config.json overrides `pretokenize_regex` with the combining-mark-aware
Qwen35 regex. The generated artifact uses that verified Qwen35 regex and no
normalizer, following the GGUF BPE path in
[llama-vocab.cpp](https://github.com/ggml-org/llama.cpp/blob/89fe24240548456477870b2a627cd8021fea1e39/src/llama-vocab.cpp).

Concrete differential examples, encoded with tokenizers 0.22.2:

| Input | Generated GGUF tokenizer IDs | Published tokenizer.json IDs |
| --- | --- | --- |
| `हिन्दी` | 215989, 150127, 177453 | 90703, 181077, 28640, 148651, 42201 |
| `cafe` + U+0301 | 895, 1795, 52033 | 895, 56868 |

All nine checked probes roundtrip exactly through the generated tokenizer,
including decomposed combining marks, Hindi, Arabic, Chinese, emoji, whitespace,
ChatML, and XML tool calls. All 248,320 token-to-ID mappings are checked after
constructing the tokenizer, including unused padding entries. This established
metadata and algorithm agreement. Executable reference parity was subsequently
measured against the original Ptah llama-server, as recorded below.

## Executable reference parity, 2026-09-14

All **15 probes matched exactly**, comparing complete token-ID vectors rather
than token counts or decoded text. The reference was the original Ptah
llama-server, successfully loaded with the exact GGUF identified above, temporarily
listening on `127.0.0.1:18189`. Each request used
`POST /tokenize` with `{"content": input, "add_special": false}`. The generated
artifact was encoded with `tokenizers==0.22.2`,
`Tokenizer.from_file(...).encode(input, add_special_tokens=False).ids`.
The tokenizer SHA-256 remained
`e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.

Only tokenization requests were made; this comparison did not request model
inference or modify the tokenizer. The following deterministic record includes
every input and its full vector, identical on both implementations. Escapes make
combining marks and invisible whitespace explicit.

```json
[
  {"probe":"ascii","input":"Hello, Ptah! Read file.txt then report its contents.","both_ids":[9419,11,49351,1413,0,4310,999,3778,1179,1830,1141,8529,13]},
  {"probe":"hindi","input":"हिन्दी भाषा में उत्तर दें।","both_ids":[215989,150127,177453,181651,149050,23121,170239,164472,161732,230609,154845]},
  {"probe":"arabic","input":"العربية: مرحبًا بالعالم","both_ids":[179415,151479,25,164422,20704,150021,182946,149650]},
  {"probe":"chinese","input":"中文：请读取文件并报告结果。","both_ids":[99986,4960,96220,111388,97380,96172,97126,97989,1710]},
  {"probe":"japanese","input":"日本語：ファイルを読んでください。","both_ids":[247359,4960,161243,235079,68798,1710]},
  {"probe":"combining","input":"cafe\u0301 e\u0301 a\u0308 n\u0303","both_ids":[895,1795,52033,378,52033,264,136,230,307,136,225]},
  {"probe":"precomposed","input":"café é ä ñ","both_ids":[895,56868,3825,12344,1624,109]},
  {"probe":"fullwidth_digits","input":"１２３４５６７８９０ 2026 ٠١٢٣ १२३४","both_ids":[19496,24128,32405,44085,42059,57486,62925,62818,64729,25191,220,17,15,17,21,220,149,254,149,94,149,95,149,96,220,12256,100,12256,101,12256,102,12256,103]},
  {"probe":"emoji","input":"👩🏽‍💻 👨‍👩‍👧‍👦 🇨🇭 🦀","both_ids":[9008,239,102,9008,237,121,373,235,88995,119,59720,101,373,235,9008,239,102,373,235,9008,239,100,373,235,9008,239,99,10838,229,101,9008,229,255,10838,99,222]},
  {"probe":"whitespace","input":"\talpha\r\n\n beta  \n\t","both_ids":[70579,78035,13053,2228,197]},
  {"probe":"chatml","input":"<|im_start|>system\nBe precise.<|im_end|>\n<|im_start|>user\nRead file.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","both_ids":[248045,8678,198,3320,22825,13,248046,198,248045,846,198,4274,999,13,248046,198,248045,74455,198,248068,271,248069,271]},
  {"probe":"tool_xml","input":"<tool_call>\n<function=read_file>\n<parameter=path>\nnotes/हिन्दी１２.txt\n</parameter>\n</function>\n</tool_call>","both_ids":[248058,198,27,1628,86779,2378,29,198,27,15704,79114,29,198,17742,14,215989,150127,177453,19496,24128,3778,198,510,15704,29,198,510,1628,29,198,248059]},
  {"probe":"tool_response","input":"<tool_response>\n{\"success\":true,\"text\":\"cafe\u0301\"}\n</tool_response>","both_ids":[248066,198,4754,5453,763,1802,1288,1272,3147,895,1795,52033,8934,198,248067]},
  {"probe":"unicode_punctuation","input":"“Quoted”—‘text’… № ½ Ⅷ K ﬁ\u00a0end\u200b.","both_ids":[2005,2114,8979,60537,13607,1272,515,1873,93546,220,25229,220,68086,100,27717,103,31422,68337,3966,400,15231,13]},
  {"probe":"mixed_marks","input":"नमस्ते a\u0301\u0327 z\u200d🏳️‍🌈 ３.１４","both_ids":[58069,84237,150104,153348,264,52033,136,100,1110,373,235,9008,237,111,29545,373,235,9008,234,230,220,32405,13,19496,44085]}
]
```

This closes the executable tokenization-parity gap for these probes, including
the published tokenizer's previously observed Hindi/combining-mark differences.
It is not an exhaustive proof for every possible Unicode string or a claim about
generation/numerical parity.

## Special tokens and template behavior

GGUF CONTROL tokens become special tokens; USER_DEFINED tokens remain visible
when decoding with special-token skipping. There are 27 controls and six
user-defined tokens. BOS/PAD is 248044 (`<|endoftext|>`), EOS is 248046
(`<|im_end|>`), and automatic BOS insertion is disabled.

| Token | ID | Classification |
| --- | --- | --- |
| `<|im_start|>` | 248045 | CONTROL |
| `<tool_call>`, `</tool_call>` | 248058, 248059 | USER_DEFINED |
| `<tool_response>`, `</tool_response>` | 248066, 248067 | USER_DEFINED |
| `<think>`, `</think>` | 248068, 248069 | USER_DEFINED |

The embedded template uses XML function/parameter tool calls, not JSON tool-call
bodies. Thinking defaults on with an open `<think>` prefix and xhigh reasoning
instructions. With `enable_thinking=false`, the assistant generation prefix is
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. These findings describe this
exact artifact; the architecture name alone does not prove template identity for
every Qwen35 model.

## Reproduction

`scripts/gguf_qwen35_tokenizer.py` declares pinned uv dependencies:
`gguf==0.17.1`, `tokenizers==0.22.2`. It loads metadata only, verifies the pinned
reference, and refuses existing output paths. Run with a fresh output pathname:

```sh
uv run scripts/gguf_qwen35_tokenizer.py \
  /usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec \
  --output /path/to/fresh/ptah-qwen38.tokenizer.json
```

Omitting `--output` performs verification and prints the record without writing
artifacts. The script is deliberately restricted to GPT2 BPE with pre=qwen35 and
no automatic BOS; other tokenizer families fail explicitly.
