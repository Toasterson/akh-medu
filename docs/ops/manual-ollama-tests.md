# Manual Ollama test suite

WI-306 (`01a0144a-b545-7e90-b444-e72d13f20adc`) quarantines every unit test
that opens an Ollama-related socket or requires live LLM inference. These tests
remain valuable, but they are deliberately excluded from default CI because
loopback HTTP streaming is environmentally flaky on the CI hosts.

The default `cargo test --lib` and `cargo test --features postgres --lib`
commands run the pure Ollama configuration, serialization, parsing, prompt, and
fake-model tests. They do not open an Ollama socket.

Run the quarantined tests manually with:

```sh
just test-ollama-manual
```

The command runs ignored tests serially. Most use loopback mock servers or
exercise unreachable-endpoint behavior. The Librarian integration test requires
an Ollama endpoint serving `AKH_LIBRARIAN_MODEL` (default `gpt-oss:120b`). To use
a non-default endpoint or an authenticated remote endpoint, set
`AKH_LIBRARIAN_OLLAMA_URL` and `AKH_LIBRARIAN_OLLAMA_API_KEY` as documented by
`OllamaLibrarianModel::from_env`.

Because the suite includes environment- and model-dependent checks, it is an
operator-run diagnostic and must not be added to automated CI or release gates.
