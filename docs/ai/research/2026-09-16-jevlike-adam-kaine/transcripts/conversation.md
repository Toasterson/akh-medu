# Jevlike conversation — verbatim

User-facing messages through the harvest request; private reasoning, tools, and system/developer instructions excluded. Historical wording, citations, and caveats are unchanged.

## 1. user

So, somebody tried their hand at a Jev https://x.com/vinnylarouge/status/2100170846346097083

## 2. assistant

I’ll open the post and check how it relates to Jev.

## 3. assistant

Yes—and this gives us a concrete baseline for the decision-head experiment we discussed. The post links to [**jevlike**](https://github.com/vinnylarouge/jevlike), an independent implementation inferred from Jev’s input/output interface.

Its mechanism is straightforward:

- Encode the context and each candidate option.
- Let each option attend to the context.
- Score those pairs with a shared scorer.
- Softmax across the candidates, producing a decision distribution in one pass.

It supports a small byte encoder trained from scratch or a frozen pretrained encoder with a trainable scoring head. That makes it a practical starting point for variable-menu decisions. [Architecture and training](https://github.com/vinnylarouge/jevlike#architecture)

**The evidence is more modest than “reverse-engineered Jev” suggests.** The README reports roughly 98% on synthetic tasks, 26% on target-disjoint Wikispeedia with frozen Qwen, and 29% from scratch. Its approximately 100× speedup compares against a decoder forced to emit 400 tokens; that doesn’t establish an advantage over a decoder producing only an option identifier. The author explicitly disclaims reproducing TypeSafe’s private training or matching Jev’s quality. [Results and limitations](https://github.com/vinnylarouge/jevlike#what-to-expect)

Connecting this to our earlier discussion, from memory: **ADAM is the clearest first fit**—score available actions against observations while keeping execution and verification in the engine. For **Kaine**, it supplies a candidate decision-head design; it doesn’t resolve the learning objective or calibration questions.

My take: this makes the architectural hypothesis testable. The useful next comparison is against constrained option selection on our own held-out decisions, measuring correctness, calibration, abstention, and actual latency.

<oai-mem-citation>
<citation_entries>
MEMORY.md:46-51|note=[Connected Jevlike to prior ADAM and Kaine experiments]
</citation_entries>
<rollout_ids>
01a0a98e-10f7-7af3-9aae-70685457a1a2
</rollout_ids>
</oai-mem-citation>

## 4. user

harvest it then

