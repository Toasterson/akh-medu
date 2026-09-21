//! Behavioral checks through the same completion interface used by akhrtd.
use akh_medu::nlu::candle_backend::{CandleBackend, CandleFinishReason, ModelFamily};

pub fn check(backend: &mut CandleBackend) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(backend.family(), ModelFamily::Qwen35);
    backend.set_context_limit(2048)?;
    const PROMPT: &str = "<|im_start|>user\nRemember the word cedar. Reply with only cedar.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let next = |prompt: &str, reply: &str| {
        format!(
            "{prompt}{reply}<|im_end|>\n<|im_start|>user\nWhat word did I ask you to remember? Reply with only that word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    };
    let first = backend.generate_completion_for_session("probe:a", PROMPT, 64, &mut || Ok(()))?;
    assert_eq!(first.finish_reason, CandleFinishReason::Stop);
    assert_eq!(first.reused_prompt_tokens, 0);
    assert!(
        first.text.to_lowercase().contains("cedar"),
        "first reply: {}",
        first.text
    );
    let prompt = next(PROMPT, &first.text);
    let resumed =
        backend.generate_completion_for_session("probe:a", &prompt, 64, &mut || Ok(()))?;
    assert_eq!(resumed.finish_reason, CandleFinishReason::Stop);
    assert!(
        resumed.reused_prompt_tokens > 0,
        "fixture must exercise real prefix reuse"
    );
    let reference =
        backend.generate_completion_for_session("probe:reference", &prompt, 64, &mut || Ok(()))?;
    assert_eq!(reference.finish_reason, CandleFinishReason::Stop);
    assert_eq!(resumed.text, reference.text);

    // Re-establish a reusable prefix, then reject a cap based on the FULL prompt,
    // even though only a suffix would need forwarding.
    backend.generate_completion_for_session("probe:a", PROMPT, 64, &mut || Ok(()))?;
    backend.set_context_limit(resumed.prompt_tokens + 63)?;
    assert!(
        backend
            .generate_completion_for_session("probe:a", &prompt, 64, &mut || Ok(()))
            .is_err()
    );
    backend.set_context_limit(2048)?;
    let recovered =
        backend.generate_completion_for_session("probe:a", &prompt, 64, &mut || Ok(()))?;
    assert_eq!(recovered.reused_prompt_tokens, 0);
    assert_eq!(recovered.text, reference.text);

    let third_prompt = next(&prompt, &recovered.text);
    let other =
        backend.generate_completion_for_session("probe:b", &third_prompt, 64, &mut || Ok(()))?;
    assert_eq!(
        other.reused_prompt_tokens, 0,
        "another key must evict state despite matching tokens"
    );
    let fourth_prompt = next(&third_prompt, &other.text);
    let evicted =
        backend.generate_completion_for_session("probe:a", &fourth_prompt, 64, &mut || Ok(()))?;
    assert_eq!(evicted.reused_prompt_tokens, 0);

    let mut checks = 0;
    assert!(
        backend
            .generate_completion_for_session("probe:a", &prompt, 64, &mut || {
                checks += 1;
                if checks == 8 {
                    return Err(akh_medu::nlu::error::NluError::LlmGenerationFailed {
                        reason: "injected cancellation after prefill".into(),
                    });
                }
                Ok(())
            })
            .is_err()
    );
    assert_eq!(checks, 8);
    let recovered =
        backend.generate_completion_for_session("probe:a", &prompt, 64, &mut || Ok(()))?;
    assert_eq!(recovered.reused_prompt_tokens, 0);
    assert_eq!(recovered.text, reference.text);
    let short = backend.generate_completion_for_session("probe:a", &prompt, 1, &mut || Ok(()))?;
    assert_eq!(short.finish_reason, CandleFinishReason::Length);
    let recovered =
        backend.generate_completion_for_session("probe:a", &prompt, 64, &mut || Ok(()))?;
    assert_eq!(recovered.reused_prompt_tokens, 0);
    assert_eq!(recovered.text, reference.text);
    println!(
        "{}",
        serde_json::json!({"phase":"continuation_checks", "pass":true,
        "reused_prompt_tokens":resumed.reused_prompt_tokens,
        "checks":["exact_prefix", "fresh_reply_equivalence", "full_context_cap", "session_eviction", "cancellation", "length_recovery"]})
    );
    Ok(())
}
