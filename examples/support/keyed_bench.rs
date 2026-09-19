//! Keyed multi-session timing for the resident native runtime.
//!
//! The backend keeps exactly one resident keyed session, so this measures the
//! two shapes a daemon actually produces: consecutive turns on one key, which
//! reuse an exact prefix, and turns that switch keys, which must prefill in
//! full. Greedy decoding makes the replies deterministic but the point here is
//! timing, not correctness; `continuation::check` owns the correctness gates.
use akh_medu::nlu::candle_backend::CandleBackend;
use std::{error::Error, time::Instant};

const TOPICS: [&str; 6] = [
    "Summarize why bounded caches matter in one sentence.",
    "Name one risk of retaining raw hidden states.",
    "Explain prefix reuse to an operator in one sentence.",
    "Give one reason to prefer features before performance.",
    "State one limit of device-wide VRAM telemetry.",
    "Describe an idle-power acceptance gate in one sentence.",
];

fn turn_prompt(history: &str, topic: &str) -> String {
    format!(
        "{history}<|im_start|>user\n{topic}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

#[cfg(feature = "vulkan")]
fn cache_stats(backend: &CandleBackend) -> Option<serde_json::Value> {
    backend.vulkan_pipeline_cache_stats().map(|stats| {
        serde_json::json!({
            "hits": stats.hits,
            "misses": stats.misses,
            "evictions": stats.evictions,
            "bypassed": stats.bypassed,
            "compile_failures": stats.compile_failures,
            "recency_resets": stats.recency_resets,
            "entries": stats.entries,
            "source_bytes": stats.source_bytes,
        })
    })
}

#[cfg(not(feature = "vulkan"))]
fn cache_stats(_backend: &CandleBackend) -> Option<serde_json::Value> {
    None
}

#[derive(Default)]
struct Group {
    requests: usize,
    decode_steps: usize,
    decode_seconds: f64,
    prefill_seconds: f64,
    prompt_tokens: usize,
    reused_tokens: usize,
    rates: Vec<f64>,
}

impl Group {
    fn summary(&self) -> serde_json::Value {
        let mut rates = self.rates.clone();
        rates.sort_by(|a, b| a.partial_cmp(b).expect("finite decode rates"));
        let median = if rates.is_empty() {
            None
        } else {
            Some(rates[rates.len() / 2])
        };
        serde_json::json!({
            "requests": self.requests,
            "decode_steps": self.decode_steps,
            "decode_seconds": self.decode_seconds,
            // Aggregate rate weights long replies correctly; the median does not.
            "aggregate_decode_steps_per_second": if self.decode_seconds > 0.0 {
                Some(self.decode_steps as f64 / self.decode_seconds)
            } else {
                None
            },
            "median_decode_steps_per_second": median,
            "mean_time_to_first_token_seconds": if self.requests > 0 {
                Some(self.prefill_seconds / self.requests as f64)
            } else {
                None
            },
            "mean_prompt_tokens": if self.requests > 0 {
                Some(self.prompt_tokens as f64 / self.requests as f64)
            } else {
                None
            },
            "mean_reused_prompt_tokens": if self.requests > 0 {
                Some(self.reused_tokens as f64 / self.requests as f64)
            } else {
                None
            },
        })
    }
}

/// `sessions` distinct keys, each carrying `turns` turns of its own history.
/// Phase `sequential` runs one key to completion before the next; phase
/// `interleaved` rotates keys every turn so each request pays a key switch.
pub fn run(
    backend: &mut CandleBackend,
    sessions: usize,
    turns: usize,
    tokens: u32,
) -> Result<(), Box<dyn Error>> {
    if sessions == 0 || turns == 0 {
        return Err("keyed benchmark needs at least one session and one turn".into());
    }
    if sessions > TOPICS.len() {
        return Err(format!("keyed benchmark supports at most {} sessions", TOPICS.len()).into());
    }
    let before = cache_stats(backend);
    let started = Instant::now();
    let mut reused_group = Group::default();
    let mut fresh_group = Group::default();
    fn record(
        phase: &str,
        session: usize,
        turn: usize,
        reply: &akh_medu::nlu::candle_backend::CandleCompletion,
        wall: f64,
        reused_group: &mut Group,
        fresh_group: &mut Group,
    ) {
        let rate = if reply.decode_steps > 0 && reply.decode_time.as_secs_f64() > 0.0 {
            Some(reply.decode_steps as f64 / reply.decode_time.as_secs_f64())
        } else {
            None
        };
        let group = if reply.reused_prompt_tokens > 0 {
            &mut *reused_group
        } else {
            &mut *fresh_group
        };
        group.requests += 1;
        group.decode_steps += reply.decode_steps as usize;
        group.decode_seconds += reply.decode_time.as_secs_f64();
        group.prefill_seconds += reply.time_to_first_token.as_secs_f64();
        group.prompt_tokens += reply.prompt_tokens;
        group.reused_tokens += reply.reused_prompt_tokens;
        if let Some(rate) = rate {
            group.rates.push(rate);
        }
        println!(
            "{}",
            serde_json::json!({"phase":"keyed_request","pass":phase,"session":session,"turn":turn,
                "seconds_including_prefill":wall,
                "prompt_tokens":reply.prompt_tokens,
                "reused_prompt_tokens":reply.reused_prompt_tokens,
                "tokens":reply.tokens_generated,
                "time_to_first_token_seconds":reply.time_to_first_token.as_secs_f64(),
                "decode_seconds":reply.decode_time.as_secs_f64(),
                "decode_steps_including_eos":reply.decode_steps,
                "decode_steps_per_second":rate,
                "finish":format!("{:?}",reply.finish_reason)})
        );
    }

    let mut histories: Vec<String> = vec![String::new(); sessions];
    for session in 0..sessions {
        for turn in 0..turns {
            let topic = TOPICS[(session + turn) % TOPICS.len()];
            let prompt = turn_prompt(&histories[session], topic);
            let key = format!("bench:s{session}");
            let at = Instant::now();
            let reply = backend.generate_completion_for_session(&key, &prompt, tokens, &mut || {
                Ok(())
            })?;
            record(
                "sequential",
                session,
                turn,
                &reply,
                at.elapsed().as_secs_f64(),
                &mut reused_group,
                &mut fresh_group,
            );
            histories[session] = format!("{prompt}{}<|im_end|>\n", reply.text);
        }
    }
    for turn in 0..turns {
        for session in 0..sessions {
            let topic = TOPICS[(session + turn + 1) % TOPICS.len()];
            let prompt = turn_prompt(&histories[session], topic);
            let key = format!("bench:s{session}");
            let at = Instant::now();
            let reply = backend.generate_completion_for_session(&key, &prompt, tokens, &mut || {
                Ok(())
            })?;
            record(
                "interleaved",
                session,
                turn,
                &reply,
                at.elapsed().as_secs_f64(),
                &mut reused_group,
                &mut fresh_group,
            );
            histories[session] = format!("{prompt}{}<|im_end|>\n", reply.text);
        }
    }
    let after = cache_stats(backend);
    println!(
        "{}",
        serde_json::json!({"phase":"keyed_summary",
            "sessions":sessions,"turns":turns,"max_tokens":tokens,
            "wall_seconds":started.elapsed().as_secs_f64(),
            "prefix_reused":reused_group.summary(),
            "full_prefill":fresh_group.summary(),
            "pipeline_cache_before":before,
            "pipeline_cache_after":after})
    );
    Ok(())
}
