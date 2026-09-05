//! Print the pre-training visual-token budget matrix required by ADR 100.

use serde::Serialize;

#[derive(Serialize)]
struct BudgetRow {
    input: &'static str,
    width: usize,
    height: usize,
    patch_size: usize,
    visual_tokens: usize,
    text_tokens: usize,
    total_tokens: usize,
    max_seq_len: usize,
    visual_fraction: f64,
    fits: bool,
    within_quarter_context: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let max_seq_len = argument("--max-seq")?.unwrap_or(1024);
    let text_tokens = argument("--text-tokens")?.unwrap_or(128);
    let inputs = [
        ("line", 384usize, 64usize),
        ("receipt", 320, 192),
        ("screenshot", 1024, 768),
        ("page", 1024, 1024),
    ];
    let mut rows = Vec::new();
    for (input, width, height) in inputs {
        for patch_size in [8usize, 16, 24, 32, 48] {
            let visual_tokens = width.div_ceil(patch_size) * height.div_ceil(patch_size);
            let total_tokens = visual_tokens + text_tokens;
            rows.push(BudgetRow {
                input,
                width,
                height,
                patch_size,
                visual_tokens,
                text_tokens,
                total_tokens,
                max_seq_len,
                visual_fraction: visual_tokens as f64 / max_seq_len.max(1) as f64,
                fits: total_tokens <= max_seq_len,
                within_quarter_context: visual_tokens * 4 <= max_seq_len,
            });
        }
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &rows)?;
    println!();
    Ok(())
}

fn argument<T: std::str::FromStr>(flag: &str) -> Result<Option<T>, Box<dyn std::error::Error>>
where
    T::Err: std::error::Error + 'static,
{
    let args: Vec<String> = std::env::args().collect();
    let Some(index) = args.iter().position(|value| value == flag) else {
        return Ok(None);
    };
    let value = args
        .get(index + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?;
    Ok(Some(value.parse()?))
}
