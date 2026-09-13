//! Print only the canonical request digest used by recovery APIs.
use clap::Parser;
#[derive(Parser)]
struct Args {
    file: std::path::PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read;
    let args = Args::parse();
    let mut bytes = vec![];
    std::fs::File::open(args.file)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("JSON exceeds 16 MiB".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    println!("{}", akh_session_services::recovery::fingerprint(&value)?);
    Ok(())
}
