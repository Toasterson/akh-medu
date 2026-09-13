use clap::Parser;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: std::path::PathBuf,
    #[arg(long)]
    session: String,
    #[arg(long)]
    watch: bool,
    #[arg(long)]
    reconnect: bool,
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let text = std::fs::read_to_string(&args.config)?;
    let config = if args.config.extension().is_some_and(|e| e == "json") {
        serde_json::from_str(&text)?
    } else {
        toml::from_str(&text)?
    };
    akh_session_services::connect::run(config, &args.session, args.watch, args.reconnect).await?;
    Ok(())
}
