use clap::Parser;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: std::path::PathBuf,
    #[arg(long)]
    profile: String,
    #[arg(value_parser=["prepare","login","status","version","verify"])]
    action: String,
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = akh_session_services::config::Config::load(&args.config)?;
    let profile = config
        .runtimes
        .get(&args.profile)
        .ok_or("runtime profile not configured")?;
    if args.action == "verify" {
        akh_session_services::runtime::validate_conformance(profile)?;
        println!("{}: installed runtime conformance verified", args.profile);
        return Ok(());
    }
    akh_session_services::runtime_profile::action(profile, &args.action).await?;
    Ok(())
}
