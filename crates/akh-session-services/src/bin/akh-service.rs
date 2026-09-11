use clap::Parser;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: std::path::PathBuf,
    #[arg(long)]
    login: bool,
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest: akh_session_services::helper::Manifest =
        serde_json::from_slice(&std::fs::read(args.manifest)?)?;
    if args.login {
        akh_session_services::helper::login(&manifest).await?;
        return Ok(());
    }
    akh_session_services::helper::supervise(manifest).await?;
    Ok(())
}
