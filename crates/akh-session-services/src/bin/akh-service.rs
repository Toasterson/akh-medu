use clap::Parser;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: std::path::PathBuf,
    #[arg(long)]
    login: bool,
    /// Run a bounded upstream action in the same private profile. Pair output is a secret.
    #[arg(long, value_parser = ["pair", "login_status"], conflicts_with = "login")]
    action: Option<String>,
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest: akh_session_services::helper::Manifest =
        serde_json::from_slice(&std::fs::read(args.manifest)?)?;
    if let Some(action) = args.action {
        let output = akh_session_services::helper::action(&manifest, &action).await?;
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }
    if args.login {
        akh_session_services::helper::login(&manifest).await?;
        return Ok(());
    }
    akh_session_services::helper::supervise(manifest).await?;
    Ok(())
}
