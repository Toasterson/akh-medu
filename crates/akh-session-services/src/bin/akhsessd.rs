#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    akh_session_services::server::run("akhsessd").await
}
