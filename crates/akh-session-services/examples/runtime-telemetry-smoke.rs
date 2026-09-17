//! A synthetic export probe: no provider requests, broker calls or session writes.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let telemetry = akh_session_services::telemetry::init("akh-runtime-smoke", "synthetic")?;
    let id = format!("otel-smoke-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs());
    let session = id.clone();
    tokio::task::spawn_blocking(move || {
        let span = tracing::info_span!("runtime.turn", session.id=%session, turn.id="synthetic-turn", test.synthetic=true);
        let _entered = span.enter();
        let mut turn = akh_session_services::telemetry::Operation::new("turn");
        let tool = tracing::info_span!("runtime.tool", tool.name="synthetic", tool.id="synthetic-tool");
        tool.in_scope(|| {
            let mut operation = akh_session_services::telemetry::Operation::new("tool");
            akh_session_services::telemetry::delta("reasoning", "Synthetic reasoning event; no model was invoked.");
            akh_session_services::telemetry::usage(&serde_json::json!({"prompt_tokens":17,"completion_tokens":3}));
            std::thread::sleep(std::time::Duration::from_millis(20));
            operation.success();
        });
        turn.success();
    }).await?;
    tokio::task::spawn_blocking(move || drop(telemetry)).await?;
    println!("Synthetic export probe session: {id}");
    Ok(())
}
