//! Cross-repository probe, launched by Anima's owned HTTP/database fixture.
use akh_session_services::{config::Endpoint, mcp};
use serde_json::json;

#[tokio::test]
#[ignore = "requires the owned Anima loopback fixture endpoint and private token file"]
async fn native_broker_reads_the_authenticated_anima_work_item() {
    let endpoint = fixture();
    let result = mcp::call(
        &endpoint,
        "owned-native-broker-read",
        "get_work_item",
        json!({}),
    )
    .await
    .unwrap();
    assert!(result.success, "native MCP read failed");
    assert!(result.text.contains("MCP owned fixture"));
}

fn fixture() -> Endpoint {
    let url = std::env::var("AKH_TEST_ANIMA_MCP_URL").expect("explicit Anima fixture URL");
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert_eq!(parsed.scheme(), "http");
    assert_eq!(parsed.host_str(), Some("127.0.0.1"));
    assert_eq!(parsed.path(), "/mcp");
    let path = std::path::PathBuf::from(
        std::env::var_os("AKH_TEST_ANIMA_MCP_TOKEN_FILE").expect("private fixture credential file"),
    );
    assert!(path.is_absolute());
    serde_json::from_value(json!({"url":url,"token_file":path})).unwrap()
}

#[tokio::test]
#[ignore = "requires the owned Anima central-result fixture and explicit argument file"]
async fn native_broker_submits_an_anima_change_with_an_immutable_operation_receipt() {
    let endpoint = fixture();
    let operation = std::env::var("AKH_TEST_ANIMA_MCP_OPERATION").expect("owned operation ID");
    assert_eq!(operation.len(), 64);
    let path = std::path::PathBuf::from(
        std::env::var_os("AKH_TEST_ANIMA_MCP_ARGUMENTS").expect("owned argument file"),
    );
    assert!(path.is_absolute());
    let arguments: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let first = mcp::call(
        &endpoint,
        &operation,
        "submit_session_change",
        arguments.clone(),
    )
    .await
    .unwrap();
    assert!(
        first.success,
        "native result submission failed: {}",
        first.text
    );
    let receipt = first.data.as_ref().unwrap()["structuredContent"].clone();
    assert_eq!(receipt["operation_id"], operation);
    assert!(receipt["change_id"].is_string());
    // Explicit server receipt retry. The ordinary Broker still never
    // automatically repeats an uncertain MCP POST.
    let repeated = mcp::call(&endpoint, &operation, "submit_session_change", arguments)
        .await
        .unwrap();
    assert!(repeated.success);
    assert_eq!(repeated.text, first.text);
}
