//! Cross-repository probe, launched by Anima's owned HTTP/database fixture.
use akh_session_services::{config::Endpoint, mcp};
use serde_json::json;

#[tokio::test]
#[ignore = "requires the owned Anima loopback fixture endpoint and private token file"]
async fn native_broker_reads_the_authenticated_anima_work_item() {
    let url = std::env::var("AKH_TEST_ANIMA_MCP_URL").expect("explicit Anima fixture URL");
    let parsed = reqwest::Url::parse(&url).unwrap();
    assert_eq!(parsed.scheme(), "http");
    assert_eq!(parsed.host_str(), Some("127.0.0.1"));
    assert_eq!(parsed.path(), "/mcp");
    let path = std::path::PathBuf::from(
        std::env::var_os("AKH_TEST_ANIMA_MCP_TOKEN_FILE").expect("private fixture credential file"),
    );
    assert!(path.is_absolute());
    let endpoint: Endpoint = serde_json::from_value(json!({"url":url,"token_file":path})).unwrap();
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
