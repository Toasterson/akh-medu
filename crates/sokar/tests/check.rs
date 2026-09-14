use std::process::Command;
#[test]
fn check_wrong_store_option_is_diagnostic_and_exit_two() {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "sokar-local-test-token-at-least-24").unwrap();
    let value = dir.path().join("option");
    std::fs::write(&value, "unused").unwrap();
    let config = dir.path().join("sokar.toml");
    std::fs::write(
        &config,
        format!(
            r#"
default_store = "bad"
[[principal]]
id = "test"
token_file = {token:?}
[[store]]
name = "bad"
service = "fs"
[store.options]
definitely_wrong_option = "file:{}"
"#,
            value.display()
        ),
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_sokard"))
        .arg("--config")
        .arg(&config)
        .arg("--check")
        .env("NO_COLOR", "1")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("sokar::storage"), "{stderr}");
    assert!(!stderr.contains("sokar-local-test-token-at-least-24"));
}
#[test]
fn check_and_reindex_local_fs_cli() {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "sokar-local-test-token-at-least-24").unwrap();
    let root = dir.path().join("root");
    std::fs::write(&root, dir.path().join("objects").to_str().unwrap()).unwrap();
    let config = dir.path().join("sokar.toml");
    std::fs::write(
        &config,
        format!(
            r#"
default_store = "local"
[[principal]]
id = "test"
token_file = {token:?}
[[store]]
name = "local"
service = "fs"
[store.options]
root = "file:{}"
"#,
            root.display()
        ),
    )
    .unwrap();
    for args in [
        vec!["--check"],
        vec!["--check-stores"],
        vec!["reindex", "--store", "local"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_sokard"))
            .arg("--config")
            .arg(&config)
            .args(args)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
#[test]
fn config_rejects_inline_secrets_and_unknown_fields() {
    assert!(toml::from_str::<sokar::config::DatabaseConfig>("url='postgres://inline'").is_err());
    assert!(toml::from_str::<sokar::config::Limits>("max_body_bytes=1024\ntypo=true").is_err());
    assert!(sokar::config::SecretRef::try_from("env:".to_owned()).is_err());
    assert!(sokar::config::SecretRef::try_from("file:relative".to_owned()).is_err());
}
