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
    assert!(sokar::config::OptionValue::try_from("env:".to_owned()).is_err());
    assert!(matches!(
        sokar::config::OptionValue::try_from("my-bucket".to_owned()),
        Ok(sokar::config::OptionValue::Literal(v)) if v == "my-bucket"
    ));
    assert!(matches!(
        sokar::config::OptionValue::try_from("literal:env:notasecret".to_owned()),
        Ok(sokar::config::OptionValue::Literal(v)) if v == "env:notasecret"
    ));
    assert!(matches!(
        sokar::config::OptionValue::try_from("env:SOKAR_X".to_owned()),
        Ok(sokar::config::OptionValue::Secret(_))
    ));
}

#[test]
fn mcp_config_defaults_and_invalid_paths_are_diagnostic() {
    use sokar::config::{McpConfig, ServerConfig};
    let defaults: McpConfig = toml::from_str("").unwrap();
    assert!(defaults.enabled);
    assert_eq!(defaults.path.as_str(), "/mcp");
    assert_eq!(defaults.session_idle_secs, 900);
    assert_eq!(defaults.max_inline_bytes, 8_388_608);
    assert!(
        toml::from_str::<ServerConfig>("")
            .unwrap()
            .public_api_url
            .is_none()
    );
    assert!(toml::from_str::<McpConfig>("typo = true").is_err());
    assert!(toml::from_str::<ServerConfig>("public_url = 'http://example.com'").is_err());

    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "config-validation-token-at-least-24").unwrap();
    let input = format!(
        r#"
default_store = "local"
[[principal]]
id = "test"
token_file = {token:?}
[[store]]
name = "local"
service = "memory"
"#
    );
    let config: sokar::config::Config = toml::from_str(&input).unwrap();
    config.validate().unwrap();
    for path in [
        "",
        "/",
        "mcp",
        "/mcp/",
        "/v1",
        "/v1/artifacts",
        "/{id}",
        "/mcp?x",
        "/mcp#x",
        "/mcp//x",
        "/mcp/../x",
        "/mcp/./x",
        "/mcp\n",
        "/mcp%2Fx",
    ] {
        assert!(path.parse::<sokar::config::RoutePath>().is_err(), "{path}");
    }
    for path in ["/mcp", "/tools/mcp", "/mcp-v2"] {
        let mut config = config.clone();
        config.mcp.path = path.parse().unwrap();
        config.validate().unwrap();
    }
    for url in [
        "file:///tmp/content",
        "https://user:password@example.com",
        "https://example.com?x=1",
        "https://example.com#x",
    ] {
        let mut config = config.clone();
        config.server.public_api_url = Some(url.parse().unwrap());
        assert!(config.validate().is_err());
    }
    let mut config = config;
    config.mcp.session_idle_secs = 0;
    assert!(config.validate().is_err());
    config.mcp.session_idle_secs = 900;
    config.mcp.max_inline_bytes = 0;
    assert!(config.validate().is_err());
    config.mcp = toml::from_str(
        "enabled = false\npath = 'invalid'\nmax_inline_bytes = 0\nsession_idle_secs = 0",
    )
    .unwrap();
    config.validate().unwrap();
    config.mcp.enabled = true;
    assert!(matches!(
        config.validate(),
        Err(sokar::SokarError::RoutePath(
            sokar::config::RoutePathError::NotAbsolute
        ))
    ));
}

#[test]
fn local_example_with_mcp_checks_and_boots_both_listeners() {
    use std::{net::TcpListener, time::Duration};
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "example-config-token-at-least-24").unwrap();
    let api = TcpListener::bind("127.0.0.1:0").unwrap();
    let admin = TcpListener::bind("127.0.0.1:0").unwrap();
    let api_address = api.local_addr().unwrap();
    let admin_address = admin.local_addr().unwrap();
    let input = include_str!("../examples/local.toml")
        .replace("/tmp/sokar-local/token", token.to_str().unwrap())
        .replace("127.0.0.1:9400", &api_address.to_string())
        .replace("127.0.0.1:9401", &admin_address.to_string());
    let path = dir.path().join("local.toml");
    std::fs::write(&path, input).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_sokard"));
    command
        .arg("--config")
        .arg(&path)
        .env("SOKAR_FS_ROOT", dir.path().join("objects"));
    let check = command.arg("--check").output().unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(
        String::from_utf8_lossy(&check.stdout)
            .contains("Sokar config, index and all stores checked")
    );
    drop((api, admin));
    struct Child(std::process::Child);
    impl Drop for Child {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut child = Child(
        Command::new(env!("CARGO_BIN_EXE_sokard"))
            .arg("--config")
            .arg(&path)
            .env("SOKAR_FS_ROOT", dir.path().join("objects"))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    for _ in 0..100 {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "daemon exited before listening"
        );
        if std::net::TcpStream::connect(api_address).is_ok()
            && std::net::TcpStream::connect(admin_address).is_ok()
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("local example did not boot both listeners");
}

#[test]
fn route_path_reasons_help_and_disabled_validation() {
    use miette::Diagnostic;
    use sokar::config::{McpConfig, RoutePath, RoutePathError::*};
    for (path, reason) in [
        ("mcp", NotAbsolute),
        ("/mcp/", TrailingSlash),
        ("/mcp/../tools", DotSegment),
        ("/mcp//tools", DotSegment),
        ("/{tool}", IllegalByte),
        ("/v1/artifacts", ReservedV1),
    ] {
        assert_eq!(path.parse::<RoutePath>().unwrap_err(), reason);
        let help = Diagnostic::help(&reason).unwrap().to_string();
        assert!(!help.is_empty());
        let input = format!("path = {path:?}");
        assert!(
            toml::from_str::<McpConfig>(&input)
                .unwrap_err()
                .to_string()
                .contains(&reason.to_string())
        );
        let mut disabled: McpConfig = toml::from_str(&format!("enabled = false\n{input}")).unwrap();
        assert_eq!(disabled.path.as_str(), path);
        disabled.enabled = true;
        assert_eq!(disabled.path.validate().unwrap_err(), reason);
    }
    #[derive(serde::Deserialize)]
    struct PathOnly {
        path: RoutePath,
    }
    assert_eq!(
        toml::from_str::<PathOnly>("path = '/tools/mcp'")
            .unwrap()
            .path
            .as_str(),
        "/tools/mcp"
    );
    assert!(toml::from_str::<PathOnly>("path = 'relative'").is_err());
}
