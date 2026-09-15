#![allow(dead_code)]
use sokar::{
    Sokar,
    config::*,
    index::{Index, MemoryIndex},
    store::StoreRegistry,
};
use sokar_client::*;
use std::{collections::BTreeMap, sync::Arc};

pub const TOKEN: &str = "sokar-local-test-token-at-least-24";
pub struct Fixture {
    pub dir: tempfile::TempDir,
    pub state: Arc<Sokar>,
    pub memory: Arc<MemoryIndex>,
    pub config: Arc<Config>,
}
impl Fixture {
    pub async fn new(fs: bool) -> Self {
        Self::with_index(fs, None).await
    }
    pub async fn with_index(fs: bool, index: Option<Arc<dyn Index>>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, TOKEN).unwrap();
        let root = dir.path().join("root-ref");
        std::fs::write(&root, dir.path().join("objects").to_str().unwrap()).unwrap();
        let name: StoreName = "local".parse().unwrap();
        let store = StoreConfig {
            name: name.clone(),
            service: if fs {
                ServiceKind::Fs
            } else {
                ServiceKind::Memory
            },
            prefix: Some("test-prefix".into()),
            options: if fs {
                BTreeMap::from([("root".into(), SecretRef::File(root).into())])
            } else {
                BTreeMap::new()
            },
        };
        let config = Arc::new(Config {
            server: ServerConfig::default(),
            mcp: McpConfig::default(),
            database: None,
            default_store: name,
            stores: vec![store],
            principals: vec![PrincipalConfig {
                id: "test-service".into(),
                token_file: Some(token),
                token_sha256_file: None,
            }],
            limits: Limits::default(),
            delivery: DeliveryConfig::default(),
        });
        config.validate().unwrap();
        let stores = Arc::new(StoreRegistry::build(&config.stores).await.unwrap());
        let memory = Arc::new(MemoryIndex::default());
        let state = Arc::new(Sokar::new(
            config.clone(),
            stores,
            index.unwrap_or_else(|| memory.clone()),
        ));
        Self {
            dir,
            state,
            memory,
            config,
        }
    }
    pub async fn serve(&self) -> Server {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = sokar::http::api(self.state.clone());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Server {
            client: Client::new(&base, TOKEN).unwrap(),
            base,
            task,
        }
    }
    pub async fn collection(&self) -> Collection {
        self.state
            .create_collection("test-service", create("test"))
            .await
            .unwrap()
    }
}
pub struct Server {
    pub client: Client,
    pub base: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
pub fn create(name: &str) -> CreateCollection {
    CreateCollection {
        scope: CollectionScope::Named { name: name.into() },
        store: None,
        retention: RetentionPolicy::Forever,
        quota: Quota::default(),
    }
}
pub fn options(path: &str) -> UploadOptions {
    UploadOptions {
        path: path.parse().unwrap(),
        media_type: "application/octet-stream".parse().unwrap(),
        kind: ArtifactKind::Document,
        labels: Labels::new(),
        idempotency_key: None,
    }
}

pub struct Daemon {
    pub client: Client,
    child: std::process::Child,
}
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Fixture {
    pub async fn serve_binary(&self) -> Daemon {
        let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = socket.local_addr().unwrap();
        let path = self.dir.path().join("daemon.toml");
        let token = self.config.principals[0].token_file.as_ref().unwrap();
        let mut config = format!(
            r#"
default_store = "{}"
[server]
api_bind = "{address}"
admin_bind = "127.0.0.1:0"
[[principal]]
id = "test-service"
token_file = {token:?}
"#,
            self.config.default_store
        );
        for store in &self.config.stores {
            config.push_str(&format!(
                "\n[[store]]\nname = {:?}\nservice = {:?}\n",
                store.name.as_str(),
                store.service.as_str()
            ));
            if let Some(prefix) = &store.prefix {
                config.push_str(&format!("prefix = {prefix:?}\n"));
            }
            config.push_str("[store.options]\n");
            for (key, value) in &store.options {
                let value = value.to_config_string();
                config.push_str(&format!("{key} = {value:?}\n"));
            }
        }
        std::fs::write(&path, config).unwrap();
        drop(socket);
        let log = self.dir.path().join("daemon.log");
        let child = std::process::Command::new(env!("CARGO_BIN_EXE_sokard"))
            .arg("--config")
            .arg(&path)
            .stdout(std::process::Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .unwrap();
        let client = Client::new(format!("http://{address}"), TOKEN).unwrap();
        let mut daemon = Daemon { client, child };
        for _ in 0..100 {
            match daemon
                .client
                .collection_by_scope(&CollectionScope::Named {
                    name: "startup-probe".into(),
                })
                .await
            {
                Ok(_) => return daemon,
                Err(e) if e.problem().is_some() => return daemon,
                Err(_) => {}
            }
            assert!(
                daemon.child.try_wait().unwrap().is_none(),
                "{}",
                std::fs::read_to_string(&log).unwrap()
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!(
            "daemon did not start: {}",
            std::fs::read_to_string(&log).unwrap()
        );
    }
}
