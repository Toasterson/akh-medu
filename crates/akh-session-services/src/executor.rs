use crate::{
    config::Config,
    store::{Store, key},
    types::*,
    workspace::{Workspace, digest},
};
use base64::Engine as _;
use serde_json::{Value, json};
use std::os::{fd::AsRawFd, unix::process::CommandExt};
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::Path,
    process::{ChildStdin, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const MAX_OUTPUT: usize = 1024 * 1024;

pub use akh_session_types::AttachWorkspace;

struct RunningProcess {
    cancel: AtomicBool,
    input: Mutex<Option<ChildStdin>>,
    stdout: Arc<Mutex<Captured>>,
    stderr: Arc<Mutex<Captured>>,
}

#[derive(Default)]
struct Captured {
    bytes: Vec<u8>,
    truncated: bool,
}

fn drain(
    mut pipe: impl Read + Send + 'static,
    output: Arc<Mutex<Captured>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0; 8192];
        while let Ok(n) = pipe.read(&mut chunk) {
            if n == 0 {
                break;
            }
            let mut output = output.lock().unwrap_or_else(|e| e.into_inner());
            let take = n.min(MAX_OUTPUT.saturating_sub(output.bytes.len()));
            output.bytes.extend_from_slice(&chunk[..take]);
            output.truncated |= take < n;
        }
    })
}

#[derive(Clone)]
pub struct Executor {
    pub store: Store,
    pub config: Arc<Config>,
    root: Arc<Workspace>,
    processes: Arc<Mutex<HashMap<String, Arc<RunningProcess>>>>,
    pub(crate) mutations: Arc<Mutex<()>>,
}

fn receipt_key(id: &str) -> String {
    key("receipt", &[id])
}
fn argument<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| Fault::invalid(format!("{name} must be a string")))
}

impl Executor {
    pub fn renew_lease(&self, id: &str, request: &RenewLease) -> Result<WorkspaceLease> {
        let _lock = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
        self.store.transaction(|tx| {
            let k = key("lease", &[id]);
            let mut lease: WorkspaceLease = tx
                .get(&k)?
                .ok_or_else(|| Fault::missing("workspace lease not found"))?;
            if lease.session_id != request.session_id || lease.generation != request.generation {
                return Err(Fault::new(
                    "forbidden",
                    "workspace lease owner/generation mismatch",
                ));
            }
            if lease.expires_at <= now_ms()
                || tx.get::<bool>(&key("released", &[id]))?.unwrap_or(false)
            {
                return Err(Fault::new(
                    "expired",
                    "workspace lease expired or was released; no implicit reacquisition",
                ));
            }
            lease.expires_at = now_ms() + 300_000;
            tx.put(&k, &lease)?;
            Ok(lease)
        })
    }
    pub fn valid_lease(&self, id: &str) -> Result<()> {
        let lease: WorkspaceLease = self.store.get(&key("lease", &[id]))?.ok_or_else(|| {
            Fault::new(
                "expired",
                "workspace has no lease; explicit recovery is required",
            )
        })?;
        if lease.expires_at <= now_ms() {
            return Err(Fault::new(
                "expired",
                "workspace lease expired; effects are fenced",
            ));
        }
        Ok(())
    }
    pub fn expire_leases(&self) -> Result<()> {
        for lease in self.store.scan::<WorkspaceLease>("lease/")? {
            if lease.expires_at > now_ms() {
                continue;
            }
            for (id, process) in self
                .processes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
            {
                if self.receipt(id)?.request.workspace_id == lease.workspace_id {
                    process.cancel.store(true, Ordering::SeqCst);
                }
            }
            // stop_services is idempotent and reports busy until every helper stopped.
            if let Err(error) = self.stop_services(&lease.workspace_id) {
                if error.code != "busy" {
                    tracing::warn!(workspace=%lease.workspace_id,code=%error.code,"expired workspace helper requires inspection");
                }
            }
        }
        Ok(())
    }
    pub fn reconcile(
        &self,
        p: &crate::config::Principal,
        id: &str,
        resolution: &Reconciliation,
    ) -> Result<OperationReceipt> {
        crate::http::role(p, "admin")?;
        let _mutations = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
        let processes = self.processes.lock().unwrap_or_else(|e| e.into_inner());
        self.store.transaction(|tx| {
            let target = receipt_key(id);
            let mut receipt: OperationReceipt = tx
                .get(&target)?
                .ok_or_else(|| Fault::missing("receipt not found"))?;
            crate::recovery::validate(p, &receipt.request, resolution)?;
            if crate::recovery::already(tx, p, &target, resolution)? {
                return Ok(receipt);
            }
            for active in processes.keys() {
                if tx
                    .get::<OperationReceipt>(&receipt_key(active))?
                    .is_some_and(|r| r.request.workspace_id == receipt.request.workspace_id)
                {
                    return Err(Fault::new(
                        "busy",
                        "workspace still has active executor processes",
                    ));
                }
            }
            if receipt.status != TurnStatus::Unknown {
                return Err(Fault::conflict(
                    "only an unknown operation can be reconciled",
                ));
            }
            let mut result = resolution.result.clone().ok_or_else(|| {
                Fault::invalid("operation reconciliation requires its inspected result")
            })?;
            if result.success != (resolution.status == TurnStatus::Completed)
                || resolution.output.is_some()
            {
                return Err(Fault::invalid(
                    "receipt success must match the resolved status",
                ));
            }
            result.receipt = Some(receipt.id.clone());
            receipt.status = resolution.status;
            receipt.result = Some(result);
            receipt.finished_at = Some(now_ms());
            tx.put(&target, &receipt)?;
            crate::recovery::record(tx, p, &target, resolution)?;
            Ok(receipt)
        })
    }
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        let path = config
            .workspace_root
            .as_ref()
            .ok_or_else(|| Fault::invalid("executor requires workspace_root"))?;
        if !path.is_absolute() {
            return Err(Fault::invalid("workspace_root must be absolute"));
        }
        std::fs::create_dir_all(path).map_err(|e| Fault::new("workspace_io", e.to_string()))?;
        let root = Arc::new(Workspace::open(path)?);
        store.transaction(|tx|{
            for mut receipt in tx.scan::<OperationReceipt>("receipt/")?{
                if !receipt.status.terminal(){
                    receipt.status=if matches!(receipt.request.operation.as_str(),"read"|"list"|"search"|"exec_poll"){TurnStatus::Failed}else{TurnStatus::Unknown};
                    receipt.result=Some(ToolResult{success:false,text:"Executor restarted before a durable terminal receipt. Reconcile the operation before further mutations.".into(),data:Some(json!({"code":"unknown_outcome"})),receipt:Some(receipt.id.clone())});
                    tx.put(&receipt_key(&receipt.id),&receipt)?;
                }
            }Ok(())
        })?;
        Ok(Self {
            store,
            config,
            root,
            processes: Default::default(),
            mutations: Default::default(),
        })
    }
    pub fn attach(&self, request: &AttachWorkspace) -> Result<WorkspaceBinding> {
        validate_key(&request.session_id)?;
        validate_key(&request.binding.id)?;
        if request.binding.executor != self.config.service_id || request.binding.generation == 0 {
            return Err(Fault::invalid("executor identity/generation mismatch"));
        }
        let relative = Path::new(&request.binding.root)
            .strip_prefix(self.config.workspace_root.as_ref().unwrap())
            .or_else(|_| Path::new(&request.binding.root).strip_prefix(&self.root.path))
            .map_err(|_| Fault::new("forbidden", "workspace root is outside this executor"))?;
        let relative = relative
            .to_str()
            .ok_or_else(|| Fault::invalid("workspace root must be UTF-8"))?;
        let _lock = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
        self.store.transaction(|tx| {
            let k = key("workspace", &[&request.binding.id]);
            if tx.get::<bool>(&key("released", &[&request.binding.id]))? == Some(true) {
                return Err(Fault::conflict(
                    "workspace release fences delayed attachment",
                ));
            }
            if let Some(old) = tx.get::<AttachWorkspace>(&k)? {
                let mut expected = request.clone();
                expected.binding.root =
                    self.root.path.join(relative).to_string_lossy().into_owned();
                expected.binding.platform = old.binding.platform.clone();
                expected.binding.confinement = old.binding.confinement.clone();
                if old != expected {
                    return Err(Fault::conflict(
                        "workspace already has a different owner or binding",
                    ));
                }
                return Ok(old.binding);
            }
            for old in tx.scan::<AttachWorkspace>("workspace/")? {
                let old_path = Path::new(&old.binding.root);
                let normalized = self.root.path.join(relative);
                let new_path = normalized.as_path();
                if old_path.starts_with(new_path) || new_path.starts_with(old_path) {
                    return Err(Fault::conflict("workspace overlaps an existing allocation"));
                }
            }
            self.root.mkdir(relative)?;
            self.root.directory(relative)?;
            let mut binding = request.binding.clone();
            binding.root = self.root.path.join(relative).to_string_lossy().into_owned();
            binding.platform = std::env::consts::OS.into();
            binding.confinement = self.config.confinement.clone();
            tx.put(
                &k,
                &AttachWorkspace {
                    session_id: request.session_id.clone(),
                    binding: binding.clone(),
                },
            )?;
            tx.put(
                &key("lease", &[&binding.id]),
                &WorkspaceLease {
                    session_id: request.session_id.clone(),
                    workspace_id: binding.id.clone(),
                    generation: binding.generation,
                    expires_at: now_ms() + 300_000,
                },
            )?;
            Ok(binding)
        })
    }
    fn workspace(&self, request: &WorkspaceOperation) -> Result<Workspace> {
        if !matches!(request.operation.as_str(), "exec_poll" | "exec_cancel") {
            self.valid_lease(&request.workspace_id)?;
        }
        if self
            .store
            .get::<bool>(&key("released", &[&request.workspace_id]))?
            .unwrap_or(false)
        {
            return Err(Fault::conflict("workspace lease has been released"));
        }
        let allocation: AttachWorkspace = self
            .store
            .get(&key("workspace", &[&request.workspace_id]))?
            .ok_or_else(|| Fault::missing("workspace not attached"))?;
        if allocation.session_id != request.session_id
            || allocation.binding.generation != request.generation
        {
            return Err(Fault::new(
                "forbidden",
                "workspace session or generation mismatch",
            ));
        }
        let path = Path::new(&allocation.binding.root);
        let relative = path
            .strip_prefix(&self.root.path)
            .map_err(|_| Fault::new("forbidden", "stored workspace escaped executor root"))?;
        let relative = relative
            .to_str()
            .ok_or_else(|| Fault::invalid("workspace path is not UTF-8"))?;
        Ok(Workspace::from_directory(
            self.root.directory(relative)?,
            path.into(),
        ))
    }
    pub fn receipt(&self, id: &str) -> Result<OperationReceipt> {
        self.store
            .get(&receipt_key(id))?
            .ok_or_else(|| Fault::missing("operation not found"))
    }

    pub fn release(&self, id: &str, session: &str, generation: u64) -> Result<serde_json::Value> {
        validate_key(id)?;
        validate_key(session)?;
        if generation == 0 {
            return Err(Fault::invalid("workspace generation must be positive"));
        }
        {
            let _lock = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
            self.store.transaction(|tx| {
                if let Some(allocation) = tx.get::<AttachWorkspace>(&key("workspace", &[id]))? {
                    if allocation.session_id != session
                        || allocation.binding.generation != generation
                    {
                        return Err(Fault::new("forbidden", "workspace binding mismatch"));
                    }
                }
                let intent_key = key("workspace_release", &[id]);
                if let Some(old) = tx.get::<RenewLease>(&intent_key)? {
                    if old.session_id != session || old.generation != generation {
                        return Err(Fault::conflict(
                            "workspace already revoked by another binding",
                        ));
                    }
                }
                if tx
                    .scan::<OperationReceipt>("receipt/")?
                    .iter()
                    .any(|r| r.request.workspace_id == id && !r.status.terminal())
                {
                    return Err(Fault::conflict(
                        "workspace still has running or unresolved operations",
                    ));
                }
                // Fence new tools/helpers AND a delayed attach before waiting for
                // helper shutdown. A missing binding is not proof no attach was sent.
                tx.put(
                    &intent_key,
                    &RenewLease {
                        session_id: session.into(),
                        generation,
                    },
                )?;
                tx.put(&key("released", &[id]), &true)?;
                if let Some(mut lease) = tx.get::<WorkspaceLease>(&key("lease", &[id]))? {
                    lease.expires_at = now_ms();
                    tx.put(&key("lease", &[id]), &lease)?;
                }
                Ok(())
            })?;
        }
        self.stop_services(id)?;
        Ok(json!({"released":true,"files_retained":true}))
    }

    pub fn operate(&self, request: WorkspaceOperation) -> Result<OperationReceipt> {
        validate_key(&request.id)?;
        if !request.arguments.is_object() {
            return Err(Fault::invalid("tool arguments must be an object"));
        }
        let _lock = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
        let ws = self.workspace(&request)?;
        let is_write = !matches!(
            request.operation.as_str(),
            "read" | "list" | "search" | "exec_poll"
        );
        let (receipt, fresh) = self.store.transaction(|tx| {
            if let Some(old) = tx.get::<OperationReceipt>(&receipt_key(&request.id))? {
                if old.request != request {
                    return Err(Fault::conflict(
                        "operation id reused with different arguments",
                    ));
                }
                return Ok((old, false));
            }
            if is_write
                && tx.scan::<OperationReceipt>("receipt/")?.iter().any(|r| {
                    r.request.workspace_id == request.workspace_id
                        && r.status == TurnStatus::Unknown
                })
            {
                return Err(Fault::conflict(
                    "workspace has an unresolved operation; mutation admission is fenced",
                ));
            }
            let receipt = OperationReceipt {
                id: request.id.clone(),
                request: request.clone(),
                status: TurnStatus::Running,
                result: None,
                started_at: now_ms(),
                finished_at: None,
            };
            tx.put(&receipt_key(&request.id), &receipt)?;
            Ok((receipt, true))
        })?;
        if !fresh {
            return Ok(receipt);
        }
        if request.operation == "exec_start" {
            match self.start(&request, &ws) {
                Ok(()) => return self.receipt(&request.id),
                Err(error) => return self.finish(receipt, Err(error)),
            }
        }
        let result = self.perform(&request, &ws);
        self.finish(receipt, result)
    }
    fn finish(
        &self,
        mut receipt: OperationReceipt,
        result: Result<Value>,
    ) -> Result<OperationReceipt> {
        receipt.finished_at = Some(now_ms());
        receipt.status = if result.is_ok() {
            TurnStatus::Completed
        } else {
            TurnStatus::Failed
        };
        receipt.result = Some(match result {
            Ok(data) => ToolResult {
                success: true,
                text: serde_json::to_string(&data).unwrap_or_default(),
                data: Some(data),
                receipt: Some(receipt.id.clone()),
            },
            Err(error) => ToolResult {
                success: false,
                text: error.to_string(),
                data: Some(json!({"error":error})),
                receipt: Some(receipt.id.clone()),
            },
        });
        self.store
            .transaction(|tx| tx.put(&receipt_key(&receipt.id), &receipt))?;
        Ok(receipt)
    }
    fn perform(&self, request: &WorkspaceOperation, ws: &Workspace) -> Result<Value> {
        let args = &request.arguments;
        match request.operation.as_str() {
            "read" => {
                let bytes = ws.read(argument(args, "path")?)?;
                let revision = digest(&bytes);
                if args["encoding"] == "base64" {
                    Ok(
                        json!({"content":base64::engine::general_purpose::STANDARD.encode(bytes),"encoding":"base64","revision":revision}),
                    )
                } else {
                    let content = String::from_utf8(bytes)
                        .map_err(|_| Fault::invalid("binary file requires encoding=base64"))?;
                    Ok(json!({"content":content,"encoding":"utf8","revision":revision}))
                }
            }
            "write" => {
                let content = argument(args, "content")?;
                let bytes = if args["encoding"] == "base64" {
                    base64::engine::general_purpose::STANDARD
                        .decode(content)
                        .map_err(|_| Fault::invalid("invalid base64"))?
                } else {
                    content.as_bytes().to_vec()
                };
                let revision = ws.write(
                    argument(args, "path")?,
                    &bytes,
                    args["expected_revision"].as_str(),
                    args["create"].as_bool().unwrap_or(false),
                )?;
                Ok(json!({"revision":revision,"bytes":bytes.len()}))
            }
            "edit" => {
                let path = argument(args, "path")?;
                let bytes = ws.read(path)?;
                let revision = digest(&bytes);
                if args["expected_revision"].as_str() != Some(&revision) {
                    return Err(Fault::conflict("edit requires current expected_revision"));
                }
                let content = String::from_utf8(bytes)
                    .map_err(|_| Fault::invalid("edit requires UTF-8 text"))?;
                let old = argument(args, "old_text")?;
                let new = argument(args, "new_text")?;
                if old.is_empty() || content.matches(old).count() != 1 {
                    return Err(Fault::conflict("edit requires exactly one old_text match"));
                }
                let changed = content.replacen(old, new, 1);
                let revision = ws.write(path, changed.as_bytes(), Some(&revision), false)?;
                Ok(json!({"revision":revision}))
            }
            "mkdir" => {
                ws.mkdir(argument(args, "path")?)?;
                Ok(json!({"created":true}))
            }
            "list" => Ok(json!({"entries":ws.list(args["path"].as_str().unwrap_or("."))?})),
            "search" => {
                let regex = regex::RegexBuilder::new(argument(args, "pattern")?)
                    .size_limit(1024 * 1024)
                    .build()
                    .map_err(|e| Fault::invalid(e.to_string()))?;
                let glob = globset::Glob::new(args["glob"].as_str().unwrap_or("**"))
                    .map_err(|e| Fault::invalid(e.to_string()))?
                    .compile_matcher();
                let mut pending = vec![args["path"].as_str().unwrap_or(".").to_owned()];
                let mut found = Vec::new();
                let mut examined = 0;
                while let Some(dir) = pending.pop() {
                    for name in ws.list(&dir)? {
                        examined += 1;
                        if examined > 10000 || found.len() >= 1000 {
                            return Ok(json!({"matches":found,"truncated":true}));
                        }
                        if name == ".git" {
                            continue;
                        }
                        let path = if dir == "." {
                            name
                        } else {
                            format!("{dir}/{name}")
                        };
                        if ws.directory(&path).is_ok() {
                            pending.push(path);
                        } else if glob.is_match(&path) {
                            if let Ok(bytes) = ws.read(&path) {
                                if let Ok(text) = String::from_utf8(bytes) {
                                    for (line, text) in text.lines().enumerate() {
                                        if regex.is_match(text) {
                                            found.push(json!({"path":path,"line":line+1,"text":text.chars().take(2000).collect::<String>()}));
                                            if found.len() >= 1000 {
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(json!({"matches":found,"truncated":false}))
            }
            "exec_poll" => {
                let receipt = self.process_receipt(request)?;
                let running = self
                    .processes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&receipt.id)
                    .cloned();
                let progress=running.map(|process|{
                    let out=process.stdout.lock().unwrap_or_else(|e|e.into_inner());let err=process.stderr.lock().unwrap_or_else(|e|e.into_inner());
                    json!({"stdout":String::from_utf8_lossy(&out.bytes),"stderr":String::from_utf8_lossy(&err.bytes),"truncated":out.truncated||err.truncated})
                });
                Ok(
                    json!({"process_id":receipt.id,"status":receipt.status,"result":receipt.result,"progress":progress}),
                )
            }
            "exec_stdin" => {
                let receipt = self.process_receipt(request)?;
                let process = self
                    .processes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&receipt.id)
                    .cloned()
                    .ok_or_else(|| Fault::conflict("process is no longer running"))?;
                let mut input = process.input.lock().unwrap_or_else(|e| e.into_inner());
                let stdin = input
                    .as_mut()
                    .ok_or_else(|| Fault::conflict("process stdin is closed"))?;
                stdin
                    .write_all(argument(args, "text")?.as_bytes())
                    .and_then(|_| stdin.flush())
                    .map_err(|e| Fault::new("process_io", e.to_string()))?;
                if args["close"].as_bool() == Some(true) {
                    input.take();
                }
                Ok(json!({"sent":true}))
            }
            "exec_cancel" => {
                let receipt = self.process_receipt(request)?;
                if let Some(process) = self
                    .processes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&receipt.id)
                {
                    process.cancel.store(true, Ordering::SeqCst);
                } else if !receipt.status.terminal() {
                    return Err(Fault::conflict(
                        "process outcome is unknown; cancellation cannot be confirmed",
                    ));
                }
                Ok(json!({"cancel_requested":true,"terminal":receipt.status.terminal()}))
            }
            _ => Err(Fault::new("unsupported", "unknown workspace operation")),
        }
    }
    fn process_receipt(&self, request: &WorkspaceOperation) -> Result<OperationReceipt> {
        let receipt = self.receipt(argument(&request.arguments, "process_id")?)?;
        if receipt.request.operation != "exec_start"
            || receipt.request.session_id != request.session_id
            || receipt.request.workspace_id != request.workspace_id
            || receipt.request.generation != request.generation
        {
            return Err(Fault::new(
                "forbidden",
                "process belongs to another binding",
            ));
        }
        Ok(receipt)
    }
    fn start(&self, request: &WorkspaceOperation, ws: &Workspace) -> Result<()> {
        let args = &request.arguments;
        if args["tty"].as_bool() == Some(true) {
            return Err(Fault::new(
                "unsupported",
                "executor process API currently supports pipes, not PTYs",
            ));
        }
        let timeout = args["timeout_ms"].as_u64().unwrap_or(600_000);
        if timeout == 0 || timeout > 3_600_000 {
            return Err(Fault::invalid("timeout_ms must be 1..3600000"));
        }
        let cwd = ws.directory(args["cwd"].as_str().unwrap_or("."))?;
        // Human/model shell commands retain their existing semantics. Trusted
        // bootstrap controllers can instead pass literal arguments without any
        // shell expansion, using an explicit executable installed in the image.
        let mut command = match (args.get("command"), args.get("argv")) {
            (Some(_), None) => {
                let mut command = Command::new("/bin/sh");
                command.args(["-c", argument(args, "command")?]);
                command
            }
            (None, Some(value)) => {
                let values = value
                    .as_array()
                    .ok_or_else(|| Fault::invalid("argv must be an array"))?;
                if values.is_empty() || values.len() > 1024 {
                    return Err(Fault::invalid("argv requires 1..1024 literal arguments"));
                }
                let arguments: Vec<&str> = values
                    .iter()
                    .map(|value| {
                        value.as_str().filter(|s| !s.contains('\0')).ok_or_else(|| {
                            Fault::invalid("argv values must be strings without NUL")
                        })
                    })
                    .collect::<Result<_>>()?;
                if !Path::new(arguments[0]).is_absolute()
                    || arguments.iter().map(|s| s.len()).sum::<usize>() > 128 * 1024
                {
                    return Err(Fault::invalid(
                        "argv requires an absolute executable and at most 128 KiB",
                    ));
                }
                let mut command = Command::new(arguments[0]);
                command.args(&arguments[1..]);
                command
            }
            _ => return Err(Fault::invalid("provide exactly one of command or argv")),
        };
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        command
            .env_clear()
            .env("PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
            .env("LANG", "C.UTF-8");
        if let Some(env) = args["env"].as_object() {
            for (name, value) in env {
                if name.is_empty() || name.contains(['=', '\0']) {
                    return Err(Fault::invalid("invalid environment variable name"));
                }
                command.env(
                    name,
                    value
                        .as_str()
                        .ok_or_else(|| Fault::invalid("environment values must be strings"))?,
                );
            }
        }
        unsafe {
            command.pre_exec(move || {
                if libc::fchdir(cwd.as_raw_fd()) < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        let mut child = command
            .spawn()
            .map_err(|e| Fault::new("process_spawn", e.to_string()))?;
        let stdout = Arc::new(Mutex::new(Captured::default()));
        let stderr = Arc::new(Mutex::new(Captured::default()));
        let process = Arc::new(RunningProcess {
            cancel: AtomicBool::new(false),
            input: Mutex::new(child.stdin.take()),
            stdout: stdout.clone(),
            stderr: stderr.clone(),
        });
        let out_thread = drain(child.stdout.take().unwrap(), stdout.clone());
        let err_thread = drain(child.stderr.take().unwrap(), stderr.clone());
        self.processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request.id.clone(), process.clone());
        let store = self.store.clone();
        let processes = self.processes.clone();
        let id = request.id.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_millis(timeout);
            let mut cancelled = false;
            let mut timed_out = false;
            let status = loop {
                if process.cancel.load(Ordering::SeqCst) || Instant::now() >= deadline {
                    cancelled = process.cancel.load(Ordering::SeqCst);
                    timed_out = !cancelled;
                    unsafe {
                        libc::kill(-(child.id() as i32), libc::SIGKILL);
                    }
                    let _ = child.kill();
                    break child.wait();
                }
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                    Err(e) => break Err(e),
                }
            };
            // Stop descendants that kept inherited pipes open after the shell exited.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            process
                .input
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            let _ = out_thread.join();
            let _ = err_thread.join();
            let out = stdout.lock().unwrap_or_else(|e| e.into_inner());
            let err = stderr.lock().unwrap_or_else(|e| e.into_inner());
            let code = status.as_ref().ok().and_then(|s| s.code());
            let terminal = if status.is_err() {
                TurnStatus::Unknown
            } else if cancelled {
                TurnStatus::Cancelled
            } else {
                TurnStatus::Completed
            };
            let data = json!({"stdout":String::from_utf8_lossy(&out.bytes),"stderr":String::from_utf8_lossy(&err.bytes),"exit_code":code,"timed_out":timed_out,"cancelled":cancelled,"truncated":out.truncated||err.truncated});
            let result = store.transaction(|tx| {
                let mut receipt: OperationReceipt = tx
                    .get(&receipt_key(&id))?
                    .ok_or_else(|| Fault::missing("process receipt disappeared"))?;
                receipt.status = terminal;
                receipt.finished_at = Some(now_ms());
                receipt.result = Some(ToolResult {
                    success: code == Some(0) && !cancelled && !timed_out,
                    text: data.to_string(),
                    data: Some(data),
                    receipt: Some(id.clone()),
                });
                tx.put(&receipt_key(&id), &receipt)
            });
            if let Err(error) = result {
                tracing::error!(operation=%id,code=%error.code,"could not persist process receipt");
            }
            processes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&id);
        });
        Ok(())
    }
    pub fn cancel_all(&self) {
        for process in self
            .processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            process.cancel.store(true, Ordering::SeqCst);
        }
    }
}
