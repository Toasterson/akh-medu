//! WASM tool runtime: load and execute WASM component-model tools.
//!
//! Feature-gated behind `wasm-tools`. Uses wasmtime's component model to
//! load `.wasm` files that implement the `akh:tool/akh-tool` world.
//!
//! The WIT interface is defined in `wit/akh-tool.wit`.

#[cfg(feature = "wasm-tools")]
pub use inner::*;

#[cfg(feature = "wasm-tools")]
mod inner {
    use std::collections::HashMap;
    use std::sync::Arc;

    use wasmtime::component::{Component, Linker, Val};
    use wasmtime::{Config, Engine as WasmEngine, Store};

    use crate::agent::error::{AgentError, AgentResult};
    use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
    use crate::agent::tool_manifest::{
        Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
    };
    use crate::engine::Engine;

    /// State shared with the WASM guest via host imports.
    pub struct ToolHostState {
        pub engine: Arc<Engine>,
        pub config: HashMap<String, String>,
    }

    /// Runtime for loading and managing WASM tool components.
    pub struct WasmToolRuntime {
        wasm_engine: WasmEngine,
    }

    impl WasmToolRuntime {
        /// Create a new WASM tool runtime.
        pub fn new() -> AgentResult<Self> {
            let mut config = Config::new();
            config.wasm_component_model(true);

            let wasm_engine = WasmEngine::new(&config).map_err(|e| AgentError::ToolExecution {
                tool_name: "wasm_runtime".into(),
                message: format!("failed to create WASM engine: {e}"),
            })?;

            Ok(Self { wasm_engine })
        }

        /// Load a WASM tool from a `.wasm` file.
        ///
        /// The component must export `manifest() -> tool-manifest` and
        /// `execute(input-json: string) -> tool-result`.
        ///
        /// This calls the guest's `manifest()` export at load time to populate
        /// the tool's metadata, then returns a `WasmTool` for the registry.
        pub fn load_tool(
            &self,
            wasm_path: &str,
            engine: Arc<Engine>,
            skill_id: String,
            config: HashMap<String, String>,
        ) -> AgentResult<WasmTool> {
            let wasm_bytes = std::fs::read(wasm_path).map_err(|e| AgentError::ToolExecution {
                tool_name: "wasm_runtime".into(),
                message: format!("failed to read WASM file {wasm_path}: {e}"),
            })?;

            let component = Component::new(&self.wasm_engine, &wasm_bytes).map_err(|e| {
                AgentError::ToolExecution {
                    tool_name: "wasm_runtime".into(),
                    message: format!("failed to compile WASM component {wasm_path}: {e}"),
                }
            })?;

            // Call manifest() to get tool metadata.
            let manifest_json = Self::call_manifest(
                &self.wasm_engine,
                &component,
                Arc::clone(&engine),
                config.clone(),
            )?;

            let manifest = parse_manifest_json(&manifest_json, &skill_id, wasm_path)?;

            Ok(WasmTool {
                manifest,
                wasm_engine: self.wasm_engine.clone(),
                component,
                engine,
                skill_id,
                config,
            })
        }

        /// Call the guest's `manifest()` export and return a JSON string.
        fn call_manifest(
            wasm_engine: &WasmEngine,
            component: &Component,
            engine: Arc<Engine>,
            config: HashMap<String, String>,
        ) -> AgentResult<String> {
            let state = ToolHostState { engine, config };
            let mut store = Store::new(wasm_engine, state);
            let linker = Linker::new(wasm_engine);

            let instance = linker
                .instantiate(&mut store, component)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "wasm_runtime".into(),
                    message: format!("failed to instantiate WASM component: {e}"),
                })?;

            let manifest_fn = instance
                .get_func(&mut store, "manifest")
                .ok_or_else(|| AgentError::ToolExecution {
                    tool_name: "wasm_runtime".into(),
                    message: "WASM component does not export 'manifest' function".into(),
                })?;

            let mut results = vec![Val::String(String::new())];
            manifest_fn
                .call(&mut store, &[], &mut results)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "wasm_runtime".into(),
                    message: format!("failed to call manifest(): {e}"),
                })?;

            match &results[0] {
                Val::String(s) => Ok(s.clone()),
                other => Err(AgentError::ToolExecution {
                    tool_name: "wasm_runtime".into(),
                    message: format!("manifest() returned unexpected type: {other:?}"),
                }),
            }
        }
    }

    /// Parse a manifest JSON string into our native types.
    fn parse_manifest_json(
        json: &str,
        skill_id: &str,
        wasm_path: &str,
    ) -> AgentResult<ToolManifest> {
        // We expect the WASM module to return JSON-encoded manifest.
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            AgentError::ToolExecution {
                tool_name: "wasm_runtime".into(),
                message: format!("failed to parse manifest JSON: {e}"),
            }
        })?;

        let name = v["name"].as_str().unwrap_or("unknown").to_string();
        let description = v["description"].as_str().unwrap_or("").to_string();

        let parameters: Vec<ToolParamSchema> = v["parameters"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|p| ToolParamSchema {
                        name: p["name"].as_str().unwrap_or("").to_string(),
                        description: p["description"].as_str().unwrap_or("").to_string(),
                        required: p["required"].as_bool().unwrap_or(false),
                        param_type: p["param_type"].as_str().unwrap_or("string").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let danger_level = match v["danger"]["level"].as_str() {
            Some("safe") => DangerLevel::Safe,
            Some("cautious") => DangerLevel::Cautious,
            Some("dangerous") => DangerLevel::Dangerous,
            Some("critical") => DangerLevel::Critical,
            _ => DangerLevel::Cautious,
        };

        let capabilities = v["danger"]["capabilities"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|c| match c.as_str()? {
                        "read-kg" => Some(Capability::ReadKg),
                        "write-kg" => Some(Capability::WriteKg),
                        "read-filesystem" => Some(Capability::ReadFilesystem),
                        "write-filesystem" => Some(Capability::WriteFilesystem),
                        "network" => Some(Capability::Network),
                        "process-exec" => Some(Capability::ProcessExec),
                        "user-interaction" => Some(Capability::UserInteraction),
                        "reason" => Some(Capability::Reason),
                        "vsa-access" => Some(Capability::VsaAccess),
                        "provenance-access" => Some(Capability::ProvenanceAccess),
                        "memory-access" => Some(Capability::MemoryAccess),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        let shadow_triggers = v["danger"]["shadow_triggers"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let danger_description = v["danger"]["description"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(ToolManifest {
            name,
            description,
            parameters,
            danger: DangerInfo {
                level: danger_level,
                capabilities,
                description: danger_description,
                shadow_triggers,
            },
            source: ToolSource::Wasm {
                skill_id: skill_id.to_string(),
                wasm_path: wasm_path.to_string(),
            },
        })
    }

    /// A tool loaded from a WASM component module.
    pub struct WasmTool {
        manifest: ToolManifest,
        wasm_engine: WasmEngine,
        component: Component,
        engine: Arc<Engine>,
        /// Skill that provided this tool (used for provenance tracking).
        pub(crate) skill_id: String,
        config: HashMap<String, String>,
    }

    impl Tool for WasmTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: self.manifest.name.clone(),
                description: self.manifest.description.clone(),
                parameters: self
                    .manifest
                    .parameters
                    .iter()
                    .map(|p| ToolParam {
                        name: p.name.clone(),
                        description: p.description.clone(),
                        required: p.required,
                    })
                    .collect(),
            }
        }

        fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
            let input_json = serde_json::to_string(&input.params).map_err(|e| {
                AgentError::ToolExecution {
                    tool_name: self.manifest.name.clone(),
                    message: format!("failed to serialize input: {e}"),
                }
            })?;

            // Each execute() creates a fresh Store for isolation.
            let state = ToolHostState {
                engine: Arc::clone(&self.engine),
                config: self.config.clone(),
            };
            let mut store = Store::new(&self.wasm_engine, state);
            let linker = Linker::new(&self.wasm_engine);

            let instance = linker
                .instantiate(&mut store, &self.component)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: self.manifest.name.clone(),
                    message: format!("failed to instantiate WASM component: {e}"),
                })?;

            let execute_fn = instance
                .get_func(&mut store, "execute")
                .ok_or_else(|| AgentError::ToolExecution {
                    tool_name: self.manifest.name.clone(),
                    message: "WASM component does not export 'execute' function".into(),
                })?;

            let mut results = vec![Val::String(String::new())];
            execute_fn
                .call(
                    &mut store,
                    &[Val::String(input_json)],
                    &mut results,
                )
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: self.manifest.name.clone(),
                    message: format!("WASM execute() trapped: {e}"),
                })?;

            // Parse the JSON result.
            let result_json = match &results[0] {
                Val::String(s) => s.clone(),
                _ => {
                    return Ok(ToolOutput::err("execute() returned unexpected type"));
                }
            };

            let v: serde_json::Value =
                serde_json::from_str(&result_json).unwrap_or(serde_json::json!({
                    "success": false,
                    "result_text": result_json,
                    "symbols_involved": []
                }));

            let success = v["success"].as_bool().unwrap_or(false);
            let result_text = v["result_text"].as_str().unwrap_or("").to_string();
            let symbols: Vec<crate::symbol::SymbolId> = v["symbols_involved"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|s| s.as_u64())
                        .filter_map(crate::symbol::SymbolId::new)
                        .collect()
                })
                .unwrap_or_default();

            if success {
                Ok(ToolOutput::ok_with_symbols(result_text, symbols))
            } else {
                Ok(ToolOutput {
                    success: false,
                    result: result_text,
                    symbols_involved: symbols,
                })
            }
        }

        fn manifest(&self) -> ToolManifest {
            self.manifest.clone()
        }
    }

    // WasmTool contains Arc<Engine> and Component, both of which are Send + Sync.
    // The Store is created fresh per execute() call, so no shared mutable state.
    unsafe impl Send for WasmTool {}
    unsafe impl Sync for WasmTool {}
}
