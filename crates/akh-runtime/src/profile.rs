//! Effective provider environment shared by runtime and account/config helpers.
use akh_session_types::ProviderConfig;
use std::collections::BTreeMap;

pub fn expand(value: &str, profile: &ProviderConfig) -> String {
    match &profile.isolation {
        Some(scope) => value.replace("{profile}", &scope.directory.to_string_lossy()),
        None => value.into(),
    }
}

pub fn environment(profile: &ProviderConfig) -> Result<BTreeMap<String, String>, String> {
    let Some(scope) = &profile.isolation else {
        return Ok(profile.environment.clone());
    };
    if !scope.directory.is_absolute() {
        return Err("runtime profile directory must be absolute".into());
    }
    let dir = &scope.directory;
    let mut env = BTreeMap::from([
        (
            "PATH".into(),
            "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin".into(),
        ),
        ("LANG".into(), "C.UTF-8".into()),
        (
            "HOME".into(),
            dir.join("home").to_string_lossy().into_owned(),
        ),
        (
            "XDG_CONFIG_HOME".into(),
            dir.join("config").to_string_lossy().into_owned(),
        ),
        (
            "XDG_CACHE_HOME".into(),
            dir.join("cache").to_string_lossy().into_owned(),
        ),
        (
            "CODEX_HOME".into(),
            dir.join("config/codex").to_string_lossy().into_owned(),
        ),
        (
            "CLAUDE_CONFIG_DIR".into(),
            dir.join("config/claude").to_string_lossy().into_owned(),
        ),
    ]);
    for name in profile
        .environment
        .keys()
        .chain(scope.environment_files.keys())
    {
        if matches!(
            name.as_str(),
            "HOME" | "XDG_CONFIG_HOME" | "XDG_CACHE_HOME" | "CODEX_HOME" | "CLAUDE_CONFIG_DIR"
        ) || name.is_empty()
            || name.contains(['=', '\0'])
        {
            return Err(
                "runtime environment attempts to override its private account/config boundary"
                    .into(),
            );
        }
    }
    for (name, value) in &profile.environment {
        env.insert(name.clone(), expand(value, profile));
    }
    for (name, path) in &scope.environment_files {
        if !path.is_absolute() || profile.environment.contains_key(name) {
            return Err("environment file requires an absolute path and a unique variable".into());
        }
        use std::io::Read;
        let mut value = String::new();
        std::fs::File::open(path)
            .and_then(|f| f.take(65537).read_to_string(&mut value))
            .map_err(|_| "runtime credential source could not be read")?;
        if value.len() > 65536 || value.trim().is_empty() || value.contains('\0') {
            return Err("runtime credential source is invalid".into());
        }
        env.insert(name.clone(), value.trim_end_matches(['\r', '\n']).into());
    }
    Ok(env)
}
