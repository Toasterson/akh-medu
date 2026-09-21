//! Offline operator helper: actual awakening -> SeaORM snapshot -> fresh-cache
//! recovery. No model, runtime, network provider, or forced re-awakening.
use akh_medu::{
    agent::{Agent, AgentConfig},
    bootstrap::{
        identity::{CharacterKnowledge, CultureOrigin, ritual_of_awakening},
        purpose::{DreyfusLevel, EntityType, PurposeModel},
    },
    engine::{Engine, EngineConfig},
};
use std::{path::PathBuf, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: roman_awaken DATABASE_URL INPUT_JSON WORK_DIRECTORY".into());
    }
    if !args[1].starts_with("sqlite:") {
        return Err("this helper requires local SQLite".into());
    }
    let root = PathBuf::from(&args[3]);
    let input: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let opts = akhomed_store::PoolOptions::new(&args[1]);
    akhomed_store::run_migrations_locked(&opts).await?;
    let db = akhomed_store::connect(&opts).await?;
    for (role, value) in input.as_object().ok_or("expected role map")? {
        if !["janus", "minerva", "vesta"].contains(&role.as_str()) {
            return Err("unexpected role".into());
        }
        let id = akh_medu::wal::akh_id_for(role);
        let workspace = akh_medu::wal::workspace_id_for(role);
        let existing = akhomed_store::snapshots::load_akh_state(&db, id).await?;
        let text = |key: &str| -> Result<String, Box<dyn std::error::Error>> {
            Ok(value[key].as_str().ok_or("missing string")?.to_string())
        };
        let strings = |key: &str| -> Result<Vec<String>, Box<dyn std::error::Error>> {
            Ok(serde_json::from_value(value[key].clone())?)
        };
        let character = CharacterKnowledge {
            name: text("name")?,
            description: text("description")?,
            entity_type: EntityType::Deity,
            culture: CultureOrigin::Latin,
            domains: strings("domains")?,
            traits: strings("traits")?,
            archetypes: strings("archetypes")?,
        };
        let purpose = PurposeModel {
            domain: character.domains[0].clone(),
            competence_level: DreyfusLevel::Expert,
            seed_concepts: character.domains.clone(),
            description: text("purpose")?,
        };
        akhomed_store::ensure_akh(&db, id, role, None).await?;
        let owner = akhomed_store::StablePgSession::connect(&opts).await?;
        if !owner.try_acquire_akh_lock(id).await? {
            return Err("akh already owned".into());
        }
        let (chosen_name, ritual_report) = if let Some(state) = existing {
            let name = state
                .psyche
                .as_ref()
                .and_then(|p| p["persona"]["name"].as_str())
                .ok_or("existing identity has no name; refusing re-awakening")?
                .to_string();
            (name, None)
        } else {
            let epoch = owner.claim_akh_residency(id, workspace).await?;
            let engine = Arc::new(Engine::new(EngineConfig {
                data_dir: Some(root.join(role).join("initial/kg")),
                compartments_dir: Some(root.join(role).join("initial/compartments")),
                ..Default::default()
            })?);
            let ritual = ritual_of_awakening(&character, &purpose, &engine, false)?;
            if ritual.provenance_ids.is_empty() {
                return Err("ritual failed to persist provenance".into());
            }
            let agent = Agent::new(engine.clone(), AgentConfig::default())?;
            let snapshot =
                akh_medu::snapshot::capture_snapshot_at_fence(&agent, id, workspace, 0, epoch)?;
            akh_medu::snapshot::persist_snapshot(snapshot, &db).await?;
            (
                ritual.chosen_name.clone(),
                Some(serde_json::to_value(&ritual)?),
            )
        };
        // A different, previously nonexistent cache proves relational recovery.
        let recovery_dir = root.join(role).join(format!(
            "recovered-{}/kg",
            akhomed_store::uuid::Uuid::new_v4()
        ));
        let recovered = akh_medu::snapshot::wake(&recovery_dir, id, workspace, &db).await?;
        let provenance = recovered.all_provenance()?;
        if !provenance.iter().any(|record| matches!(&record.kind,
            akh_medu::provenance::DerivationKind::RitualOfAwakening { chosen_name: name, .. } if name == &chosen_name)) {
            return Err("recovered awakening provenance missing".into());
        }
        let restored = Agent::resume(recovered, AgentConfig::default())?;
        let psyche = restored.psyche().ok_or("recovered psyche missing")?;
        if !psyche.is_awakened() || psyche.persona.name != chosen_name {
            return Err("recovered identity does not match awakening".into());
        }
        std::fs::create_dir_all(root.join("evidence"))?;
        std::fs::write(
            root.join("evidence").join(format!("{role}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({"role":role,"chosen_name":chosen_name,
                "awakened":psyche.is_awakened(),"provenance_count":provenance.len(),
                "recovery_dir":recovery_dir,"ritual":ritual_report,"recovery_verified":true}))?,
        )?;
        println!(
            "{role}: awakened as {}; SQLite snapshot and fresh-cache identity recovery verified",
            chosen_name
        );
        owner.release_akh_lock(id).await?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/sqlite_identity_roundtrip.rs"]
mod recovery_regression;
