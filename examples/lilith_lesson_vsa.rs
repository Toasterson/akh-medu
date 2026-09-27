//! Local structural lesson ranking using the existing VSA bind/bundle/similarity core.
use akh_medu::{
    simd::best_kernel,
    vsa::{Dimension, Encoding, HyperVec, ops::VsaOps},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{self, Read};

const DIM: usize = 8192;
const CODEBOOK: &str = "lilith-symbolic-lesson/1:sha256-counter-le:roles";

#[derive(Deserialize)]
struct Item {
    id: String,
    features: Vec<(String, String)>,
}
#[derive(Deserialize)]
struct Request {
    query: Vec<(String, String)>,
    items: Vec<Item>,
}

fn atom(label: &str) -> HyperVec {
    let mut bytes = Vec::with_capacity(DIM / 8);
    for counter in 0u32..32 {
        let mut hash = Sha256::new();
        hash.update(CODEBOOK.as_bytes());
        hash.update((label.len() as u64).to_le_bytes());
        hash.update(label.as_bytes());
        hash.update(counter.to_le_bytes());
        bytes.extend_from_slice(&hash.finalize());
    }
    HyperVec::from_raw(bytes, Dimension(DIM), Encoding::Bipolar)
}

fn encode(
    ops: &VsaOps,
    features: &[(String, String)],
) -> Result<HyperVec, Box<dyn std::error::Error>> {
    if features.is_empty() || features.len() > 64 {
        return Err("expected 1..64 structural features".into());
    }
    let unique: std::collections::BTreeSet<_> = features.iter().collect();
    let mut bound = Vec::new();
    for (role, value) in unique {
        if role.len() > 128 || value.len() > 256 {
            return Err("feature exceeds budget".into());
        }
        bound.push(ops.bind(
            &atom(&format!("role:{role}")),
            &atom(&format!("value:{value}")),
        )?);
    }
    Ok(ops.bundle(&bound.iter().collect::<Vec<_>>())?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin()
        .take(1024 * 1024 + 1)
        .read_to_string(&mut input)?;
    if input.len() > 1024 * 1024 {
        return Err("request exceeds 1 MiB".into());
    }
    let request: Request = serde_json::from_str(&input)?;
    if request.items.len() > 1024 {
        return Err("too many lessons".into());
    }
    let ops = VsaOps::new(best_kernel(), Dimension(DIM), Encoding::Bipolar);
    let query = encode(&ops, &request.query)?;
    let mut hits = Vec::new();
    for item in request.items {
        hits.push((
            item.id,
            ops.similarity(&query, &encode(&ops, &item.features)?)?,
        ));
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!(
        "{}",
        serde_json::json!({"space":{"encoder":CODEBOOK,"dimensions":DIM,
        "encoding":"bipolar-xor-majority","scope":"local symbolic features; no neural hidden-state alignment"},
        "hits":hits})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roles_distinguish_identical_bags_of_values_and_duplicates_do_not_reweight() {
        let ops = VsaOps::new(best_kernel(), Dimension(DIM), Encoding::Bipolar);
        let features = vec![
            ("operation".into(), "capacity".into()),
            ("policy".into(), "reject".into()),
            ("domain".into(), "integer".into()),
        ];
        let original = encode(&ops, &features).unwrap();
        let mut duplicate = features.clone();
        duplicate.reverse();
        duplicate.push(features[0].clone());
        assert_eq!(original, encode(&ops, &duplicate).unwrap());
        let swapped = vec![
            ("operation".into(), "reject".into()),
            ("policy".into(), "capacity".into()),
            ("domain".into(), "integer".into()),
        ];
        assert!(
            ops.similarity(&original, &encode(&ops, &swapped).unwrap())
                .unwrap()
                < 0.85
        );
    }
    #[test]
    fn empty_or_oversized_feature_sets_are_rejected() {
        let ops = VsaOps::new(best_kernel(), Dimension(DIM), Encoding::Bipolar);
        assert!(encode(&ops, &[]).is_err());
        assert!(encode(&ops, &vec![("r".into(), "v".into()); 65]).is_err());
    }
}
