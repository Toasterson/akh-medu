//! Validate a frozen OCR corpus before it can participate in training or a
//! promotion audit. Evidence is printed even on failure.

use grug_data::{
    SiblingSeparationPolicy, VisionCorpusDomain, VisionSeparationReport, VisionTextManifest,
    audit_vision_sibling_separation, vision_perceptual_hash, vision_pixel_hash,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Serialize)]
struct ValidationIssue {
    #[serde(skip_serializing_if = "Option::is_none")]
    case_id: Option<String>,
    kind: &'static str,
    detail: String,
}

#[derive(Debug, Serialize)]
struct CorpusValidationReport {
    schema_version: u32,
    valid: bool,
    corpus: String,
    manifest_hash: String,
    cases: usize,
    semantic_items: usize,
    required_domain: String,
    minimum_cases_per_language_layout_corruption: usize,
    minimum_observed_cell_cases: usize,
    renderers: BTreeSet<String>,
    source_hashes: BTreeSet<String>,
    per_language: BTreeMap<String, usize>,
    per_layout: BTreeMap<String, usize>,
    per_corruption: BTreeMap<String, usize>,
    per_language_layout_corruption: BTreeMap<String, usize>,
    separation_from_training: Option<VisionSeparationReport>,
    issues: Vec<ValidationIssue>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(required(&args, "--corpus")?);
    let minimum_cell = optional(&args, "--min-per-cell")?.unwrap_or(10usize);
    let required_domain =
        optional_string(&args, "--require-domain")?.unwrap_or_else(|| "audit".to_string());
    let required_domain_value = parse_domain(&required_domain)?;
    let manifest_path = corpus.join("manifest.jsonl");
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest_hash = format!("blake3:{}", blake3::hash(&manifest_bytes).to_hex());
    let cases = read_manifest(&manifest_path)?;
    if cases.is_empty() {
        return Err("vision manifest contains no cases".into());
    }

    let mut issues = Vec::new();
    validate_completion_metadata(&corpus, &manifest_hash, cases.len(), &mut issues);
    let mut case_ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut semantic_variants = BTreeSet::new();
    let mut semantic_ids = BTreeSet::new();
    let mut pixel_hashes = BTreeMap::<String, String>::new();
    let mut renderers = BTreeSet::new();
    let mut source_hashes = BTreeSet::new();
    let mut per_language = BTreeMap::new();
    let mut per_layout = BTreeMap::new();
    let mut per_corruption = BTreeMap::new();
    let mut cells = BTreeMap::new();

    for case in &cases {
        if case.domain != required_domain_value {
            issue(
                &mut issues,
                Some(case),
                "wrong_domain",
                format!(
                    "found {:?}, required {:?}",
                    case.domain, required_domain_value
                ),
            );
        }
        if !case_ids.insert(case.case_id.clone()) {
            issue(
                &mut issues,
                Some(case),
                "duplicate_case_id",
                case.case_id.clone(),
            );
        }
        if !semantic_variants.insert((case.semantic_id.clone(), case.variant)) {
            issue(
                &mut issues,
                Some(case),
                "duplicate_semantic_variant",
                format!("{} variant {}", case.semantic_id, case.variant),
            );
        }
        semantic_ids.insert(case.semantic_id.clone());
        renderers.insert(case.renderer_id.clone());
        let Some(provenance) = case.provenance.as_ref() else {
            issue(
                &mut issues,
                Some(case),
                "missing_provenance",
                "audit and newly generated corpora require source identity".into(),
            );
            continue;
        };
        if provenance.source_id.trim().is_empty()
            || provenance.source_revision.trim().is_empty()
            || provenance.source_license.trim().is_empty()
            || !provenance.source_hash.starts_with("blake3:")
        {
            issue(
                &mut issues,
                Some(case),
                "invalid_provenance",
                format!("{:?}", provenance),
            );
        }
        source_hashes.insert(provenance.source_hash.clone());
        if required_domain_value == VisionCorpusDomain::Audit
            && case.renderer_id == grug_data::vision_text::VISION_TEXT_RENDERER_ID
        {
            issue(
                &mut issues,
                Some(case),
                "non_independent_renderer",
                "audit case uses the training/rolling-exam bitmap renderer".into(),
            );
        }

        let relative = Path::new(&case.image_path);
        if !safe_relative_path(relative) {
            issue(
                &mut issues,
                Some(case),
                "unsafe_image_path",
                case.image_path.clone(),
            );
            continue;
        }
        if !paths.insert(case.image_path.clone()) {
            issue(
                &mut issues,
                Some(case),
                "duplicate_image_path",
                case.image_path.clone(),
            );
        }
        match read_rgb_png(&corpus.join(relative)) {
            Ok((width, height, rgb)) => {
                if width != case.width || height != case.height || case.channels != 3 {
                    issue(
                        &mut issues,
                        Some(case),
                        "image_shape_mismatch",
                        format!(
                            "decoded {width}x{height}x3, manifest {}x{}x{}",
                            case.width, case.height, case.channels
                        ),
                    );
                }
                let fnv_hash = vision_pixel_hash(&rgb);
                if fnv_hash != case.pixel_hash_fnv1a64 {
                    issue(
                        &mut issues,
                        Some(case),
                        "pixel_hash_mismatch",
                        format!("decoded {fnv_hash}, manifest {}", case.pixel_hash_fnv1a64),
                    );
                }
                let blake_hash = format!("blake3:{}", blake3::hash(&rgb).to_hex());
                if case.pixel_hash_blake3.as_deref() != Some(blake_hash.as_str()) {
                    issue(
                        &mut issues,
                        Some(case),
                        "cryptographic_pixel_hash_mismatch",
                        format!(
                            "decoded {blake_hash}, manifest {:?}",
                            case.pixel_hash_blake3
                        ),
                    );
                }
                let perceptual = vision_perceptual_hash(&rgb, width, height);
                if perceptual != case.perceptual_hash {
                    issue(
                        &mut issues,
                        Some(case),
                        "perceptual_hash_mismatch",
                        format!("decoded {perceptual}, manifest {}", case.perceptual_hash),
                    );
                }
                if let Some(first_case) = pixel_hashes.insert(blake_hash, case.case_id.clone()) {
                    issue(
                        &mut issues,
                        Some(case),
                        "duplicate_pixels",
                        format!("same decoded pixels as {first_case}"),
                    );
                }
            }
            Err(error) => issue(
                &mut issues,
                Some(case),
                "image_read_failed",
                error.to_string(),
            ),
        }

        let language = language_label(case);
        let layout = layout_label(case);
        let corruption = corruption_label(case);
        *per_language.entry(language.clone()).or_default() += 1;
        *per_layout.entry(layout.clone()).or_default() += 1;
        *per_corruption.entry(corruption.clone()).or_default() += 1;
        *cells
            .entry(format!("{language}/{layout}/{corruption}"))
            .or_default() += 1;
    }

    let required_languages = ["en", "de", "fr"];
    let required_layouts = ["line", "paragraph", "receipt", "terminal", "api_error"];
    let required_corruptions = [
        "clean",
        "low_contrast",
        "salt_pepper",
        "box_blur",
        "occlusion",
        "uneven_lighting",
        "horizontal_shear",
    ];
    let mut minimum_observed = usize::MAX;
    for language in required_languages {
        for layout in required_layouts {
            for corruption in required_corruptions {
                let key = format!("{language}/{layout}/{corruption}");
                let observed = cells.get(&key).copied().unwrap_or(0);
                minimum_observed = minimum_observed.min(observed);
                if observed < minimum_cell {
                    issue(
                        &mut issues,
                        None,
                        "under_sampled_slice",
                        format!("{key} has {observed}, requires {minimum_cell}"),
                    );
                }
            }
        }
    }
    if minimum_observed == usize::MAX {
        minimum_observed = 0;
    }

    let separation_from_training = if let Some(path) =
        optional_string(&args, "--training-manifest")?
    {
        let training = read_manifest(Path::new(&path))?;
        let report =
            audit_vision_sibling_separation(&cases, &training, SiblingSeparationPolicy::default());
        if !report.admitted {
            issue(
                &mut issues,
                None,
                "training_separation_rejected",
                report
                    .rejection
                    .clone()
                    .unwrap_or_else(|| "separation policy rejected corpus".into()),
            );
        }
        if required_domain_value == VisionCorpusDomain::Audit
            && !report
                .evidence
                .held_out_axes
                .iter()
                .any(|axis| axis == "renderer")
        {
            issue(
                &mut issues,
                None,
                "renderer_not_held_out",
                "audit renderers overlap the training renderers".into(),
            );
        }
        Some(report)
    } else {
        None
    };

    let report = CorpusValidationReport {
        schema_version: 1,
        valid: issues.is_empty(),
        corpus: corpus.display().to_string(),
        manifest_hash,
        cases: cases.len(),
        semantic_items: semantic_ids.len(),
        required_domain,
        minimum_cases_per_language_layout_corruption: minimum_cell,
        minimum_observed_cell_cases: minimum_observed,
        renderers,
        source_hashes,
        per_language,
        per_layout,
        per_corruption,
        per_language_layout_corruption: cells,
        separation_from_training,
        issues,
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    if !report.valid {
        return Err("vision corpus validation failed; see JSON evidence above".into());
    }
    Ok(())
}

fn validate_completion_metadata(
    corpus: &Path,
    manifest_hash: &str,
    cases: usize,
    issues: &mut Vec<ValidationIssue>,
) {
    let path = corpus.join("audit-metadata.json");
    let metadata: Value = match std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(value) => value,
        None => {
            issue(
                issues,
                None,
                "missing_completion_metadata",
                path.display().to_string(),
            );
            return;
        }
    };
    if metadata.get("complete").and_then(Value::as_bool) != Some(true) {
        issue(
            issues,
            None,
            "incomplete_corpus",
            "audit metadata is not marked complete".into(),
        );
    }
    if metadata.get("manifest_hash").and_then(Value::as_str) != Some(manifest_hash) {
        issue(
            issues,
            None,
            "manifest_hash_mismatch",
            format!(
                "metadata {:?}, decoded {manifest_hash}",
                metadata.get("manifest_hash")
            ),
        );
    }
    if metadata.get("rendered_cases").and_then(Value::as_u64) != Some(cases as u64) {
        issue(
            issues,
            None,
            "case_count_mismatch",
            format!(
                "metadata {:?}, manifest {cases}",
                metadata.get("rendered_cases")
            ),
        );
    }
}

fn issue(
    issues: &mut Vec<ValidationIssue>,
    case: Option<&VisionTextManifest>,
    kind: &'static str,
    detail: String,
) {
    issues.push(ValidationIssue {
        case_id: case.map(|value| value.case_id.clone()),
        kind,
        detail,
    });
}

fn read_rgb_png(path: &Path) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    let mut reader = decoder.read_info()?;
    let output_size = reader
        .output_buffer_size()
        .ok_or("PNG decoder could not bound output buffer")?;
    let mut buffer = vec![0; output_size];
    let info = reader.next_frame(&mut buffer)?;
    if info.color_type != png::ColorType::Rgb || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "{} is {:?}/{:?}; required 8-bit RGB",
            path.display(),
            info.color_type,
            info.bit_depth
        )
        .into());
    }
    buffer.truncate(info.buffer_size());
    Ok((info.width, info.height, buffer))
}

fn safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn read_manifest(path: &Path) -> Result<Vec<VisionTextManifest>, Box<dyn Error>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(
                serde_json::from_str(&line)
                    .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1).into()),
            ),
            Err(error) => Some(Err(error.into())),
        })
        .collect()
}

fn parse_domain(value: &str) -> Result<VisionCorpusDomain, Box<dyn Error>> {
    match value {
        "audit" => Ok(VisionCorpusDomain::Audit),
        "exam" => Ok(VisionCorpusDomain::Exam),
        "training-sibling" => Ok(VisionCorpusDomain::TrainingSibling),
        _ => Err(format!("unknown domain {value:?}").into()),
    }
}

fn language_label(case: &VisionTextManifest) -> String {
    serde_json::to_value(case.language)
        .ok()
        .and_then(|value| value.as_str().map(String::from))
        .map(|value| match value.as_str() {
            "english" => "en".into(),
            "german" => "de".into(),
            "french" => "fr".into(),
            _ => value,
        })
        .unwrap_or_else(|| "unknown".into())
}

fn layout_label(case: &VisionTextManifest) -> String {
    json_label(case.layout)
}

fn corruption_label(case: &VisionTextManifest) -> String {
    json_label(case.corruption)
}

fn json_label<T: Serialize>(value: T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(String::from))
        .unwrap_or_else(|| "unknown".into())
}

fn required<'a>(args: &'a [String], flag: &str) -> Result<&'a str, Box<dyn Error>> {
    let index = args
        .iter()
        .position(|value| value == flag)
        .ok_or_else(|| format!("{flag} is required"))?;
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn optional<T: std::str::FromStr>(args: &[String], flag: &str) -> Result<Option<T>, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    match optional_string(args, flag)? {
        Some(value) => Ok(Some(value.parse()?)),
        None => Ok(None),
    }
}

fn optional_string(args: &[String], flag: &str) -> Result<Option<String>, Box<dyn Error>> {
    let Some(index) = args.iter().position(|value| value == flag) else {
        return Ok(None);
    };
    Ok(Some(
        args.get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?
            .clone(),
    ))
}
