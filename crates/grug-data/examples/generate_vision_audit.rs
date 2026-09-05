//! Generate a frozen OCR audit corpus with a content-hashed OpenType font and
//! an outline rasterizer independent from the training/rolling-exam renderer.

use grug_data::{VisionAuditGenerator, VisionCorruption, VisionLanguage, VisionLayout, write_png};
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    out: PathBuf,
    font: PathBuf,
    font_id: String,
    font_revision: String,
    font_license: String,
    seed: u64,
    count: u64,
    index_stride: u64,
    variants: u32,
    difficulty: u8,
}

#[derive(Debug, Serialize)]
struct AuditCorpusMetadata<'a> {
    schema_version: u32,
    complete: bool,
    generator_version: &'static str,
    renderer_id: &'a str,
    font_identity: &'a grug_data::vision_audit::VisionAuditFontIdentity,
    seed: u64,
    semantic_items: u64,
    rendered_cases: u64,
    index_stride: u64,
    variants: u32,
    difficulty: u8,
    manifest_file: &'static str,
    manifest_hash: String,
    per_language: BTreeMap<String, u64>,
    per_layout: BTreeMap<String, u64>,
    per_corruption: BTreeMap<String, u64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.out.exists() && args.out.read_dir()?.next().is_some() {
        return Err(format!(
            "output {} is not empty; a frozen audit corpus is never overwritten",
            args.out.display()
        )
        .into());
    }
    let font_bytes = fs::read(&args.font)?;
    let generator = VisionAuditGenerator::new(
        font_bytes,
        args.font_id,
        args.font_revision,
        args.font_license,
    )?;
    let images_dir = args.out.join("images");
    fs::create_dir_all(&images_dir)?;
    let manifest_path = args.out.join("manifest.jsonl");
    let mut manifest = BufWriter::new(File::create(&manifest_path)?);
    let mut per_language = BTreeMap::new();
    let mut per_layout = BTreeMap::new();
    let mut per_corruption = BTreeMap::new();

    for ordinal in 0..args.count {
        let item_index = ordinal
            .checked_mul(args.index_stride)
            .ok_or("item index overflow")?;
        for variant in 0..args.variants {
            let filename = format!("audit-{item_index:06}-variant-{variant:02}.png");
            let relative = format!("images/{filename}");
            let case =
                generator.render(args.seed, item_index, args.difficulty, variant, &relative)?;
            let file = BufWriter::new(File::create(images_dir.join(filename))?);
            write_png(&case, file)?;
            serde_json::to_writer(&mut manifest, &case.manifest)?;
            manifest.write_all(b"\n")?;
            *per_language
                .entry(language_label(case.manifest.language).into())
                .or_default() += 1;
            *per_layout
                .entry(layout_label(case.manifest.layout).into())
                .or_default() += 1;
            *per_corruption
                .entry(corruption_label(case.manifest.corruption).into())
                .or_default() += 1;
        }
    }
    manifest.flush()?;
    drop(manifest);
    let manifest_hash = format!(
        "blake3:{}",
        blake3::hash(&fs::read(&manifest_path)?).to_hex()
    );
    let metadata = AuditCorpusMetadata {
        schema_version: 1,
        complete: true,
        generator_version: grug_data::VISION_AUDIT_GENERATOR_VERSION,
        renderer_id: generator.renderer_id(),
        font_identity: generator.font_identity(),
        seed: args.seed,
        semantic_items: args.count,
        rendered_cases: args.count * u64::from(args.variants),
        index_stride: args.index_stride,
        variants: args.variants,
        difficulty: args.difficulty,
        manifest_file: "manifest.jsonl",
        manifest_hash,
        per_language,
        per_layout,
        per_corruption,
    };
    // Written last: its presence marks a complete, frozen audit corpus.
    fs::write(
        args.out.join("audit-metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &metadata)?;
    println!();
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let values: Vec<String> = std::env::args().collect();
    let out = required(&values, "--out").map(PathBuf::from)?;
    let font = required(&values, "--font").map(PathBuf::from)?;
    let font_id = required(&values, "--font-id")?.to_string();
    let font_revision = required(&values, "--font-revision")?.to_string();
    let font_license = required(&values, "--font-license")?.to_string();
    let seed = optional(&values, "--seed")?.unwrap_or(91_700u64);
    let count = optional(&values, "--count")?.unwrap_or(150u64);
    let index_stride = optional(&values, "--index-stride")?.unwrap_or(1u64);
    let variants = optional(&values, "--variants")?.unwrap_or(7u32);
    let difficulty = optional(&values, "--difficulty")?.unwrap_or(2u8);
    if count == 0 || index_stride == 0 || variants == 0 {
        return Err("--count, --index-stride and --variants must be positive".into());
    }
    if difficulty > 5 {
        return Err("--difficulty must be in 0..=5".into());
    }
    Ok(Args {
        out,
        font,
        font_id,
        font_revision,
        font_license,
        seed,
        count,
        index_stride,
        variants,
        difficulty,
    })
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
    let Some(index) = args.iter().position(|value| value == flag) else {
        return Ok(None);
    };
    let value = args
        .get(index + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?;
    Ok(Some(value.parse()?))
}

fn language_label(value: VisionLanguage) -> &'static str {
    match value {
        VisionLanguage::English => "en",
        VisionLanguage::German => "de",
        VisionLanguage::French => "fr",
    }
}

fn layout_label(value: VisionLayout) -> &'static str {
    match value {
        VisionLayout::Line => "line",
        VisionLayout::Paragraph => "paragraph",
        VisionLayout::Receipt => "receipt",
        VisionLayout::Terminal => "terminal",
        VisionLayout::ApiError => "api_error",
    }
}

fn corruption_label(value: VisionCorruption) -> &'static str {
    match value {
        VisionCorruption::Clean => "clean",
        VisionCorruption::LowContrast => "low_contrast",
        VisionCorruption::SaltPepper => "salt_pepper",
        VisionCorruption::BoxBlur => "box_blur",
        VisionCorruption::Occlusion => "occlusion",
        VisionCorruption::UnevenLighting => "uneven_lighting",
        VisionCorruption::HorizontalShear => "horizontal_shear",
    }
}
