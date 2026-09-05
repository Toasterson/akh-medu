//! Generate a deterministic PNG + JSONL text-vision corpus.
//!
//! ```text
//! cargo run -p grug-data --example generate_vision_exam -- \
//!   --out /tmp/babyllm-vision-exam --seed 42 --count 30 --variants 3 \
//!   --difficulty 2 --domain exam
//! ```

use grug_data::{VisionCorpusDomain, VisionTextGenerator, write_png};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    out: PathBuf,
    seed: u64,
    count: u64,
    index_stride: u64,
    variants: u32,
    difficulty: u8,
    domain: VisionCorpusDomain,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let images_dir = args.out.join("images");
    fs::create_dir_all(&images_dir)?;
    let manifest_path = args.out.join("manifest.jsonl");
    let mut manifest = BufWriter::new(File::create(&manifest_path)?);
    let generator = VisionTextGenerator;

    for ordinal in 0..args.count {
        let item_index = ordinal
            .checked_mul(args.index_stride)
            .ok_or("item index overflow")?;
        let spec = generator.spec(args.seed, item_index, args.domain, args.difficulty)?;
        for variant in 0..args.variants {
            let filename = format!("item-{item_index:06}-variant-{variant:02}.png");
            let relative = format!("images/{filename}");
            let case = generator.render(&spec, variant, &relative)?;
            let file = BufWriter::new(File::create(images_dir.join(filename))?);
            write_png(&case, file)?;
            serde_json::to_writer(&mut manifest, &case.manifest)?;
            manifest.write_all(b"\n")?;
        }
    }
    manifest.flush()?;
    eprintln!(
        "generated {} semantic items / {} rendered cases at {}",
        args.count,
        args.count * u64::from(args.variants),
        args.out.display()
    );
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut out = None;
    let mut seed = 42u64;
    let mut count = 30u64;
    let mut index_stride = 1u64;
    let mut variants = 3u32;
    let mut difficulty = 2u8;
    let mut domain = VisionCorpusDomain::Exam;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value)),
            "--seed" => seed = value.parse()?,
            "--count" => count = value.parse()?,
            "--index-stride" => index_stride = value.parse()?,
            "--variants" => variants = value.parse()?,
            "--difficulty" => difficulty = value.parse()?,
            "--domain" => {
                domain = match value.as_str() {
                    "exam" => VisionCorpusDomain::Exam,
                    "training-sibling" => VisionCorpusDomain::TrainingSibling,
                    _ => return Err(format!("unknown domain {value:?}").into()),
                }
            }
            _ => return Err(format!("unknown flag {flag:?}").into()),
        }
    }
    let out = out.ok_or("--out is required")?;
    if count == 0 || index_stride == 0 || variants == 0 {
        return Err("--count, --index-stride and --variants must be positive".into());
    }
    Ok(Args {
        out,
        seed,
        count,
        index_stride,
        variants,
        difficulty,
        domain,
    })
}
