//! Deterministic synthetic text-vision cases and OCR scoring.
//!
//! The bootstrap renderer uses the `font8x8` bitmaps compiled into this crate,
//! so pixels do not depend on host fonts, FreeType, locale, or network access.
//! It is intentionally a *first* renderer, not an independent audit source.
//! Real images and a different rendering stack are still required for the
//! generator-independent audit lane defined by ADR 100.

use crate::adaptive_exam::{
    SiblingSeparationEvidence, SiblingSeparationPolicy, WilsonInterval, character_ngram_jaccard,
    perceptual_hash_hamming, validate_sibling_separation, wilson_interval,
};
use crate::similarity::levenshtein;
use font8x8::{BASIC_FONTS, LATIN_FONTS, UnicodeFonts};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Seek, Write};
use std::path::{Component, Path};
use unicode_normalization::UnicodeNormalization;

pub const VISION_TEXT_GENERATOR_VERSION: &str = "babyllm-vision-text-v1";
pub const VISION_TEXT_RENDERER_ID: &str = "font8x8-unicode-0.3/babyllm-raster-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionCorpusDomain {
    Exam,
    TrainingSibling,
    /// Frozen cases created by a separate renderer or collected from real
    /// images. The bootstrap generator refuses to manufacture this domain.
    Audit,
}

impl VisionCorpusDomain {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exam => "exam",
            Self::TrainingSibling => "training-sibling",
            Self::Audit => "audit",
        }
    }

    fn content_bank(self) -> &'static str {
        match self {
            Self::Exam => "exam-heldout-v1",
            Self::TrainingSibling => "training-sibling-v1",
            Self::Audit => "audit-external-v1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionLanguage {
    English,
    German,
    French,
}

impl VisionLanguage {
    pub const ALL: [Self; 3] = [Self::English, Self::German, Self::French];

    pub fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::German => "de",
            Self::French => "fr",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionLayout {
    Line,
    Paragraph,
    Receipt,
    Terminal,
    ApiError,
}

impl VisionLayout {
    pub const ALL: [Self; 5] = [
        Self::Line,
        Self::Paragraph,
        Self::Receipt,
        Self::Terminal,
        Self::ApiError,
    ];

    fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Line => (384, 64),
            Self::Paragraph => (384, 144),
            Self::Receipt => (320, 192),
            Self::Terminal | Self::ApiError => (480, 176),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Paragraph => "paragraph",
            Self::Receipt => "receipt",
            Self::Terminal => "terminal",
            Self::ApiError => "api_error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionCorruption {
    Clean,
    LowContrast,
    SaltPepper,
    BoxBlur,
    Occlusion,
    UnevenLighting,
    HorizontalShear,
}

impl VisionCorruption {
    fn for_variant(variant: u32) -> Self {
        match variant % 7 {
            0 => Self::Clean,
            1 => Self::LowContrast,
            2 => Self::SaltPepper,
            3 => Self::BoxBlur,
            4 => Self::Occlusion,
            5 => Self::UnevenLighting,
            _ => Self::HorizontalShear,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::LowContrast => "low_contrast",
            Self::SaltPepper => "salt_pepper",
            Self::BoxBlur => "box_blur",
            Self::Occlusion => "occlusion",
            Self::UnevenLighting => "uneven_lighting",
            Self::HorizontalShear => "horizontal_shear",
        }
    }
}

/// Immutable source identity carried by every newly generated image. Older
/// V1 manifests deserialize without it, but promotion-grade audit manifests
/// must provide all fields and a content hash for their font/image source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionArtifactProvenance {
    pub kind: String,
    pub source_id: String,
    pub source_revision: String,
    pub source_license: String,
    pub source_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionTextSpec {
    pub seed: u64,
    pub item_index: u64,
    pub domain: VisionCorpusDomain,
    pub language: VisionLanguage,
    pub layout: VisionLayout,
    pub difficulty: u8,
    pub transcript: String,
    pub content_bank: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionTextManifest {
    pub schema_version: u32,
    pub generator_version: String,
    pub renderer_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<VisionArtifactProvenance>,
    /// Hash of the exact compiled glyph bitmaps used by this transcript.
    pub glyph_digest: String,
    pub case_id: String,
    /// Shared by corruption/rendering variants of the same transcript.
    pub semantic_id: String,
    pub domain: VisionCorpusDomain,
    pub language: VisionLanguage,
    pub layout: VisionLayout,
    pub difficulty: u8,
    pub font_scale: u8,
    pub corruption: VisionCorruption,
    pub corruption_severity: u8,
    pub seed: u64,
    pub variant: u32,
    pub content_bank: String,
    pub unicode_normalization: String,
    pub transcript: String,
    pub image_path: String,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub pixel_hash_fnv1a64: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_hash_blake3: Option<String>,
    pub perceptual_hash: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedVisionTextCase {
    pub manifest: VisionTextManifest,
    pub rgb: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum VisionTextError {
    #[error("difficulty {0} is outside the supported range 0..=5")]
    InvalidDifficulty(u8),
    #[error("the bootstrap generator cannot create audit-domain cases; use an independent source")]
    AuditMustBeIndependent,
    #[error("renderer has no pinned glyph for {character:?} (U+{codepoint:04X})")]
    MissingGlyph { character: char, codepoint: u32 },
    #[error("RGB buffer length {actual} does not match {width}x{height}x3 ({expected})")]
    InvalidImageLength {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    #[error("PNG encoding failed: {0}")]
    Png(#[from] png::EncodingError),
    #[error("PNG decoding failed: {0}")]
    PngDecode(#[from] png::DecodingError),
    #[error("PNG output buffer is too large to bound safely")]
    UnboundedPngOutput,
    #[error("PNG must be 8-bit RGB, found {color_type:?}/{bit_depth:?}")]
    UnsupportedPngFormat {
        color_type: png::ColorType,
        bit_depth: png::BitDepth,
    },
    #[error("image has {pixels} pixels, above the decoding limit {maximum}")]
    ImageTooLarge { pixels: u64, maximum: u64 },
    #[error("image path {0:?} is not a safe relative path")]
    UnsafeImagePath(String),
    #[error("destination image dimensions must be positive, got {width}x{height}")]
    EmptyDestination { width: u32, height: u32 },
    #[error("prediction {0:?} appears more than once")]
    DuplicatePrediction(String),
    #[error("prediction {0:?} has no matching manifest case")]
    UnknownPrediction(String),
    #[error(
        "glyph probe does not fit: glyph={character:?}, scale={scale}, offset=({offset_x},{offset_y}), canvas={width}x{height}"
    )]
    InvalidGlyphProbeGeometry {
        character: char,
        scale: usize,
        offset_x: usize,
        offset_y: usize,
        width: usize,
        height: usize,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VisionTextGenerator;

impl VisionTextGenerator {
    /// Select a language/layout and a domain-specific content bank from a
    /// stable seed. Exam and training sibling banks do not share phrases.
    pub fn spec(
        &self,
        seed: u64,
        item_index: u64,
        domain: VisionCorpusDomain,
        difficulty: u8,
    ) -> Result<VisionTextSpec, VisionTextError> {
        if difficulty > 5 {
            return Err(VisionTextError::InvalidDifficulty(difficulty));
        }
        if domain == VisionCorpusDomain::Audit {
            return Err(VisionTextError::AuditMustBeIndependent);
        }
        let language = VisionLanguage::ALL[item_index as usize % VisionLanguage::ALL.len()];
        let layout = VisionLayout::ALL
            [(item_index as usize / VisionLanguage::ALL.len()) % VisionLayout::ALL.len()];
        let selection_seed = domain_seed(seed, domain.as_str(), item_index, "content");
        let phrases = content_bank(domain, language);
        let first = phrases[selection_seed as usize % phrases.len()];
        let second = phrases[(selection_seed.rotate_left(17) as usize + 1) % phrases.len()];
        let transcript = format_layout(layout, first, second, item_index);
        let transcript: String = transcript.nfc().collect();
        // Fail at construction time rather than silently substituting a glyph.
        glyph_digest(&transcript)?;
        Ok(VisionTextSpec {
            seed,
            item_index,
            domain,
            language,
            layout,
            difficulty,
            transcript,
            content_bank: domain.content_bank().into(),
        })
    }

    /// Render one semantic item under a deterministic corruption variant.
    pub fn render(
        &self,
        spec: &VisionTextSpec,
        variant: u32,
        image_path: impl Into<String>,
    ) -> Result<RenderedVisionTextCase, VisionTextError> {
        if spec.difficulty > 5 {
            return Err(VisionTextError::InvalidDifficulty(spec.difficulty));
        }
        let corruption = VisionCorruption::for_variant(variant);
        let severity = if corruption == VisionCorruption::Clean {
            0
        } else {
            spec.difficulty.max(1)
        };
        let (width, height) = spec.layout.dimensions();
        let render_seed = domain_seed(
            spec.seed,
            spec.domain.as_str(),
            spec.item_index,
            &variant.to_string(),
        );
        let (mut rgb, font_scale) = render_text(
            &spec.transcript,
            width,
            height,
            spec.difficulty,
            corruption,
            severity,
            render_seed,
        )?;
        apply_vision_corruption(&mut rgb, width, height, corruption, severity, render_seed);

        let semantic_hash = fnv1a64(
            format!(
                "{}\u{ff}{}\u{ff}{}",
                spec.domain.as_str(),
                spec.language.code(),
                spec.transcript
            )
            .as_bytes(),
        );
        let case_hash = fnv1a64(
            format!(
                "{}\u{ff}{}\u{ff}{}\u{ff}{}",
                semantic_hash, render_seed, variant, VISION_TEXT_GENERATOR_VERSION
            )
            .as_bytes(),
        );
        let pixel_hash = fnv1a64(&rgb);
        let manifest = VisionTextManifest {
            schema_version: 1,
            generator_version: VISION_TEXT_GENERATOR_VERSION.into(),
            renderer_id: VISION_TEXT_RENDERER_ID.into(),
            provenance: Some(VisionArtifactProvenance {
                kind: "synthetic_training_or_rolling_exam".into(),
                source_id: "font8x8-unicode".into(),
                source_revision: "0.3.1".into(),
                source_license: "MIT".into(),
                source_hash: glyph_digest(&spec.transcript)?,
            }),
            glyph_digest: glyph_digest(&spec.transcript)?,
            case_id: format!("vtext-{case_hash:016x}"),
            semantic_id: format!("vsemantic-{semantic_hash:016x}"),
            domain: spec.domain,
            language: spec.language,
            layout: spec.layout,
            difficulty: spec.difficulty,
            font_scale: font_scale as u8,
            corruption,
            corruption_severity: severity,
            seed: render_seed,
            variant,
            content_bank: spec.content_bank.clone(),
            unicode_normalization: "NFC".into(),
            transcript: spec.transcript.clone(),
            image_path: image_path.into(),
            width,
            height,
            channels: 3,
            pixel_hash_fnv1a64: format!("fnv1a64:{pixel_hash:016x}"),
            pixel_hash_blake3: Some(format!("blake3:{}", blake3::hash(&rgb).to_hex())),
            perceptual_hash: foreground_difference_hash(&rgb, width, height),
        };
        Ok(RenderedVisionTextCase { manifest, rgb })
    }
}

/// Write a rendered case as an 8-bit RGB PNG.
pub fn write_png(case: &RenderedVisionTextCase, writer: impl Write) -> Result<(), VisionTextError> {
    let expected = case.manifest.width as usize * case.manifest.height as usize * 3;
    if case.rgb.len() != expected {
        return Err(VisionTextError::InvalidImageLength {
            width: case.manifest.width,
            height: case.manifest.height,
            expected,
            actual: case.rgb.len(),
        });
    }
    let mut encoder = png::Encoder::new(writer, case.manifest.width, case.manifest.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(&case.rgb)?;
    Ok(())
}

/// Decode an exact 8-bit RGB PNG with an explicit pixel-allocation ceiling.
/// No palette/alpha/bit-depth conversion is hidden in this boundary, so the
/// decoded bytes are the same bytes covered by manifest pixel hashes.
pub fn read_rgb_png(
    reader: impl BufRead + Seek,
    maximum_pixels: u64,
) -> Result<(u32, u32, Vec<u8>), VisionTextError> {
    let decoder = png::Decoder::new(reader);
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    let pixels = u64::from(info.width) * u64::from(info.height);
    if pixels > maximum_pixels {
        return Err(VisionTextError::ImageTooLarge {
            pixels,
            maximum: maximum_pixels,
        });
    }
    let output_size = reader
        .output_buffer_size()
        .ok_or(VisionTextError::UnboundedPngOutput)?;
    let mut buffer = vec![0; output_size];
    let frame = reader.next_frame(&mut buffer)?;
    if frame.color_type != png::ColorType::Rgb || frame.bit_depth != png::BitDepth::Eight {
        return Err(VisionTextError::UnsupportedPngFormat {
            color_type: frame.color_type,
            bit_depth: frame.bit_depth,
        });
    }
    buffer.truncate(frame.buffer_size());
    Ok((frame.width, frame.height, buffer))
}

/// Decode a user-supplied PNG into 8-bit RGB for native inference. Unlike the
/// strict corpus validator above, this accepts the common grayscale/RGBA forms
/// emitted by screenshot tools. Alpha is composited over white; indexed and
/// sub-8-bit inputs are expanded by the PNG decoder. Corpus validation keeps
/// using [`read_rgb_png`] so artifact pixels never change implicitly.
pub fn read_png_normalized_rgb(
    reader: impl BufRead + Seek,
    maximum_pixels: u64,
) -> Result<(u32, u32, Vec<u8>), VisionTextError> {
    let mut decoder = png::Decoder::new(reader);
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    let pixels = u64::from(info.width) * u64::from(info.height);
    if pixels > maximum_pixels {
        return Err(VisionTextError::ImageTooLarge {
            pixels,
            maximum: maximum_pixels,
        });
    }
    let output_size = reader
        .output_buffer_size()
        .ok_or(VisionTextError::UnboundedPngOutput)?;
    let mut buffer = vec![0; output_size];
    let frame = reader.next_frame(&mut buffer)?;
    if frame.bit_depth != png::BitDepth::Eight {
        return Err(VisionTextError::UnsupportedPngFormat {
            color_type: frame.color_type,
            bit_depth: frame.bit_depth,
        });
    }
    buffer.truncate(frame.buffer_size());
    let composite = |value: u8, alpha: u8| {
        let alpha = u16::from(alpha);
        ((u16::from(value) * alpha + 255 * (255 - alpha) + 127) / 255) as u8
    };
    let rgb = match frame.color_type {
        png::ColorType::Rgb => buffer,
        png::ColorType::Rgba => buffer
            .chunks_exact(4)
            .flat_map(|pixel| {
                [
                    composite(pixel[0], pixel[3]),
                    composite(pixel[1], pixel[3]),
                    composite(pixel[2], pixel[3]),
                ]
            })
            .collect(),
        png::ColorType::Grayscale => buffer
            .into_iter()
            .flat_map(|value| [value, value, value])
            .collect(),
        png::ColorType::GrayscaleAlpha => buffer
            .chunks_exact(2)
            .flat_map(|pixel| {
                let value = composite(pixel[0], pixel[1]);
                [value, value, value]
            })
            .collect(),
        png::ColorType::Indexed => {
            return Err(VisionTextError::UnsupportedPngFormat {
                color_type: frame.color_type,
                bit_depth: frame.bit_depth,
            });
        }
    };
    Ok((frame.width, frame.height, rgb))
}

/// Resolve and decode one manifest image without allowing absolute paths or
/// `..` traversal outside the corpus root.
pub fn load_manifest_rgb(
    corpus_root: &Path,
    case: &VisionTextManifest,
    maximum_pixels: u64,
) -> Result<(u32, u32, Vec<u8>), VisionTextError> {
    let relative = Path::new(&case.image_path);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(VisionTextError::UnsafeImagePath(case.image_path.clone()));
    }
    let file = std::fs::File::open(corpus_root.join(relative))
        .map_err(|error| VisionTextError::PngDecode(png::DecodingError::IoError(error)))?;
    read_rgb_png(std::io::BufReader::new(file), maximum_pixels)
}

/// Aspect-preserving bilinear resize into a centered RGB canvas. This gives
/// mixed line/receipt/screenshot batches one fixed tensor shape without
/// cropping text or stretching glyph geometry.
pub fn letterbox_rgb(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    fill: [u8; 3],
) -> Result<Vec<u8>, VisionTextError> {
    let expected = source_width as usize * source_height as usize * 3;
    if source.len() != expected {
        return Err(VisionTextError::InvalidImageLength {
            width: source_width,
            height: source_height,
            expected,
            actual: source.len(),
        });
    }
    if target_width == 0 || target_height == 0 {
        return Err(VisionTextError::EmptyDestination {
            width: target_width,
            height: target_height,
        });
    }
    let scale = (target_width as f64 / source_width.max(1) as f64)
        .min(target_height as f64 / source_height.max(1) as f64);
    let resized_width = ((source_width as f64 * scale).round() as u32).clamp(1, target_width);
    let resized_height = ((source_height as f64 * scale).round() as u32).clamp(1, target_height);
    let offset_x = (target_width - resized_width) / 2;
    let offset_y = (target_height - resized_height) / 2;
    let mut output = vec![0u8; target_width as usize * target_height as usize * 3];
    for pixel in output.chunks_exact_mut(3) {
        pixel.copy_from_slice(&fill);
    }
    for target_y in 0..resized_height {
        let source_y = ((f64::from(target_y) + 0.5) / scale - 0.5)
            .clamp(0.0, f64::from(source_height.saturating_sub(1)));
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(source_height - 1);
        let wy = source_y - f64::from(y0);
        for target_x in 0..resized_width {
            let source_x = ((f64::from(target_x) + 0.5) / scale - 0.5)
                .clamp(0.0, f64::from(source_width.saturating_sub(1)));
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(source_width - 1);
            let wx = source_x - f64::from(x0);
            let destination = ((target_y + offset_y) as usize * target_width as usize
                + (target_x + offset_x) as usize)
                * 3;
            for channel in 0..3 {
                let sample = |x: u32, y: u32| {
                    f64::from(
                        source[(y as usize * source_width as usize + x as usize) * 3 + channel],
                    )
                };
                let top = sample(x0, y0) * (1.0 - wx) + sample(x1, y0) * wx;
                let bottom = sample(x0, y1) * (1.0 - wx) + sample(x1, y1) * wx;
                output[destination + channel] =
                    (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

/// Render one pinned bitmap glyph for the stem-resolution experiment.
///
/// The probe deliberately bypasses layout and language modelling: a caller
/// can vary glyph pixel size and sub-patch position, then ask whether the
/// visual stem plus a small readout retains class information. This is not an
/// OCR score and must not be reported as one.
pub fn render_glyph_probe(
    character: char,
    scale: usize,
    offset_x: usize,
    offset_y: usize,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, VisionTextError> {
    let bitmap = glyph(character).ok_or(VisionTextError::MissingGlyph {
        character,
        codepoint: character as u32,
    })?;
    if scale == 0
        || width == 0
        || height == 0
        || offset_x + 8 * scale > width
        || offset_y + 8 * scale > height
    {
        return Err(VisionTextError::InvalidGlyphProbeGeometry {
            character,
            scale,
            offset_x,
            offset_y,
            width,
            height,
        });
    }
    let mut rgb = vec![248u8; width * height * 3];
    for (glyph_y, row) in bitmap.iter().copied().enumerate() {
        for glyph_x in 0..8 {
            if row & (1 << glyph_x) == 0 {
                continue;
            }
            for dy in 0..scale {
                for dx in 0..scale {
                    let x = offset_x + glyph_x * scale + dx;
                    let y = offset_y + glyph_y * scale + dy;
                    let base = (y * width + x) * 3;
                    rgb[base..base + 3].fill(16);
                }
            }
        }
    }
    Ok(rgb)
}

fn content_bank(domain: VisionCorpusDomain, language: VisionLanguage) -> &'static [&'static str] {
    match (domain, language) {
        (VisionCorpusDomain::Exam, VisionLanguage::English) => &[
            "The local reader stays available",
            "Total CHF 42.70",
            "Retry after twelve seconds",
            "Invoice number A-1937",
            "Service unavailable",
        ],
        (VisionCorpusDomain::Exam, VisionLanguage::German) => &[
            "Die Überprüfung schlägt fehl",
            "Größe: zwölf Bytes",
            "Straße 17, Zürich",
            "Summe CHF 42.70",
            "Dienst nicht verfügbar",
        ],
        (VisionCorpusDomain::Exam, VisionLanguage::French) => &[
            "Échec de l'API: délai dépassé",
            "Montant CHF 42.70",
            "Où est le reçu numéro 17",
            "Réessayez dans douze secondes",
            "Service indisponible",
        ],
        (VisionCorpusDomain::TrainingSibling, VisionLanguage::English) => &[
            "Read the screen without a network",
            "Balance CHF 18.25",
            "Try again in seven seconds",
            "Order reference B-5824",
            "Gateway returned an error",
        ],
        (VisionCorpusDomain::TrainingSibling, VisionLanguage::German) => &[
            "Das Bild wird lokal gelesen",
            "Betrag CHF 18.25",
            "Nächster Versuch in sieben Sekunden",
            "Bestellung B-5824",
            "Verbindung vorübergehend gestört",
        ],
        (VisionCorpusDomain::TrainingSibling, VisionLanguage::French) => &[
            "Lecture locale de l'image",
            "Solde CHF 18.25",
            "Nouvel essai dans sept secondes",
            "Commande B-5824",
            "Connexion temporairement coupée",
        ],
        (VisionCorpusDomain::Audit, _) => &[],
    }
}

fn format_layout(layout: VisionLayout, first: &str, second: &str, index: u64) -> String {
    match layout {
        VisionLayout::Line => format!("{first} [{index:04}]"),
        VisionLayout::Paragraph => format!("{first}.\nRef {index:04}: {second}."),
        VisionLayout::Receipt => format!(
            "OPEN FLOW LAB\nRECEIPT {index:04}\n----------------\n{first}\n{second}\nTOTAL CHF {:02}.{:02}",
            10 + index % 80,
            (index * 17) % 100
        ),
        VisionLayout::Terminal => {
            format!("$ akh inspect --local\nERROR 503\n{first}\n{second}\ntrace_id={index:08x}")
        }
        VisionLayout::ApiError => {
            format!("{{\"error\":\"timeout\",\"retry\":false}}\n{first}\nrequest_id={index:08x}")
        }
    }
}

fn glyph(character: char) -> Option<[u8; 8]> {
    BASIC_FONTS
        .get(character)
        .or_else(|| LATIN_FONTS.get(character))
}

fn glyph_digest(text: &str) -> Result<String, VisionTextError> {
    let mut unique: BTreeSet<char> = text.chars().filter(|c| *c != '\n' && *c != '\r').collect();
    unique.insert(' ');
    let mut bytes = Vec::with_capacity(unique.len() * 12);
    bytes.extend_from_slice(VISION_TEXT_RENDERER_ID.as_bytes());
    for character in unique {
        let bitmap = glyph(character).ok_or(VisionTextError::MissingGlyph {
            character,
            codepoint: character as u32,
        })?;
        bytes.extend_from_slice(&(character as u32).to_le_bytes());
        bytes.extend_from_slice(&bitmap);
    }
    Ok(format!("fnv1a64:{:016x}", fnv1a64(&bytes)))
}

fn render_text(
    transcript: &str,
    width: u32,
    height: u32,
    difficulty: u8,
    corruption: VisionCorruption,
    severity: u8,
    seed: u64,
) -> Result<(Vec<u8>, usize), VisionTextError> {
    let mut scale = match difficulty {
        0 | 1 => 4,
        2 | 3 => 3,
        _ => 2,
    };
    let background = if corruption == VisionCorruption::LowContrast {
        232u8
    } else {
        248u8
    };
    let foreground = if corruption == VisionCorruption::LowContrast {
        150u8.saturating_add(severity.saturating_mul(5))
    } else {
        16u8
    };
    let margin = 8usize;
    let lines = loop {
        let glyph_advance = 9usize * scale;
        let max_chars = ((width as usize).saturating_sub(margin * 2) / glyph_advance).max(1);
        let candidate = wrap_lines(transcript, max_chars);
        let required_height =
            margin + candidate.len().saturating_sub(1) * 10 * scale + 8 * scale + margin;
        if required_height <= height as usize || scale == 1 {
            break candidate;
        }
        scale -= 1;
    };
    let mut rgb = vec![background; width as usize * height as usize * 3];
    let glyph_advance = 9usize * scale;
    let line_advance = 10usize * scale;
    let mut y = margin;
    let x_jitter = (seed as usize % scale.max(1)).min(2);
    for line in lines {
        if y + 8 * scale >= height as usize {
            break;
        }
        let mut x = margin + x_jitter;
        for character in line.chars() {
            let bitmap = glyph(character).ok_or(VisionTextError::MissingGlyph {
                character,
                codepoint: character as u32,
            })?;
            for (glyph_y, row) in bitmap.iter().copied().enumerate() {
                for glyph_x in 0..8 {
                    if row & (1 << glyph_x) == 0 {
                        continue;
                    }
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = x + glyph_x * scale + dx;
                            let py = y + glyph_y * scale + dy;
                            if px < width as usize && py < height as usize {
                                let base = (py * width as usize + px) * 3;
                                rgb[base..base + 3].fill(foreground);
                            }
                        }
                    }
                }
            }
            x += glyph_advance;
            if x + 8 * scale >= width as usize {
                break;
            }
        }
        y += line_advance;
    }
    Ok((rgb, scale))
}

fn wrap_lines(text: &str, max_chars: usize) -> Vec<String> {
    let mut output = Vec::new();
    for source_line in text.lines() {
        let mut current = String::new();
        for word in source_line.split_whitespace() {
            let required =
                current.chars().count() + usize::from(!current.is_empty()) + word.chars().count();
            if required > max_chars && !current.is_empty() {
                output.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
        output.push(current);
    }
    output
}

pub(crate) fn apply_vision_corruption(
    rgb: &mut [u8],
    width: u32,
    height: u32,
    corruption: VisionCorruption,
    severity: u8,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(domain_seed(
        seed,
        "corruption",
        u64::from(severity),
        "pixels",
    ));
    match corruption {
        VisionCorruption::Clean | VisionCorruption::LowContrast => {}
        VisionCorruption::SaltPepper => {
            let probability = (0.004 * f64::from(severity)).min(0.03);
            for pixel in rgb.chunks_exact_mut(3) {
                if rng.gen_bool(probability) {
                    pixel.fill(if rng.gen_bool(0.5) { 0 } else { 255 });
                }
            }
        }
        VisionCorruption::BoxBlur => {
            let source = rgb.to_vec();
            let radius = usize::from(severity >= 4) + 1;
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let mut sum = [0u32; 3];
                    let mut count = 0u32;
                    for dy in y.saturating_sub(radius)..=(y + radius).min(height as usize - 1) {
                        for dx in x.saturating_sub(radius)..=(x + radius).min(width as usize - 1) {
                            let base = (dy * width as usize + dx) * 3;
                            for channel in 0..3 {
                                sum[channel] += u32::from(source[base + channel]);
                            }
                            count += 1;
                        }
                    }
                    let base = (y * width as usize + x) * 3;
                    for channel in 0..3 {
                        rgb[base + channel] = (sum[channel] / count) as u8;
                    }
                }
            }
        }
        VisionCorruption::Occlusion => {
            let bar_height = (2 + usize::from(severity) * 2).min(height as usize / 4);
            let start = rng.gen_range(0..(height as usize).saturating_sub(bar_height).max(1));
            for y in start..(start + bar_height).min(height as usize) {
                let row = &mut rgb[y * width as usize * 3..(y + 1) * width as usize * 3];
                row.fill(220);
            }
        }
        VisionCorruption::UnevenLighting => {
            let direction = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            let strength = 10.0 + 6.0 * f32::from(severity);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let normalized = x as f32 / width.saturating_sub(1).max(1) as f32 - 0.5;
                    let vertical = y as f32 / height.saturating_sub(1).max(1) as f32 - 0.5;
                    let delta = direction * strength * normalized + strength * 0.25 * vertical;
                    let base = (y * width as usize + x) * 3;
                    for channel in &mut rgb[base..base + 3] {
                        *channel = (f32::from(*channel) + delta).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        VisionCorruption::HorizontalShear => {
            let source = rgb.to_vec();
            let maximum = usize::from(severity.max(1)) * 2;
            let background = 248u8;
            rgb.fill(background);
            for y in 0..height as usize {
                let centered = y as isize - height as isize / 2;
                let offset = centered * maximum as isize / height.max(1) as isize;
                for x in 0..width as usize {
                    let target_x = x as isize + offset;
                    if !(0..width as isize).contains(&target_x) {
                        continue;
                    }
                    let source_base = (y * width as usize + x) * 3;
                    let target_base = (y * width as usize + target_x as usize) * 3;
                    rgb[target_base..target_base + 3]
                        .copy_from_slice(&source[source_base..source_base + 3]);
                }
            }
        }
    }
}

/// Stable compatibility hash over decoded RGB pixels.
pub fn vision_pixel_hash(rgb: &[u8]) -> String {
    format!("fnv1a64:{:016x}", fnv1a64(rgb))
}

/// Foreground-normalized difference hash used by corpus quarantine checks.
pub fn vision_perceptual_hash(rgb: &[u8], width: u32, height: u32) -> u64 {
    foreground_difference_hash(rgb, width, height)
}

pub(crate) fn vision_domain_seed(seed: u64, domain: &str, index: u64, purpose: &str) -> u64 {
    domain_seed(seed, domain, index, purpose)
}

fn foreground_difference_hash(rgb: &[u8], width: u32, height: u32) -> u64 {
    let width = width as usize;
    let height = height as usize;
    let luminance = |x: usize, y: usize| -> u32 {
        let base = (y * width + x) * 3;
        (u32::from(rgb[base]) + u32::from(rgb[base + 1]) + u32::from(rgb[base + 2])) / 3
    };
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..height {
        for x in 0..width {
            if luminance(x, y) < 210 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if min_x > max_x || min_y > max_y {
        return 0;
    }
    let crop_width = max_x - min_x + 1;
    let crop_height = max_y - min_y + 1;
    let mut cells = [[0u32; 9]; 8];
    let mut counts = [[0u32; 9]; 8];
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let cell_x = ((x - min_x) * 9 / crop_width).min(8);
            let cell_y = ((y - min_y) * 8 / crop_height).min(7);
            cells[cell_y][cell_x] += luminance(x, y);
            counts[cell_y][cell_x] += 1;
        }
    }
    let mut hash = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            let left = cells[y][x] / counts[y][x].max(1);
            let right = cells[y][x + 1] / counts[y][x + 1].max(1);
            if left < right {
                hash |= 1 << (y * 8 + x);
            }
        }
    }
    hash
}

fn domain_seed(seed: u64, domain: &str, index: u64, purpose: &str) -> u64 {
    fnv1a64(format!("{seed}\u{ff}{domain}\u{ff}{index}\u{ff}{purpose}").as_bytes())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionTextPrediction {
    pub case_id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VisionSliceScore {
    pub cases: u64,
    pub predicted: u64,
    pub exact_rate: f64,
    pub exact_rate_interval_95: WilsonInterval,
    pub character_error_rate: f64,
    pub character_error_rate_interval_95: MetricInterval,
    pub word_error_rate: f64,
    pub word_error_rate_interval_95: MetricInterval,
    pub blank_rate: f64,
    pub blank_rate_interval_95: WilsonInterval,
    pub near_miss_rate: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct MetricInterval {
    pub lower: f64,
    pub estimate: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VisionTextScoreSummary {
    pub overall: VisionSliceScore,
    /// Exact normalized prediction equality across >=2 renderings of one text.
    pub metamorphic_consistency: f64,
    pub metamorphic_groups: u64,
    /// Distinct representative predictions / semantic items. Rendering
    /// variants of one semantic item count once.
    pub unique_semantic_prediction_rate: f64,
    /// Fraction of semantic items sharing an output with another, different
    /// semantic item. High values are a collapse alarm.
    pub cross_semantic_collapse_rate: f64,
    pub suggested_pass_count: u64,
    pub suggested_precursor_count: u64,
    /// Hard blank/invalid outcome count. Unlike predictive precursors, this
    /// can trigger a stop while precursor steering remains in shadow mode.
    pub suggested_catastrophic_count: u64,
    pub per_language: BTreeMap<String, VisionSliceScore>,
    pub per_layout: BTreeMap<String, VisionSliceScore>,
    pub per_corruption: BTreeMap<String, VisionSliceScore>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionSeparationReport {
    pub exam_cases: u64,
    pub training_cases: u64,
    pub exact_transcript_overlap: u64,
    pub exact_pixel_overlap: u64,
    pub evidence: SiblingSeparationEvidence,
    pub policy: SiblingSeparationPolicy,
    pub admitted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejection: Option<String>,
}

/// Compute the evidence used by the adaptive controller's sibling admission
/// gate. A report is always returned, including the reason for rejection, so
/// failed corpora remain auditable rather than disappearing from a pipeline.
pub fn audit_vision_sibling_separation(
    exam: &[VisionTextManifest],
    training: &[VisionTextManifest],
    policy: SiblingSeparationPolicy,
) -> VisionSeparationReport {
    let exam_banks: BTreeSet<&str> = exam.iter().map(|case| case.content_bank.as_str()).collect();
    let training_banks: BTreeSet<&str> = training
        .iter()
        .map(|case| case.content_bank.as_str())
        .collect();
    let exam_renderers: BTreeSet<&str> =
        exam.iter().map(|case| case.renderer_id.as_str()).collect();
    let training_renderers: BTreeSet<&str> = training
        .iter()
        .map(|case| case.renderer_id.as_str())
        .collect();
    let mut held_out_axes = Vec::new();
    if exam_banks.is_disjoint(&training_banks) {
        held_out_axes.push("content_bank".into());
    }
    if exam_renderers.is_disjoint(&training_renderers) {
        held_out_axes.push("renderer".into());
    }

    let exam_transcripts: BTreeSet<String> = exam
        .iter()
        .map(|case| normalize_transcript(&case.transcript))
        .collect();
    let training_transcripts: BTreeSet<String> = training
        .iter()
        .map(|case| normalize_transcript(&case.transcript))
        .collect();
    let exact_transcript_overlap =
        exam_transcripts.intersection(&training_transcripts).count() as u64;
    let exam_pixels: BTreeSet<&str> = exam
        .iter()
        .map(|case| case.pixel_hash_fnv1a64.as_str())
        .collect();
    let training_pixels: BTreeSet<&str> = training
        .iter()
        .map(|case| case.pixel_hash_fnv1a64.as_str())
        .collect();
    let exact_pixel_overlap = exam_pixels.intersection(&training_pixels).count() as u64;

    let mut max_jaccard = 0.0f64;
    let mut min_hash_distance = None;
    for exam_case in exam
        .iter()
        .filter(|case| case.corruption == VisionCorruption::Clean)
    {
        for training_case in training
            .iter()
            .filter(|case| case.corruption == VisionCorruption::Clean)
        {
            let similarity =
                character_ngram_jaccard(&exam_case.transcript, &training_case.transcript, 3);
            let distance =
                perceptual_hash_hamming(exam_case.perceptual_hash, training_case.perceptual_hash);
            if similarity > max_jaccard {
                max_jaccard = similarity;
                min_hash_distance = Some(distance);
            } else if (similarity - max_jaccard).abs() < f64::EPSILON {
                min_hash_distance =
                    Some(min_hash_distance.map_or(distance, |old: u32| old.min(distance)));
            }
        }
    }
    let evidence = SiblingSeparationEvidence {
        held_out_axes,
        max_transcript_ngram_jaccard: max_jaccard,
        min_image_hash_hamming_distance: min_hash_distance,
    };
    let validation = validate_sibling_separation(&policy, &evidence);
    VisionSeparationReport {
        exam_cases: exam.len() as u64,
        training_cases: training.len() as u64,
        exact_transcript_overlap,
        exact_pixel_overlap,
        evidence,
        policy,
        admitted: validation.is_ok() && exact_transcript_overlap == 0 && exact_pixel_overlap == 0,
        rejection: if exact_transcript_overlap > 0 || exact_pixel_overlap > 0 {
            Some(format!(
                "exact overlap: {exact_transcript_overlap} transcripts, {exact_pixel_overlap} images"
            ))
        } else {
            validation.err().map(|error| error.to_string())
        },
    }
}

#[derive(Default)]
struct ScoreAccumulator {
    cases: u64,
    predicted: u64,
    exact: u64,
    character_edits: u64,
    reference_characters: u64,
    word_edits: u64,
    reference_words: u64,
    blank: u64,
    near_miss: u64,
    case_metrics: Vec<CaseMetric>,
}

#[derive(Clone, Copy)]
struct CaseMetric {
    character_edits: u64,
    reference_characters: u64,
    word_edits: u64,
    reference_words: u64,
}

impl ScoreAccumulator {
    fn add(&mut self, reference: &str, prediction: Option<&str>) {
        self.cases += 1;
        let prediction = prediction.unwrap_or("");
        if !prediction.is_empty() {
            self.predicted += 1;
        }
        let reference = normalize_transcript(reference);
        let prediction = normalize_transcript(prediction);
        let ref_chars: Vec<char> = reference.chars().collect();
        let pred_chars: Vec<char> = prediction.chars().collect();
        let char_edits = levenshtein(&ref_chars, &pred_chars);
        let ref_words: Vec<&str> = reference.split_whitespace().collect();
        let pred_words: Vec<&str> = prediction.split_whitespace().collect();
        let word_edits = sequence_levenshtein(&ref_words, &pred_words);
        let exact = reference == prediction;
        if exact {
            self.exact += 1;
        }
        if prediction.is_empty() {
            self.blank += 1;
        }
        let case_cer = char_edits as f64 / ref_chars.len().max(1) as f64;
        if !exact && !prediction.is_empty() && case_cer <= 0.20 {
            self.near_miss += 1;
        }
        self.character_edits += char_edits as u64;
        self.reference_characters += ref_chars.len().max(1) as u64;
        self.word_edits += word_edits as u64;
        self.reference_words += ref_words.len().max(1) as u64;
        self.case_metrics.push(CaseMetric {
            character_edits: char_edits as u64,
            reference_characters: ref_chars.len().max(1) as u64,
            word_edits: word_edits as u64,
            reference_words: ref_words.len().max(1) as u64,
        });
    }

    fn finish(self) -> VisionSliceScore {
        let cases = self.cases.max(1) as f64;
        let character_error_rate =
            self.character_edits as f64 / self.reference_characters.max(1) as f64;
        let word_error_rate = self.word_edits as f64 / self.reference_words.max(1) as f64;
        VisionSliceScore {
            cases: self.cases,
            predicted: self.predicted,
            exact_rate: self.exact as f64 / cases,
            exact_rate_interval_95: wilson_interval(self.exact, self.cases, 1.959_963_984_540_054),
            character_error_rate,
            character_error_rate_interval_95: bootstrap_error_interval(
                &self.case_metrics,
                false,
                character_error_rate,
            ),
            word_error_rate,
            word_error_rate_interval_95: bootstrap_error_interval(
                &self.case_metrics,
                true,
                word_error_rate,
            ),
            blank_rate: self.blank as f64 / cases,
            blank_rate_interval_95: wilson_interval(self.blank, self.cases, 1.959_963_984_540_054),
            near_miss_rate: self.near_miss as f64 / cases,
        }
    }
}

/// Deterministic case-resampling interval. Case-level bootstrap preserves
/// within-line character/word dependence instead of pretending every glyph
/// is an independent Bernoulli trial.
fn bootstrap_error_interval(cases: &[CaseMetric], words: bool, estimate: f64) -> MetricInterval {
    if cases.is_empty() {
        return MetricInterval {
            lower: 0.0,
            estimate,
            upper: 0.0,
        };
    }
    if cases.len() == 1 {
        return MetricInterval {
            lower: estimate,
            estimate,
            upper: estimate,
        };
    }
    let mut seed_bytes = Vec::with_capacity(cases.len() * 32 + 1);
    seed_bytes.push(u8::from(words));
    for case in cases {
        seed_bytes.extend_from_slice(&case.character_edits.to_le_bytes());
        seed_bytes.extend_from_slice(&case.reference_characters.to_le_bytes());
        seed_bytes.extend_from_slice(&case.word_edits.to_le_bytes());
        seed_bytes.extend_from_slice(&case.reference_words.to_le_bytes());
    }
    let mut rng = StdRng::seed_from_u64(fnv1a64(&seed_bytes));
    let mut estimates = Vec::with_capacity(1_000);
    for _ in 0..1_000 {
        let mut edits = 0u64;
        let mut units = 0u64;
        for _ in 0..cases.len() {
            let case = cases[rng.gen_range(0..cases.len())];
            if words {
                edits += case.word_edits;
                units += case.reference_words;
            } else {
                edits += case.character_edits;
                units += case.reference_characters;
            }
        }
        estimates.push(edits as f64 / units.max(1) as f64);
    }
    estimates.sort_by(f64::total_cmp);
    MetricInterval {
        lower: estimates[24],
        estimate,
        upper: estimates[974],
    }
}

pub fn score_vision_text(
    cases: &[VisionTextManifest],
    predictions: &[VisionTextPrediction],
) -> Result<VisionTextScoreSummary, VisionTextError> {
    let manifest_ids: BTreeSet<&str> = cases.iter().map(|case| case.case_id.as_str()).collect();
    let mut prediction_map = BTreeMap::new();
    for prediction in predictions {
        if !manifest_ids.contains(prediction.case_id.as_str()) {
            return Err(VisionTextError::UnknownPrediction(
                prediction.case_id.clone(),
            ));
        }
        if prediction_map
            .insert(prediction.case_id.as_str(), prediction.text.as_str())
            .is_some()
        {
            return Err(VisionTextError::DuplicatePrediction(
                prediction.case_id.clone(),
            ));
        }
    }

    let mut overall = ScoreAccumulator::default();
    let mut language: BTreeMap<String, ScoreAccumulator> = BTreeMap::new();
    let mut layout: BTreeMap<String, ScoreAccumulator> = BTreeMap::new();
    let mut corruption: BTreeMap<String, ScoreAccumulator> = BTreeMap::new();
    let mut semantic: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut exact_count = 0u64;
    let mut precursor_count = 0u64;
    let mut catastrophic_count = 0u64;

    for case in cases {
        let prediction = prediction_map.get(case.case_id.as_str()).copied();
        let normalized_reference = normalize_transcript(&case.transcript);
        let normalized_prediction = normalize_transcript(prediction.unwrap_or(""));
        let chars_ref: Vec<char> = normalized_reference.chars().collect();
        let chars_pred: Vec<char> = normalized_prediction.chars().collect();
        let edits = levenshtein(&chars_ref, &chars_pred);
        let exact = normalized_reference == normalized_prediction;
        let near_miss = !exact
            && !normalized_prediction.is_empty()
            && edits as f64 / chars_ref.len().max(1) as f64 <= 0.20;
        if exact {
            exact_count += 1;
        }
        if normalized_prediction.is_empty() {
            catastrophic_count += 1;
        }
        if normalized_prediction.is_empty() || near_miss {
            precursor_count += 1;
        }

        overall.add(&case.transcript, prediction);
        language
            .entry(case.language.code().into())
            .or_default()
            .add(&case.transcript, prediction);
        layout
            .entry(case.layout.as_str().into())
            .or_default()
            .add(&case.transcript, prediction);
        corruption
            .entry(case.corruption.as_str().into())
            .or_default()
            .add(&case.transcript, prediction);
        semantic
            .entry(&case.semantic_id)
            .or_default()
            .push(normalized_prediction);
    }

    let comparable: Vec<&Vec<String>> =
        semantic.values().filter(|group| group.len() >= 2).collect();
    let consistent = comparable
        .iter()
        .filter(|group| group.windows(2).all(|pair| pair[0] == pair[1]))
        .count();
    let metamorphic_groups = comparable.len() as u64;
    let metamorphic_consistency = if comparable.is_empty() {
        0.0
    } else {
        consistent as f64 / comparable.len() as f64
    };
    let representatives: Vec<(&str, String)> = semantic
        .iter()
        .map(|(semantic_id, outputs)| {
            let mut counts = BTreeMap::<&str, usize>::new();
            for output in outputs {
                *counts.entry(output.as_str()).or_default() += 1;
            }
            let representative = counts
                .into_iter()
                .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(left.0)))
                .map(|(output, _)| output.to_string())
                .unwrap_or_default();
            (*semantic_id, representative)
        })
        .collect();
    let unique_predictions: BTreeSet<&str> = representatives
        .iter()
        .map(|(_, output)| output.as_str())
        .collect();
    let mut output_semantics: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (semantic_id, output) in &representatives {
        output_semantics
            .entry(output.as_str())
            .or_default()
            .insert(*semantic_id);
    }
    let collapsed_semantics: BTreeSet<&str> = output_semantics
        .values()
        .filter(|semantic_ids| semantic_ids.len() > 1)
        .flat_map(|semantic_ids| semantic_ids.iter().copied())
        .collect();
    let semantic_count = representatives.len().max(1) as f64;
    let unique_semantic_prediction_rate = unique_predictions.len() as f64 / semantic_count;
    let cross_semantic_collapse_rate = collapsed_semantics.len() as f64 / semantic_count;
    // Each inconsistent semantic group is one additional precursor, capped at
    // the item count so the value can feed the binomial controller directly.
    // Cross-semantic repeated outputs are an additional early collapse signal.
    precursor_count = precursor_count
        .saturating_add((comparable.len() - consistent) as u64)
        .saturating_add(
            cases
                .iter()
                .filter(|case| collapsed_semantics.contains(case.semantic_id.as_str()))
                .count() as u64,
        )
        .min(cases.len() as u64);

    Ok(VisionTextScoreSummary {
        overall: overall.finish(),
        metamorphic_consistency,
        metamorphic_groups,
        unique_semantic_prediction_rate,
        cross_semantic_collapse_rate,
        suggested_pass_count: exact_count,
        suggested_precursor_count: precursor_count,
        suggested_catastrophic_count: catastrophic_count,
        per_language: language
            .into_iter()
            .map(|(key, value)| (key, value.finish()))
            .collect(),
        per_layout: layout
            .into_iter()
            .map(|(key, value)| (key, value.finish()))
            .collect(),
        per_corruption: corruption
            .into_iter()
            .map(|(key, value)| (key, value.finish()))
            .collect(),
    })
}

fn normalize_transcript(value: &str) -> String {
    value
        .replace("\r\n", "\n")
        .nfc()
        .collect::<String>()
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn sequence_levenshtein<T: Eq>(left: &[T], right: &[T]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (i, left_item) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, right_item) in right.iter().enumerate() {
            let cost = usize::from(left_item != right_item);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_is_pixel_and_manifest_deterministic() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(42, 1, VisionCorpusDomain::Exam, 3).unwrap();
        let first = generator.render(&spec, 2, "images/a.png").unwrap();
        let second = generator.render(&spec, 2, "images/a.png").unwrap();
        assert_eq!(first, second);
        assert!(first.rgb.iter().any(|pixel| *pixel < 200));
    }

    #[test]
    fn glyph_probe_is_deterministic_and_rejects_clipping() {
        let first = render_glyph_probe('A', 1, 3, 5, 24, 24).unwrap();
        let second = render_glyph_probe('A', 1, 3, 5, 24, 24).unwrap();
        assert_eq!(first, second);
        assert!(first.contains(&16));
        assert!(matches!(
            render_glyph_probe('A', 2, 9, 0, 24, 24),
            Err(VisionTextError::InvalidGlyphProbeGeometry { .. })
        ));
    }

    #[test]
    fn exam_and_training_use_separate_seed_and_content_domains() {
        let generator = VisionTextGenerator;
        let exam = generator.spec(7, 0, VisionCorpusDomain::Exam, 1).unwrap();
        let training = generator
            .spec(7, 0, VisionCorpusDomain::TrainingSibling, 1)
            .unwrap();
        assert_ne!(exam.content_bank, training.content_bank);
        assert_ne!(exam.transcript, training.transcript);
        let exam = generator.render(&exam, 0, "exam.png").unwrap();
        let training = generator.render(&training, 0, "train.png").unwrap();
        assert_ne!(exam.manifest.seed, training.manifest.seed);
    }

    #[test]
    fn perfect_predictions_score_zero_error_and_full_consistency() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(42, 0, VisionCorpusDomain::Exam, 2).unwrap();
        let cases: Vec<_> = (0..3)
            .map(|variant| {
                generator
                    .render(&spec, variant, format!("{variant}.png"))
                    .unwrap()
                    .manifest
            })
            .collect();
        let predictions: Vec<_> = cases
            .iter()
            .map(|case| VisionTextPrediction {
                case_id: case.case_id.clone(),
                text: case.transcript.clone(),
                confidence: Some(1.0),
            })
            .collect();
        let score = score_vision_text(&cases, &predictions).unwrap();
        assert_eq!(score.overall.exact_rate, 1.0);
        assert_eq!(score.overall.character_error_rate, 0.0);
        assert_eq!(score.overall.word_error_rate, 0.0);
        assert_eq!(score.metamorphic_groups, 1);
        assert_eq!(score.metamorphic_consistency, 1.0);
        assert_eq!(score.suggested_catastrophic_count, 0);
    }

    #[test]
    fn one_character_deletion_is_a_bounded_near_miss() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(42, 0, VisionCorpusDomain::Exam, 2).unwrap();
        let case = generator.render(&spec, 0, "0.png").unwrap().manifest;
        let mut prediction = case.transcript.clone();
        prediction.pop();
        let score = score_vision_text(
            std::slice::from_ref(&case),
            &[VisionTextPrediction {
                case_id: case.case_id.clone(),
                text: prediction,
                confidence: None,
            }],
        )
        .unwrap();
        assert!(score.overall.character_error_rate > 0.0);
        assert!(score.overall.character_error_rate < 0.20);
        assert_eq!(score.overall.near_miss_rate, 1.0);
        assert_eq!(score.suggested_precursor_count, 1);
        assert_eq!(score.suggested_catastrophic_count, 0);
    }

    #[test]
    fn blank_prediction_is_a_catastrophic_outcome() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(42, 0, VisionCorpusDomain::Exam, 2).unwrap();
        let case = generator.render(&spec, 0, "0.png").unwrap().manifest;
        let score = score_vision_text(
            std::slice::from_ref(&case),
            &[VisionTextPrediction {
                case_id: case.case_id.clone(),
                text: String::new(),
                confidence: None,
            }],
        )
        .unwrap();
        assert_eq!(score.suggested_precursor_count, 1);
        assert_eq!(score.suggested_catastrophic_count, 1);
    }

    #[test]
    fn png_writer_emits_png_signature() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(9, 2, VisionCorpusDomain::Exam, 1).unwrap();
        let case = generator.render(&spec, 0, "x.png").unwrap();
        let mut bytes = Vec::new();
        write_png(&case, &mut bytes).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn png_decode_round_trips_exact_manifest_pixels() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(9, 2, VisionCorpusDomain::Exam, 1).unwrap();
        let case = generator.render(&spec, 0, "x.png").unwrap();
        let mut bytes = Vec::new();
        write_png(&case, &mut bytes).unwrap();
        let (width, height, decoded) =
            read_rgb_png(std::io::Cursor::new(bytes), 1_000_000).unwrap();
        assert_eq!((width, height), (case.manifest.width, case.manifest.height));
        assert_eq!(decoded, case.rgb);
    }

    #[test]
    fn user_png_decoder_composites_rgba_without_weakening_corpus_validation() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&[0, 0, 0, 255, 12, 34, 56, 0])
                .unwrap();
        }
        assert!(matches!(
            read_rgb_png(std::io::Cursor::new(bytes.clone()), 100),
            Err(VisionTextError::UnsupportedPngFormat { .. })
        ));
        let (width, height, rgb) =
            read_png_normalized_rgb(std::io::Cursor::new(bytes), 100).unwrap();
        assert_eq!((width, height), (2, 1));
        assert_eq!(rgb, vec![0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn letterbox_preserves_aspect_and_uses_requested_fill() {
        let source = vec![0, 0, 0, 255, 255, 255];
        let output = letterbox_rgb(&source, 2, 1, 4, 4, [17, 18, 19]).unwrap();
        assert_eq!(output.len(), 4 * 4 * 3);
        assert_eq!(&output[..3], &[17, 18, 19]);
        assert_eq!(&output[(3 * 4 * 3)..(3 * 4 * 3 + 3)], &[17, 18, 19]);
        assert_eq!(&output[(1 * 4 * 3)..(1 * 4 * 3 + 3)], &[0, 0, 0]);
        assert_eq!(
            &output[(1 * 4 * 3 + 3 * 3)..(1 * 4 * 3 + 4 * 3)],
            &[255, 255, 255]
        );
    }

    #[test]
    fn manifest_loader_rejects_path_traversal_before_io() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(9, 2, VisionCorpusDomain::Exam, 1).unwrap();
        let mut case = generator.render(&spec, 0, "x.png").unwrap().manifest;
        case.image_path = "../outside.png".into();
        assert!(matches!(
            load_manifest_rgb(Path::new("unused"), &case, 1_000_000),
            Err(VisionTextError::UnsafeImagePath(_))
        ));
    }

    #[test]
    fn repeated_output_across_distinct_items_triggers_collapse_alarm() {
        let generator = VisionTextGenerator;
        let cases: Vec<_> = [0, 15]
            .into_iter()
            .map(|index| {
                let spec = generator
                    .spec(42, index, VisionCorpusDomain::Exam, 2)
                    .unwrap();
                generator
                    .render(&spec, 0, format!("{index}.png"))
                    .unwrap()
                    .manifest
            })
            .collect();
        assert_ne!(cases[0].transcript, cases[1].transcript);
        let predictions: Vec<_> = cases
            .iter()
            .map(|case| VisionTextPrediction {
                case_id: case.case_id.clone(),
                text: "same answer".into(),
                confidence: None,
            })
            .collect();
        let score = score_vision_text(&cases, &predictions).unwrap();
        assert_eq!(score.unique_semantic_prediction_rate, 0.5);
        assert_eq!(score.cross_semantic_collapse_rate, 1.0);
        assert_eq!(score.suggested_precursor_count, 2);
    }

    #[test]
    fn separation_audit_rejects_an_exam_reused_as_training() {
        let generator = VisionTextGenerator;
        let spec = generator.spec(9, 2, VisionCorpusDomain::Exam, 1).unwrap();
        let case = generator.render(&spec, 0, "x.png").unwrap().manifest;
        let report = audit_vision_sibling_separation(
            std::slice::from_ref(&case),
            std::slice::from_ref(&case),
            SiblingSeparationPolicy::default(),
        );
        assert!(!report.admitted);
        assert_eq!(report.exact_transcript_overlap, 1);
        assert_eq!(report.exact_pixel_overlap, 1);
    }
}
