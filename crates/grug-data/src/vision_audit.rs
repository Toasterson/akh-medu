//! Generator-independent synthetic OCR audit renderer.
//!
//! This module intentionally does not call [`crate::VisionTextGenerator`] or
//! use its bitmap glyphs/content bank. It loads one explicit, content-hashed
//! OpenType font and rasterizes outlines through `ab_glyph`. That gives the
//! frozen audit lane a genuinely different glyph source and rendering stack.
//! Real and human-red-team images can share the same manifest schema later;
//! this generator is the reproducible first independent rung, not a claim
//! that synthetic audit images cover real camera/document distributions.

use ab_glyph::{Font, FontArc, PxScale, ScaleFont, point};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::vision_text::{
    RenderedVisionTextCase, VisionArtifactProvenance, VisionCorpusDomain, VisionCorruption,
    VisionLanguage, VisionLayout, VisionTextManifest, apply_vision_corruption, vision_domain_seed,
    vision_perceptual_hash, vision_pixel_hash,
};

pub const VISION_AUDIT_GENERATOR_VERSION: &str = "babyllm-vision-audit-ttf-v1";
pub const VISION_AUDIT_RENDERER_VERSION: &str = "ab-glyph-0.2.32/outline-raster-v1";

#[derive(Debug, thiserror::Error)]
pub enum VisionAuditGeneratorError {
    #[error("difficulty {0} is outside the supported range 0..=5")]
    InvalidDifficulty(u8),
    #[error("OpenType font is invalid: {0}")]
    InvalidFont(String),
    #[error("source field {0} must not be blank")]
    BlankSourceField(&'static str),
    #[error("font has no glyph for {character:?} (U+{codepoint:04X})")]
    MissingGlyph { character: char, codepoint: u32 },
    #[error("text did not fit audit canvas {width}x{height}")]
    TextDoesNotFit { width: u32, height: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionAuditFontIdentity {
    pub source_id: String,
    pub source_revision: String,
    pub source_license: String,
    pub font_hash: String,
}

/// A separate OpenType-backed renderer for frozen audit cases.
#[derive(Debug, Clone)]
pub struct VisionAuditGenerator {
    font: FontArc,
    identity: VisionAuditFontIdentity,
    renderer_id: String,
}

impl VisionAuditGenerator {
    pub fn new(
        font_bytes: Vec<u8>,
        source_id: impl Into<String>,
        source_revision: impl Into<String>,
        source_license: impl Into<String>,
    ) -> Result<Self, VisionAuditGeneratorError> {
        let source_id = source_id.into();
        let source_revision = source_revision.into();
        let source_license = source_license.into();
        for (field, value) in [
            ("source_id", source_id.as_str()),
            ("source_revision", source_revision.as_str()),
            ("source_license", source_license.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(VisionAuditGeneratorError::BlankSourceField(field));
            }
        }
        let font_hash = format!("blake3:{}", blake3::hash(&font_bytes).to_hex());
        let short_hash = font_hash
            .strip_prefix("blake3:")
            .unwrap_or(&font_hash)
            .get(..16)
            .unwrap_or("unknown");
        let renderer_id = format!("{VISION_AUDIT_RENDERER_VERSION}/{source_id}@{short_hash}");
        let font = FontArc::try_from_vec(font_bytes)
            .map_err(|error| VisionAuditGeneratorError::InvalidFont(error.to_string()))?;
        Ok(Self {
            font,
            identity: VisionAuditFontIdentity {
                source_id,
                source_revision,
                source_license,
                font_hash,
            },
            renderer_id,
        })
    }

    pub fn font_identity(&self) -> &VisionAuditFontIdentity {
        &self.identity
    }

    pub fn renderer_id(&self) -> &str {
        &self.renderer_id
    }

    pub fn render(
        &self,
        seed: u64,
        item_index: u64,
        difficulty: u8,
        variant: u32,
        image_path: impl Into<String>,
    ) -> Result<RenderedVisionTextCase, VisionAuditGeneratorError> {
        if difficulty > 5 {
            return Err(VisionAuditGeneratorError::InvalidDifficulty(difficulty));
        }
        let language = VisionLanguage::ALL[item_index as usize % VisionLanguage::ALL.len()];
        let layout = VisionLayout::ALL
            [(item_index as usize / VisionLanguage::ALL.len()) % VisionLayout::ALL.len()];
        let transcript = audit_transcript(seed, item_index, language, layout);
        self.validate_glyphs(&transcript)?;
        let corruption = audit_corruption(variant);
        let severity = if corruption == VisionCorruption::Clean {
            0
        } else {
            difficulty.max(1)
        };
        let (width, height) = audit_dimensions(layout);
        let render_seed = vision_domain_seed(seed, "audit-ttf", item_index, &variant.to_string());
        let (mut rgb, font_pixels) =
            self.render_text(&transcript, width, height, difficulty, corruption)?;
        apply_vision_corruption(&mut rgb, width, height, corruption, severity, render_seed);

        let semantic_hash =
            blake3::hash(format!("audit\u{ff}{}\u{ff}{transcript}", language.code()).as_bytes());
        let case_hash = blake3::hash(
            format!(
                "{}\u{ff}{render_seed}\u{ff}{variant}\u{ff}{VISION_AUDIT_GENERATOR_VERSION}",
                semantic_hash.to_hex()
            )
            .as_bytes(),
        );
        let glyph_digest = self.glyph_digest(&transcript);
        let manifest = VisionTextManifest {
            schema_version: 2,
            generator_version: VISION_AUDIT_GENERATOR_VERSION.into(),
            renderer_id: self.renderer_id.clone(),
            provenance: Some(VisionArtifactProvenance {
                kind: "synthetic_independent_renderer".into(),
                source_id: self.identity.source_id.clone(),
                source_revision: self.identity.source_revision.clone(),
                source_license: self.identity.source_license.clone(),
                source_hash: self.identity.font_hash.clone(),
            }),
            glyph_digest,
            case_id: format!("vaudit-{}", &case_hash.to_hex()[..16]),
            semantic_id: format!("vaudit-semantic-{}", &semantic_hash.to_hex()[..16]),
            domain: VisionCorpusDomain::Audit,
            language,
            layout,
            difficulty,
            font_scale: font_pixels,
            corruption,
            corruption_severity: severity,
            seed: render_seed,
            variant,
            content_bank: "audit-independent-ttf-v1".into(),
            unicode_normalization: "NFC".into(),
            transcript,
            image_path: image_path.into(),
            width,
            height,
            channels: 3,
            pixel_hash_fnv1a64: vision_pixel_hash(&rgb),
            pixel_hash_blake3: Some(format!("blake3:{}", blake3::hash(&rgb).to_hex())),
            perceptual_hash: vision_perceptual_hash(&rgb, width, height),
        };
        Ok(RenderedVisionTextCase { manifest, rgb })
    }

    fn validate_glyphs(&self, text: &str) -> Result<(), VisionAuditGeneratorError> {
        for character in text.chars().filter(|character| !character.is_whitespace()) {
            if self.font.glyph_id(character).0 == 0 {
                return Err(VisionAuditGeneratorError::MissingGlyph {
                    character,
                    codepoint: character as u32,
                });
            }
        }
        Ok(())
    }

    fn glyph_digest(&self, text: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(VISION_AUDIT_RENDERER_VERSION.as_bytes());
        hasher.update(self.identity.font_hash.as_bytes());
        let mut characters: Vec<char> = text
            .chars()
            .filter(|value| !value.is_whitespace())
            .collect();
        characters.sort_unstable();
        characters.dedup();
        for character in characters {
            hasher.update(&(character as u32).to_le_bytes());
            hasher.update(&self.font.glyph_id(character).0.to_le_bytes());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    fn render_text(
        &self,
        text: &str,
        width: u32,
        height: u32,
        difficulty: u8,
        corruption: VisionCorruption,
    ) -> Result<(Vec<u8>, u8), VisionAuditGeneratorError> {
        let margin = 10.0f32;
        let mut pixels = match difficulty {
            0 => 25.0,
            1 => 23.0,
            2 => 21.0,
            3 => 19.0,
            4 => 17.0,
            _ => 15.0,
        };
        let (lines, line_height) = loop {
            let scale = PxScale::from(pixels);
            let lines = wrap_audit_lines(&self.font, text, scale, width as f32 - 2.0 * margin);
            let scaled = self.font.as_scaled(scale);
            let line_height = scaled.height() + scaled.line_gap().max(1.0);
            if margin * 2.0 + line_height * lines.len() as f32 <= height as f32 {
                break (lines, line_height);
            }
            if pixels <= 10.0 {
                return Err(VisionAuditGeneratorError::TextDoesNotFit { width, height });
            }
            pixels -= 1.0;
        };

        let (background, foreground) = if corruption == VisionCorruption::LowContrast {
            ([232u8, 234, 230], [145u8, 147, 143])
        } else {
            ([246u8, 244, 238], [18u8, 24, 31])
        };
        let mut rgb = vec![0u8; width as usize * height as usize * 3];
        for pixel in rgb.chunks_exact_mut(3) {
            pixel.copy_from_slice(&background);
        }
        let scale = PxScale::from(pixels);
        let scaled = self.font.as_scaled(scale);
        let mut baseline = margin + scaled.ascent();
        for line in lines {
            let mut caret = margin;
            let mut previous = None;
            for character in line.chars() {
                let glyph_id = scaled.glyph_id(character);
                if let Some(previous) = previous {
                    caret += scaled.kern(previous, glyph_id);
                }
                let glyph = glyph_id.with_scale_and_position(scale, point(caret, baseline));
                if let Some(outlined) = self.font.outline_glyph(glyph) {
                    let bounds = outlined.px_bounds();
                    outlined.draw(|local_x, local_y, coverage| {
                        let x = bounds.min.x.floor() as i32 + local_x as i32;
                        let y = bounds.min.y.floor() as i32 + local_y as i32;
                        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                            return;
                        }
                        let base = (y as usize * width as usize + x as usize) * 3;
                        for channel in 0..3 {
                            let blended = f32::from(background[channel]) * (1.0 - coverage)
                                + f32::from(foreground[channel]) * coverage;
                            rgb[base + channel] = blended.round().clamp(0.0, 255.0) as u8;
                        }
                    });
                }
                caret += scaled.h_advance(glyph_id);
                previous = Some(glyph_id);
            }
            baseline += line_height;
        }
        Ok((rgb, pixels.round() as u8))
    }
}

fn audit_dimensions(layout: VisionLayout) -> (u32, u32) {
    match layout {
        VisionLayout::Line => (448, 80),
        VisionLayout::Paragraph => (448, 176),
        VisionLayout::Receipt => (352, 224),
        VisionLayout::Terminal | VisionLayout::ApiError => (512, 192),
    }
}

fn audit_corruption(variant: u32) -> VisionCorruption {
    match variant % 7 {
        0 => VisionCorruption::Clean,
        1 => VisionCorruption::LowContrast,
        2 => VisionCorruption::SaltPepper,
        3 => VisionCorruption::BoxBlur,
        4 => VisionCorruption::Occlusion,
        5 => VisionCorruption::UnevenLighting,
        _ => VisionCorruption::HorizontalShear,
    }
}

fn audit_transcript(
    seed: u64,
    item_index: u64,
    language: VisionLanguage,
    layout: VisionLayout,
) -> String {
    let bank = audit_content_bank(language);
    let selection = vision_domain_seed(seed, "audit-content", item_index, language.code());
    let first = bank[selection as usize % bank.len()];
    let second = bank[(selection.rotate_left(23) as usize + 7) % bank.len()];
    let code = (selection % 89_000) + 10_000;
    let amount_major = 3 + selection % 197;
    let amount_minor = selection.rotate_left(9) % 100;
    let text = match layout {
        VisionLayout::Line => format!("{first} | {code}"),
        VisionLayout::Paragraph => format!(
            "{first}.\nDocument {code}: {second}.\nPage {} of {}",
            1 + item_index % 7,
            8 + item_index % 19
        ),
        VisionLayout::Receipt => format!(
            "RIVER BOOKS\nTicket {code}\n==================\n{first}\n{second}\nAMOUNT CHF {amount_major}.{amount_minor:02}"
        ),
        VisionLayout::Terminal => format!(
            "> medu notes verify --offline\nstatus=degraded code={code}\n{first}\n{second}\nretry_ms={}",
            125 + item_index % 8_000
        ),
        VisionLayout::ApiError => format!(
            "HTTP/1.1 429 TOO MANY REQUESTS\nx-request: audit-{code}\n{{\"cached\":true,\"wait_ms\":{}}}\n{first}\n{second}",
            250 + item_index % 9_000
        ),
    };
    text.nfc().collect()
}

fn audit_content_bank(language: VisionLanguage) -> &'static [&'static str] {
    match language {
        VisionLanguage::English => &[
            "Chapter index loaded from disk",
            "The citation points to page thirty",
            "Keep the earlier note as history",
            "A cached copy answers this request",
            "No verified passage supports the claim",
            "The scanner found three headings",
            "Table cell seven contains the total",
            "Use the newer edition for this value",
            "Local inference remains operational",
            "The attachment checksum does not match",
            "Search the textbook before answering",
            "The queue will retry this operation",
            "Read column two from the invoice",
            "The service returned malformed JSON",
            "Preserve both revisions of the note",
            "Evidence is missing for this answer",
        ],
        VisionLanguage::German => &[
            "Das Kapitel wurde lokal geladen",
            "Das Zitat verweist auf Seite dreißig",
            "Die frühere Notiz bleibt erhalten",
            "Eine Kopie beantwortet diese Anfrage",
            "Kein Beleg stützt diese Aussage",
            "Der Scan fand drei Überschriften",
            "Zelle sieben enthält die Summe",
            "Nutze die neuere Ausgabe für den Wert",
            "Lokale Inferenz bleibt verfügbar",
            "Die Prüfsumme des Anhangs ist falsch",
            "Durchsuche zuerst das Lehrbuch",
            "Die Warteschlange versucht es erneut",
            "Lies Spalte zwei aus der Rechnung",
            "Der Dienst lieferte ungültiges JSON",
            "Erhalte beide Fassungen der Notiz",
            "Für diese Antwort fehlt ein Beleg",
        ],
        VisionLanguage::French => &[
            "Le chapitre est chargé depuis le disque",
            "La citation renvoie à la page trente",
            "Conservez la note précédente dans l'historique",
            "Une copie locale répond à cette requête",
            "Aucun passage vérifié ne soutient l'affirmation",
            "Le scan a trouvé trois titres",
            "La cellule sept contient le total",
            "Utilisez la nouvelle édition pour cette valeur",
            "L'inférence locale reste disponible",
            "La somme de contrôle du fichier est incorrecte",
            "Recherchez d'abord dans le manuel",
            "La file réessaiera cette opération",
            "Lisez la deuxième colonne de la facture",
            "Le service a renvoyé un JSON invalide",
            "Conservez les deux versions de la note",
            "Cette réponse manque de preuve",
        ],
    }
}

fn wrap_audit_lines<F: Font>(font: &F, text: &str, scale: PxScale, max_width: f32) -> Vec<String> {
    let mut output = Vec::new();
    for source_line in text.lines() {
        let mut current = String::new();
        for word in source_line.split_whitespace() {
            let candidate = if current.is_empty() {
                word.to_string()
            } else {
                format!("{current} {word}")
            };
            if !current.is_empty() && measure_line(font, &candidate, scale) > max_width {
                output.push(std::mem::take(&mut current));
                current.push_str(word);
            } else {
                current = candidate;
            }
        }
        output.push(current);
    }
    output
}

fn measure_line<F: Font>(font: &F, text: &str, scale: PxScale) -> f32 {
    let scaled = font.as_scaled(scale);
    let mut width = 0.0;
    let mut previous = None;
    for character in text.chars() {
        let glyph = scaled.glyph_id(character);
        if let Some(previous) = previous {
            width += scaled.kern(previous, glyph);
        }
        width += scaled.h_advance(glyph);
        previous = Some(glyph);
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_font_and_blank_provenance_fail_closed() {
        assert!(matches!(
            VisionAuditGenerator::new(vec![1, 2, 3], "font", "1", "OFL-1.1"),
            Err(VisionAuditGeneratorError::InvalidFont(_))
        ));
        assert!(matches!(
            VisionAuditGenerator::new(vec![1, 2, 3], "", "1", "OFL-1.1"),
            Err(VisionAuditGeneratorError::BlankSourceField("source_id"))
        ));
    }
}
