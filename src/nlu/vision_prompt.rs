//! Reserve an image's place in a tokenized prompt.
//!
//! The caller writes `<|vision_start|><|image_pad|><|vision_end|>` where the
//! image belongs. The single `<|image_pad|>` is expanded to one token per merged
//! patch, and the surrounding tokens keep their text positions. Nothing here
//! guesses where an image goes: a prompt without the placeholder, or with more
//! than one, is refused.

use super::error::{NluError, NluResult};
use super::vision_positions::{ImageGrid, PromptPiece};
use std::ops::Range;

/// The three special token ids the placeholder is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionTokens {
    pub start: u32,
    pub pad: u32,
    pub end: u32,
}

/// A prompt with its image placeholder expanded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedPrompt {
    /// Token ids with one `pad` per merged patch.
    pub tokens: Vec<u32>,
    /// Text before, the image, text after - in the shape `position_triples`
    /// wants.
    pub pieces: Vec<PromptPiece>,
    /// Where the image's rows go in `tokens`.
    pub image_span: Range<usize>,
}

/// Expand the one bracketed `<|image_pad|>` in `tokens` to `grid.tokens()` pads.
///
/// Refuses zero placeholders, more than one, or one that is not immediately
/// bracketed by `start` and `end` - all with the exact string to use.
pub fn expand_single_image(
    tokens: &[u32],
    specials: VisionTokens,
    grid: ImageGrid,
) -> NluResult<ExpandedPrompt> {
    let pads: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, id)| **id == specials.pad)
        .map(|(index, _)| index)
        .collect();
    let refuse = |bracketed: bool| NluError::ImagePlaceholder {
        found: pads.len(),
        bracketed,
        grid_tokens: grid.tokens(),
    };
    let [at] = pads.as_slice() else {
        return Err(refuse(false));
    };
    let at = *at;
    let bracketed = at > 0
        && tokens[at - 1] == specials.start
        && tokens.get(at + 1) == Some(&specials.end);
    if !bracketed {
        return Err(refuse(false));
    }

    let count = grid.tokens();
    let mut expanded = Vec::with_capacity(tokens.len() - 1 + count);
    expanded.extend_from_slice(&tokens[..at]);
    expanded.extend(std::iter::repeat(specials.pad).take(count));
    expanded.extend_from_slice(&tokens[at + 1..]);

    // `at` tokens precede the image, including `<|vision_start|>`; everything
    // after the original pad follows it, starting with `<|vision_end|>`.
    let after = tokens.len() - at - 1;
    Ok(ExpandedPrompt {
        pieces: vec![
            PromptPiece::Text(at),
            PromptPiece::Image(grid),
            PromptPiece::Text(after),
        ],
        image_span: at..at + count,
        tokens: expanded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIALS: VisionTokens = VisionTokens {
        start: 900,
        pad: 901,
        end: 902,
    };

    #[test]
    fn one_bracketed_placeholder_expands_to_a_token_per_merged_patch() {
        let grid = ImageGrid { rows: 2, cols: 3 };
        let tokens = [10, 11, 900, 901, 902, 12];
        let expanded = expand_single_image(&tokens, SPECIALS, grid).unwrap();
        assert_eq!(
            expanded.tokens,
            vec![10, 11, 900, 901, 901, 901, 901, 901, 901, 902, 12]
        );
        assert_eq!(expanded.image_span, 3..9);
        assert_eq!(
            expanded.pieces,
            vec![
                PromptPiece::Text(3),
                PromptPiece::Image(grid),
                PromptPiece::Text(2),
            ]
        );
        // The pieces and the tokens must agree on length, or the positions
        // will be misassigned from the image onward.
        let piece_total: usize = expanded
            .pieces
            .iter()
            .map(|piece| match piece {
                PromptPiece::Text(n) => *n,
                PromptPiece::Image(g) => g.tokens(),
            })
            .sum();
        assert_eq!(piece_total, expanded.tokens.len());
    }

    #[test]
    fn a_prompt_without_a_placeholder_is_refused_not_guessed() {
        let grid = ImageGrid { rows: 1, cols: 1 };
        let error = expand_single_image(&[10, 11], SPECIALS, grid).unwrap_err();
        assert!(matches!(
            error,
            NluError::ImagePlaceholder { found: 0, bracketed: false, grid_tokens: 1 }
        ));
    }

    #[test]
    fn two_placeholders_are_refused_as_unsupported() {
        let grid = ImageGrid { rows: 1, cols: 1 };
        let tokens = [900, 901, 902, 900, 901, 902];
        let error = expand_single_image(&tokens, SPECIALS, grid).unwrap_err();
        assert!(matches!(error, NluError::ImagePlaceholder { found: 2, .. }));
    }

    #[test]
    fn an_unbracketed_placeholder_is_refused() {
        let grid = ImageGrid { rows: 1, cols: 1 };
        for tokens in [
            vec![901, 902],      // no start
            vec![900, 901],      // no end
            vec![901],           // bare
            vec![900, 5, 901, 902], // start not adjacent
        ] {
            let error = expand_single_image(&tokens, SPECIALS, grid).unwrap_err();
            assert!(
                matches!(error, NluError::ImagePlaceholder { found: 1, bracketed: false, .. }),
                "{tokens:?}"
            );
        }
    }
}
