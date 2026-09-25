//! Rope position triples for a prompt that mixes text and images.
//!
//! Text advances all three rope sections together, one index per token. An
//! image does not: its tokens share a temporal index and spread over the height
//! and width sections according to their place in the merged patch grid. That is
//! what the three sections are for, and it is why an image occupies fewer
//! position slots than it has tokens.
//!
//! Getting the advance wrong is not a visible failure. The model still runs and
//! still produces fluent text; it just places everything after the image at the
//! wrong distance.

/// The merged patch grid of one image, in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGrid {
    pub rows: usize,
    pub cols: usize,
}

impl ImageGrid {
    /// Tokens this image contributes to the sequence.
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }

    /// Position slots it consumes, which is not its token count.
    ///
    /// The height and width sections advance in parallel over the grid, so the
    /// image costs the longer of the two sides. A 16x16 grid is 256 tokens and
    /// 16 slots.
    pub fn position_span(&self) -> usize {
        self.rows.max(self.cols)
    }
}

/// One stretch of a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptPiece {
    /// This many text tokens.
    Text(usize),
    /// One image occupying its grid's worth of tokens.
    Image(ImageGrid),
}

/// Build a `(temporal, height, width)` triple for every token in order.
///
/// Text tokens get equal indices, which is the ordinary 1-D rope written as a
/// triple. Image tokens at grid cell `(row, col)` get `(start, start + row,
/// start + col)`, and the cursor then advances by [`ImageGrid::position_span`].
pub fn position_triples(pieces: &[PromptPiece]) -> Vec<[usize; 3]> {
    let capacity = pieces
        .iter()
        .map(|piece| match piece {
            PromptPiece::Text(n) => *n,
            PromptPiece::Image(grid) => grid.tokens(),
        })
        .sum();
    let mut triples = Vec::with_capacity(capacity);
    let mut next = 0usize;
    for piece in pieces {
        match piece {
            PromptPiece::Text(count) => {
                for offset in 0..*count {
                    let index = next + offset;
                    triples.push([index, index, index]);
                }
                next += count;
            }
            PromptPiece::Image(grid) => {
                for row in 0..grid.rows {
                    for col in 0..grid.cols {
                        triples.push([next, next + row, next + col]);
                    }
                }
                next += grid.position_span();
            }
        }
    }
    triples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_alone_is_the_ordinary_one_dimensional_rope() {
        let triples = position_triples(&[PromptPiece::Text(4)]);
        assert_eq!(triples, vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    }

    #[test]
    fn an_image_spreads_over_height_and_width_and_shares_one_temporal_index() {
        // A 2x3 grid: 6 tokens, 3 position slots.
        let grid = ImageGrid { rows: 2, cols: 3 };
        assert_eq!(grid.tokens(), 6);
        assert_eq!(grid.position_span(), 3);

        let triples = position_triples(&[PromptPiece::Text(2), PromptPiece::Image(grid)]);
        assert_eq!(&triples[..2], &[[0, 0, 0], [1, 1, 1]]);
        // Row-major from cursor 2: temporal fixed, height by row, width by col.
        assert_eq!(
            &triples[2..],
            &[
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4],
            ]
        );
    }

    #[test]
    fn an_image_costs_its_longer_side_in_positions_not_its_token_count() {
        let grid = ImageGrid { rows: 2, cols: 3 };
        let triples = position_triples(&[PromptPiece::Image(grid), PromptPiece::Text(2)]);
        assert_eq!(triples.len(), grid.tokens() + 2);
        // Six image tokens, but the text resumes at 3, not at 6. Advancing by
        // the token count instead would place every later token too far away.
        assert_eq!(&triples[grid.tokens()..], &[[3, 3, 3], [4, 4, 4]]);
    }

    #[test]
    fn a_single_cell_image_behaves_exactly_like_one_text_token() {
        let one = position_triples(&[
            PromptPiece::Text(1),
            PromptPiece::Image(ImageGrid { rows: 1, cols: 1 }),
            PromptPiece::Text(1),
        ]);
        assert_eq!(one, position_triples(&[PromptPiece::Text(3)]));
    }

    #[test]
    fn every_triple_is_produced_and_the_count_matches_the_tokens() {
        let pieces = [
            PromptPiece::Text(3),
            PromptPiece::Image(ImageGrid { rows: 4, cols: 4 }),
            PromptPiece::Text(2),
            PromptPiece::Image(ImageGrid { rows: 1, cols: 5 }),
        ];
        let expected: usize = 3 + 16 + 2 + 5;
        let triples = position_triples(&pieces);
        assert_eq!(triples.len(), expected);
        // Nothing may exceed the cursor's final value.
        let end = 3 + 4 + 2 + 5;
        assert!(
            triples
                .iter()
                .all(|[t, h, w]| *t < end && *h < end && *w < end)
        );
    }
}
