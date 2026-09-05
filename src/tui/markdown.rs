//! Pragmatic Markdown → ratatui renderer for the chat message stream.
//!
//! LLM output is mostly CommonMark-flavoured Markdown: headings, bullet and
//! numbered lists, fenced code blocks, blockquotes, horizontal rules, and the
//! usual inline emphasis. This module renders that subset into owned
//! [`ratatui::text::Line`]s so the TUI can drop them straight into a
//! `Paragraph`. It is deliberately a hand-written scanner — no regex, no
//! parser crate — and it favours *passing unknown markup through verbatim*
//! over guessing wrong and dropping characters.
//!
//! Everything produced is `'static`: spans clone their text out of `input`, so
//! the returned lines outlive the borrow.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Render a Markdown string into owned ratatui lines for display in the
/// message stream. Supports a pragmatic subset of CommonMark common in LLM
/// output (headings, lists, code fences, blockquotes, horizontal rules, and
/// inline bold/italic/code/strikethrough). Unrecognized markup is passed
/// through verbatim rather than dropped.
pub fn render_markdown(input: &str) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    // When `Some`, we are inside a fenced code block accumulating raw lines.
    // The fence lines themselves are dropped; inner lines render as DarkGray
    // with no inline parsing.
    let mut in_code_block = false;

    for raw in input.split('\n') {
        if is_fence(raw) {
            // Toggle: opening fence (optional language tag is ignored) or
            // closing fence. Either way the fence line is not rendered.
            in_code_block = !in_code_block;
            continue;
        }

        if in_code_block {
            lines.push(code_line(raw));
            continue;
        }

        lines.push(render_block_line(raw));
    }

    lines
}

/// Render one non-fence, non-code block-level line.
fn render_block_line(raw: &str) -> Line<'static> {
    // Blank line → empty line.
    if raw.trim().is_empty() {
        return Line::default();
    }

    // Horizontal rule: exactly 3+ of `-`, `*`, or `_`, optionally surrounded
    // by whitespace, and nothing else.
    if is_horizontal_rule(raw) {
        return Line::from(Span::styled(
            "─".repeat(40),
            Style::default().fg(Color::DarkGray),
        ));
    }

    // ATX heading: 1–6 `#` then a space.
    if let Some((level, text)) = parse_heading(raw) {
        // Higher levels are slightly dimmer; level 1–2 stay bright cyan.
        let color = if level <= 2 {
            Color::Cyan
        } else {
            Color::LightBlue
        };
        return Line::from(Span::styled(
            text.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    }

    // Blockquote: `>` then one optional space. Prefix a dim gutter, then
    // inline-render the remainder.
    if let Some(rest) = parse_blockquote(raw) {
        let mut spans = vec![Span::styled("▏ ", Style::default().fg(Color::DarkGray))];
        spans.extend(render_inline(rest));
        return Line::from(spans);
    }

    // Unordered list item: ^\s*[-*+]\s+ . Preserve leading indent as spaces,
    // swap the marker for a bullet, inline-render the remainder.
    if let Some((indent, rest)) = parse_unordered_item(raw) {
        let mut spans = vec![Span::raw(format!("{indent}• "))];
        spans.extend(render_inline(rest));
        return Line::from(spans);
    }

    // Ordered list item: ^\s*\d+\.\s+ . Keep the number + dot, inline-render
    // the remainder.
    if let Some((indent, number, rest)) = parse_ordered_item(raw) {
        let mut spans = vec![Span::raw(format!("{indent}{number}. "))];
        spans.extend(render_inline(rest));
        return Line::from(spans);
    }

    // Plain text → inline-render the whole line.
    Line::from(render_inline(raw))
}

/// A line of code inside a fence: DarkGray, no inline parsing.
fn code_line(raw: &str) -> Line<'static> {
    Line::from(Span::styled(
        raw.to_string(),
        Style::default().fg(Color::DarkGray),
    ))
}

/// Is `raw` a code-fence line? A fence is a line whose trimmed content starts
/// with three backticks (an optional language tag may follow on an opening
/// fence; closing fences are typically bare).
fn is_fence(raw: &str) -> bool {
    raw.trim_start().starts_with("```")
}

/// Is `raw` a horizontal rule — 3+ of the same `-`/`*`/`_` char, optionally
/// surrounded by whitespace, with nothing else on the line?
fn is_horizontal_rule(raw: &str) -> bool {
    let trimmed = raw.trim();
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if first != '-' && first != '*' && first != '_' {
        return false;
    }
    // The whole trimmed string must be that one repeated char, at least 3 of
    // them. (`first` is one; require ≥2 more all equal to `first`.)
    let count = 1 + chars.clone().take_while(|&c| c == first).count();
    count >= 3 && chars.all(|c| c == first)
}

/// Parse an ATX heading. Returns `(level, heading_text)` where `level` is
/// 1–6 and the text has the markers (and the single separating space)
/// stripped. A trailing run of `#` (closing ATX sequence) is also trimmed.
fn parse_heading(raw: &str) -> Option<(usize, &str)> {
    let hashes = raw.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let after = &raw[hashes..];
    // Must be a space (or tab) immediately after the run of `#`.
    let mut chars = after.chars();
    match chars.next() {
        Some(' ') | Some('\t') => {}
        _ => return None,
    }
    // Strip the single separating whitespace, then any leading/trailing
    // whitespace and a closing run of `#`.
    let text = after[1..].trim().trim_end_matches('#').trim_end();
    Some((hashes, text))
}

/// Parse a blockquote line. Returns the content with the leading `>` and one
/// optional following space removed. Leading whitespace before `>` is
/// tolerated.
fn parse_blockquote(raw: &str) -> Option<&str> {
    let trimmed = raw.trim_start();
    let rest = trimmed.strip_prefix('>')?;
    // Drop one optional following space.
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

/// Parse an unordered list item (`-`, `*`, or `+` marker followed by at least
/// one space). Returns `(leading_indent_as_spaces, remainder)`.
fn parse_unordered_item(raw: &str) -> Option<(String, &str)> {
    let indent_len = raw.len() - raw.trim_start().len();
    let indent = &raw[..indent_len];
    let body = &raw[indent_len..];
    let mut chars = body.chars();
    let marker = chars.next()?;
    if marker != '-' && marker != '*' && marker != '+' {
        return None;
    }
    // Require at least one space after the marker.
    let after_marker = &body[marker.len_utf8()..];
    if !after_marker.starts_with(' ') && !after_marker.starts_with('\t') {
        return None;
    }
    let rest = after_marker.trim_start();
    // Normalise the indent to spaces so tabs don't render oddly.
    let indent_spaces = " ".repeat(indent.chars().count());
    Some((indent_spaces, rest))
}

/// Parse an ordered list item (`\d+.` followed by at least one space). Returns
/// `(leading_indent_as_spaces, number_string, remainder)`.
fn parse_ordered_item(raw: &str) -> Option<(String, &str, &str)> {
    let indent_len = raw.len() - raw.trim_start().len();
    let indent = &raw[..indent_len];
    let body = &raw[indent_len..];

    let digit_len = body.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_len == 0 {
        return None;
    }
    let number = &body[..digit_len];
    let after_digits = &body[digit_len..];
    let after_dot = after_digits.strip_prefix('.')?;
    // Require at least one space after the dot.
    if !after_dot.starts_with(' ') && !after_dot.starts_with('\t') {
        return None;
    }
    let rest = after_dot.trim_start();
    let indent_spaces = " ".repeat(indent.chars().count());
    Some((indent_spaces, number, rest))
}

/// Scan inline text into styled spans. Handles `**bold**`/`__bold__`,
/// `*italic*`/`_italic_`, `` `code` ``, and `~~strike~~`. A marker with no
/// matching closer on the same line is emitted literally (we never eat the
/// rest of the line on a dangling marker). Backtick code spans take precedence
/// over emphasis: nothing inside `` `...` `` is re-parsed.
fn render_inline(text: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    // Plain text accumulated since the last emitted styled span.
    let mut plain = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];

        // Inline code: `...` — highest precedence, no inner parsing.
        if c == '`' {
            if let Some(end) = find_char(&chars, i + 1, '`') {
                let content: String = chars[i + 1..end].iter().collect();
                flush_plain(&mut plain, &mut spans);
                spans.push(Span::styled(
                    content,
                    Style::default().fg(Color::LightGreen),
                ));
                i = end + 1;
                continue;
            }
            // No closer → literal backtick.
            plain.push(c);
            i += 1;
            continue;
        }

        // Bold: ** ... ** or __ ... __ (check the double-marker before single).
        if (c == '*' || c == '_') && i + 1 < n && chars[i + 1] == c {
            let marker = c;
            if let Some(end) = find_double(&chars, i + 2, marker) {
                let content: String = chars[i + 2..end].iter().collect();
                flush_plain(&mut plain, &mut spans);
                spans.push(Span::styled(
                    content,
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                i = end + 2;
                continue;
            }
            // No closer → emit the two marker chars literally, advance past
            // both so we don't reinterpret them as italic.
            plain.push(marker);
            plain.push(marker);
            i += 2;
            continue;
        }

        // Italic: * ... * or _ ... _ (single marker).
        if c == '*' || c == '_' {
            if let Some(end) = find_char(&chars, i + 1, c) {
                let content: String = chars[i + 1..end].iter().collect();
                flush_plain(&mut plain, &mut spans);
                spans.push(Span::styled(
                    content,
                    Style::default().add_modifier(Modifier::ITALIC),
                ));
                i = end + 1;
                continue;
            }
            // No closer → literal marker.
            plain.push(c);
            i += 1;
            continue;
        }

        // Strikethrough: ~~ ... ~~ .
        if c == '~' && i + 1 < n && chars[i + 1] == '~' {
            if let Some(end) = find_double(&chars, i + 2, '~') {
                let content: String = chars[i + 2..end].iter().collect();
                flush_plain(&mut plain, &mut spans);
                spans.push(Span::styled(
                    content,
                    Style::default().add_modifier(Modifier::CROSSED_OUT),
                ));
                i = end + 2;
                continue;
            }
            plain.push('~');
            plain.push('~');
            i += 2;
            continue;
        }

        // Ordinary character.
        plain.push(c);
        i += 1;
    }

    flush_plain(&mut plain, &mut spans);

    // An empty input yields a single empty raw span so callers always get at
    // least one span to inspect.
    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }

    spans
}

/// Emit the accumulated plain text (if any) as a raw span and reset the buffer.
fn flush_plain(plain: &mut String, spans: &mut Vec<Span<'static>>) {
    if !plain.is_empty() {
        spans.push(Span::raw(std::mem::take(plain)));
    }
}

/// Find the index of the next occurrence of `target` in `chars` at or after
/// `from`. Used to locate single-char closers.
fn find_char(chars: &[char], from: usize, target: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == target)
}

/// Find the index of the next *doubled* `target` (`target` at `j` and `j+1`)
/// at or after `from`. Returns the index of the first of the two chars.
fn find_double(chars: &[char], from: usize, target: char) -> Option<usize> {
    let n = chars.len();
    let mut j = from;
    while j + 1 < n {
        if chars[j] == target && chars[j + 1] == target {
            return Some(j);
        }
        j += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: pull the (content, style) pairs out of a line's spans.
    fn spans_of<'a>(line: &'a Line<'static>) -> Vec<(&'a str, Style)> {
        line.spans
            .iter()
            .map(|s| (s.content.as_ref(), s.style))
            .collect()
    }

    #[test]
    fn bold_produces_bold_span_with_stripped_text() {
        let spans = render_inline("**hi**");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "hi");
        assert!(spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn bold_underscore_variant() {
        let spans = render_inline("__hi__");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "hi");
        assert!(spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn italic_produces_italic_span_with_stripped_text() {
        let spans = render_inline("*hi*");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "hi");
        assert!(spans[0].style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn italic_underscore_variant() {
        let spans = render_inline("_hi_");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "hi");
        assert!(spans[0].style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn inline_code_produces_light_green_span() {
        let spans = render_inline("`code`");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "code");
        assert_eq!(spans[0].style.fg, Some(Color::LightGreen));
    }

    #[test]
    fn strikethrough_produces_crossed_out_span() {
        let spans = render_inline("~~gone~~");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "gone");
        assert!(spans[0].style.add_modifier.contains(Modifier::CROSSED_OUT));
    }

    #[test]
    fn unmatched_asterisk_preserved_literally() {
        let spans = render_inline("a * b");
        // No closer for the single `*` → the whole thing stays as plain text.
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "a * b");
        assert_eq!(spans[0].style, Style::default());
    }

    #[test]
    fn unmatched_double_asterisk_preserved_literally() {
        let spans = render_inline("a ** b");
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "a ** b");
        // Nothing should be bold.
        assert!(
            spans
                .iter()
                .all(|s| !s.style.add_modifier.contains(Modifier::BOLD))
        );
    }

    #[test]
    fn heading_yields_bold_cyan_span_with_hashes_stripped() {
        let lines = render_markdown("# Title");
        assert_eq!(lines.len(), 1);
        let s = spans_of(&lines[0]);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].0, "Title");
        assert_eq!(s[0].1.fg, Some(Color::Cyan));
        assert!(s[0].1.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn deep_heading_still_bold() {
        let lines = render_markdown("### Sub");
        let s = spans_of(&lines[0]);
        assert_eq!(s[0].0, "Sub");
        assert!(s[0].1.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn unordered_item_starts_with_bullet() {
        let lines = render_markdown("- item");
        assert_eq!(lines.len(), 1);
        let s = spans_of(&lines[0]);
        assert_eq!(s[0].0, "• ");
        assert_eq!(s[1].0, "item");
    }

    #[test]
    fn unordered_item_other_markers() {
        for src in ["* item", "+ item"] {
            let lines = render_markdown(src);
            let s = spans_of(&lines[0]);
            assert_eq!(s[0].0, "• ", "marker for {src:?}");
            assert_eq!(s[1].0, "item");
        }
    }

    #[test]
    fn unordered_item_preserves_indent() {
        let lines = render_markdown("  - nested");
        let s = spans_of(&lines[0]);
        assert_eq!(s[0].0, "  • ");
        assert_eq!(s[1].0, "nested");
    }

    #[test]
    fn ordered_item_keeps_number() {
        let lines = render_markdown("1. item");
        assert_eq!(lines.len(), 1);
        let s = spans_of(&lines[0]);
        assert_eq!(s[0].0, "1. ");
        assert_eq!(s[1].0, "item");
    }

    #[test]
    fn fenced_code_block_renders_inner_dark_gray_and_drops_fences() {
        let input = "```rust\nlet x = 1;\n*not italic*\n```";
        let lines = render_markdown(input);
        // Two inner lines, no fence lines.
        assert_eq!(lines.len(), 2);
        let l0 = spans_of(&lines[0]);
        assert_eq!(l0.len(), 1);
        assert_eq!(l0[0].0, "let x = 1;");
        assert_eq!(l0[0].1.fg, Some(Color::DarkGray));
        // Inline markup inside code is NOT parsed.
        let l1 = spans_of(&lines[1]);
        assert_eq!(l1.len(), 1);
        assert_eq!(l1[0].0, "*not italic*");
        assert_eq!(l1[0].1.fg, Some(Color::DarkGray));
    }

    #[test]
    fn unterminated_code_block_still_renders() {
        let input = "```\nstill here";
        let lines = render_markdown(input);
        assert_eq!(lines.len(), 1);
        let s = spans_of(&lines[0]);
        assert_eq!(s[0].0, "still here");
        assert_eq!(s[0].1.fg, Some(Color::DarkGray));
    }

    #[test]
    fn plain_text_returns_one_intact_line() {
        let lines = render_markdown("just some text");
        assert_eq!(lines.len(), 1);
        let s = spans_of(&lines[0]);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].0, "just some text");
        assert_eq!(s[0].1, Style::default());
    }

    #[test]
    fn mixed_inline_splits_into_right_span_sequence() {
        // "a **b** c `d`"
        let spans = render_inline("a **b** c `d`");
        let parts: Vec<&str> = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(parts, vec!["a ", "b", " c ", "d"]);
        // Plain, bold, plain, code.
        assert_eq!(spans[0].style, Style::default());
        assert!(spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(spans[2].style, Style::default());
        assert_eq!(spans[3].style.fg, Some(Color::LightGreen));
    }

    #[test]
    fn blockquote_has_dim_gutter() {
        let lines = render_markdown("> quoted");
        let s = spans_of(&lines[0]);
        assert_eq!(s[0].0, "▏ ");
        assert_eq!(s[0].1.fg, Some(Color::DarkGray));
        assert_eq!(s[1].0, "quoted");
    }

    #[test]
    fn horizontal_rule_renders_dark_gray_bar() {
        for src in ["---", "***", "___", "  ----  "] {
            let lines = render_markdown(src);
            assert_eq!(lines.len(), 1, "rule for {src:?}");
            let s = spans_of(&lines[0]);
            assert_eq!(s.len(), 1);
            assert_eq!(s[0].0.chars().count(), 40);
            assert_eq!(s[0].1.fg, Some(Color::DarkGray));
        }
    }

    #[test]
    fn blank_line_is_empty() {
        let lines = render_markdown("a\n\nb");
        assert_eq!(lines.len(), 3);
        assert!(lines[1].spans.is_empty());
    }

    #[test]
    fn code_span_suppresses_emphasis_inside() {
        let spans = render_inline("`a*b*c`");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "a*b*c");
        assert_eq!(spans[0].style.fg, Some(Color::LightGreen));
    }

    #[test]
    fn not_a_heading_without_space() {
        // "#tag" is not an ATX heading; should pass through as plain text.
        let lines = render_markdown("#tag");
        let s = spans_of(&lines[0]);
        assert_eq!(s[0].0, "#tag");
        assert!(!s[0].1.add_modifier.contains(Modifier::BOLD));
    }
}
