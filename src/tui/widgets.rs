//! TUI widget rendering: message list, input area, status bar.

#[cfg(feature = "acp")]
use ratatui::Frame;
#[cfg(feature = "acp")]
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
#[cfg(feature = "acp")]
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use crate::message::AkhMessage;
#[cfg(feature = "acp")]
use crate::tui::ModelPicker;

/// Render a single AkhMessage as a styled Line.
pub fn message_to_line(msg: &AkhMessage) -> Line<'static> {
    match msg {
        AkhMessage::Fact {
            text, confidence, ..
        } => {
            let mut spans = vec![
                Span::styled("[fact] ", Style::default().fg(Color::Cyan)),
                Span::raw(text.clone()),
            ];
            if let Some(c) = confidence {
                spans.push(Span::styled(
                    format!(" ({c:.2})"),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            Line::from(spans)
        }
        AkhMessage::Reasoning { step, expression } => {
            let mut spans = vec![
                Span::styled("[reasoning] ", Style::default().fg(Color::Yellow)),
                Span::raw(step.clone()),
            ];
            if let Some(expr) = expression {
                spans.push(Span::styled(
                    format!(" :: {expr}"),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            Line::from(spans)
        }
        AkhMessage::Gap {
            entity,
            description,
        } => Line::from(vec![
            Span::styled("[gap] ", Style::default().fg(Color::Red)),
            Span::styled(
                entity.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(": {description}")),
        ]),
        AkhMessage::ToolResult {
            tool,
            success,
            output,
        } => {
            let (label, color) = if *success {
                ("ok", Color::Green)
            } else {
                ("FAIL", Color::Red)
            };
            Line::from(vec![
                Span::styled(format!("[{tool}:{label}] "), Style::default().fg(color)),
                Span::raw(output.clone()),
            ])
        }
        AkhMessage::Narrative { text, grammar } => Line::from(vec![
            Span::styled(format!("[{grammar}] "), Style::default().fg(Color::Magenta)),
            Span::raw(text.clone()),
        ]),
        AkhMessage::System { text } => Line::from(vec![Span::styled(
            text.clone(),
            Style::default().fg(Color::DarkGray),
        )]),
        AkhMessage::Error {
            code,
            message,
            help,
        } => {
            let mut spans = vec![
                Span::styled(
                    format!("[error:{code}] "),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::raw(message.clone()),
            ];
            if let Some(h) = help {
                spans.push(Span::styled(
                    format!(" (help: {h})"),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            Line::from(spans)
        }
        AkhMessage::GoalProgress {
            goal,
            status,
            detail,
        } => {
            let mut spans = vec![
                Span::styled("[goal] ", Style::default().fg(Color::Blue)),
                Span::styled(goal.clone(), Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(format!(" — {status}")),
            ];
            if let Some(d) = detail {
                spans.push(Span::styled(
                    format!(": {d}"),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            Line::from(spans)
        }
        AkhMessage::Prompt { question } => Line::from(vec![
            Span::styled("? ", Style::default().fg(Color::Green)),
            Span::raw(question.clone()),
        ]),
        AkhMessage::AuditLog {
            id,
            kind,
            summary,
            ..
        } => Line::from(vec![
            Span::styled(
                format!("[audit:{kind}] "),
                Style::default().fg(Color::Magenta),
            ),
            Span::styled(
                format!("#{id} "),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(summary.clone()),
        ]),
    }
}

/// Main TUI layout: header, message stream, input box (with a live cursor).
///
/// Tool/agent "steps" (`ToolResult` messages — the model's tool calls and their
/// results) collapse into a single one-line summary unless `show_thinking` is
/// on, so a multi-tool turn doesn't bury the final answer. Toggle with Tab.
///
/// When `picker` is `Some`, the model-picker overlay is drawn centered ON TOP of
/// the main layout (see [`render_model_picker`]).
#[cfg(feature = "acp")]
pub fn render(
    frame: &mut Frame,
    workspace: &str,
    messages: &[AkhMessage],
    input: &str,
    scroll_offset: usize,
    show_thinking: bool,
    picker: Option<&ModelPicker>,
) {
    let [header_area, messages_area, input_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    // Header.
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " akh-medu ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(" :: workspace: {workspace}  ")),
        Span::styled(
            if show_thinking {
                "[Tab: hide thinking]"
            } else {
                "[Tab: show thinking]"
            },
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    frame.render_widget(header, header_area);

    // Message stream. Collapse runs of tool/agent steps into one summary line
    // unless the user expanded them (opencode-style).
    let mut lines: Vec<Line> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        if matches!(messages[i], AkhMessage::ToolResult { .. }) {
            let start = i;
            while i < messages.len() && matches!(messages[i], AkhMessage::ToolResult { .. }) {
                i += 1;
            }
            if show_thinking {
                for m in &messages[start..i] {
                    lines.push(message_to_line(m));
                }
            } else {
                let count = i - start;
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("▸ {count} step{}", if count == 1 { "" } else { "s" }),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(
                        "  (Tab to expand)",
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
        } else {
            lines.push(message_to_line(&messages[i]));
            i += 1;
        }
    }

    // Clamp scroll to the rendered line count (collapsing changes the count).
    let scroll = scroll_offset.min(lines.len().saturating_sub(1));
    let visible_lines = if lines.len() > scroll {
        &lines[scroll..]
    } else {
        &[]
    };
    let messages_widget = Paragraph::new(visible_lines.to_vec())
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(messages_widget, messages_area);

    // Input box + a live cursor at the end of the typed text.
    let input_widget = Paragraph::new(input)
        .block(Block::default().borders(Borders::ALL).title(" > "))
        .style(Style::default().fg(Color::White));
    frame.render_widget(input_widget, input_area);
    let cursor_x = (input_area.x + 1 + input.chars().count() as u16)
        .min(input_area.x + input_area.width.saturating_sub(2));
    frame.set_cursor_position((cursor_x, input_area.y + 1));

    // Model picker overlays the main layout when open.
    if let Some(picker) = picker {
        render_model_picker(frame, picker);
    }
}

/// Draw the centered model-picker overlay: a bordered, scrollable list of model
/// names with the selected row highlighted. The area underneath is cleared
/// first ([`Clear`]) so the chat does not bleed through.
#[cfg(feature = "acp")]
pub fn render_model_picker(frame: &mut Frame, picker: &ModelPicker) {
    let area = centered_rect(60, 60, frame.area());
    // Clear whatever is underneath before drawing the popup.
    frame.render_widget(Clear, area);

    let items: Vec<ListItem> = picker
        .models
        .iter()
        .map(|m| ListItem::new(Line::from(Span::raw(m.clone()))))
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Select model (↑/↓, Enter, Esc) "),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");

    let mut state = ListState::default();
    if !picker.models.is_empty() {
        state.select(Some(picker.selected.min(picker.models.len() - 1)));
    }
    frame.render_stateful_widget(list, area, &mut state);
}

/// Compute a Rect centered within `area`, sized to `percent_x` × `percent_y`
/// of it.
#[cfg(feature = "acp")]
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let [_, vmid, _] = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .areas(area);
    let [_, mid, _] = Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .areas(vmid);
    mid
}
