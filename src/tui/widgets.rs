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
use crate::tui::{AkhTui, CommandPalette, ModelPicker};

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

/// Main TUI layout: header, body (message stream + optional right sidebar),
/// statusline footer, and input box (with a live cursor).
///
/// Tool/agent "steps" (`ToolResult` messages — the model's tool calls and their
/// results) collapse into a single one-line summary unless `show_thinking` is
/// on, so a multi-tool turn doesn't bury the final answer. Toggle with Tab.
///
/// The right sidebar (Context / Goals / Capabilities) is drawn when
/// [`AkhTui::show_sidebar`](crate::tui::AkhTui) is set *and* the terminal is wide
/// enough; toggle with `ctrl+b`. The model picker and command palette overlays,
/// when open, are drawn centered ON TOP of everything.
#[cfg(feature = "acp")]
pub fn render(frame: &mut Frame, tui: &AkhTui) {
    // Vertical: header (1) · body (fill) · footer (1) · input (3).
    let [header_area, body_area, footer_area, input_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    // The body splits into the message stream and an optional right sidebar. The
    // sidebar is hidden when toggled off or when the terminal is too narrow to
    // host it without crowding the chat.
    const SIDEBAR_W: u16 = 34;
    const MIN_W_FOR_SIDEBAR: u16 = 96;
    let (messages_area, sidebar_area) =
        if tui.show_sidebar && body_area.width >= MIN_W_FOR_SIDEBAR {
            let [m, s] =
                Layout::horizontal([Constraint::Fill(1), Constraint::Length(SIDEBAR_W)])
                    .areas(body_area);
            (m, Some(s))
        } else {
            (body_area, None)
        };

    render_header(frame, header_area, tui);
    render_messages(frame, messages_area, tui);
    if let Some(area) = sidebar_area {
        render_sidebar(frame, area, tui);
    }
    render_footer(frame, footer_area, tui);
    render_input(frame, input_area, tui);

    // Overlays (mutually exclusive in practice) draw centered on top.
    if let Some(picker) = tui.picker.as_ref() {
        render_model_picker(frame, picker);
    }
    if let Some(palette) = tui.palette.as_ref() {
        render_command_palette(frame, palette);
    }
}

/// The one-line header: the `akh-medu` badge, the session title (or workspace
/// name until the first prompt), and a compact key-hint legend.
#[cfg(feature = "acp")]
fn render_header(frame: &mut Frame, area: Rect, tui: &AkhTui) {
    let title = tui.session_title.as_deref().unwrap_or(tui.workspace.as_str());
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " akh-medu ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(" :: {title}  ")),
        Span::styled(
            "ctrl+b sidebar · ctrl+p cmds · Tab thinking",
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    frame.render_widget(header, area);
}

/// The scrollable message stream. Runs of tool/agent steps collapse into one
/// summary line unless `show_thinking` is on (opencode-style).
#[cfg(feature = "acp")]
fn render_messages(frame: &mut Frame, area: Rect, tui: &AkhTui) {
    let messages = &tui.messages;
    let mut lines: Vec<Line> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        if matches!(messages[i], AkhMessage::ToolResult { .. }) {
            let start = i;
            while i < messages.len() && matches!(messages[i], AkhMessage::ToolResult { .. }) {
                i += 1;
            }
            if tui.show_thinking {
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
                    Span::styled("  (Tab to expand)", Style::default().fg(Color::DarkGray)),
                ]));
            }
        } else {
            lines.extend(message_to_lines(&messages[i]));
            i += 1;
        }
    }

    // Clamp scroll to the rendered line count (collapsing changes the count).
    let scroll = tui.scroll_offset.min(lines.len().saturating_sub(1));
    let visible_lines = if lines.len() > scroll {
        &lines[scroll..]
    } else {
        &[]
    };
    let messages_widget = Paragraph::new(visible_lines.to_vec())
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(messages_widget, area);
}

/// The input box + a live cursor at the end of the typed text. While a turn is
/// in flight, the border title becomes an animated spinner instead of the
/// prompt.
#[cfg(feature = "acp")]
fn render_input(frame: &mut Frame, area: Rect, tui: &AkhTui) {
    let title: Line = if tui.awaiting {
        Line::from(Span::styled(
            format!(" {} ", spinner_text(tui.spinner_frame)),
            Style::default()
                .fg(Color::Rgb(212, 175, 55)) // gilded "Egyptian gold"
                .add_modifier(Modifier::BOLD),
        ))
    } else {
        Line::from(" > ")
    };
    let input_widget = Paragraph::new(tui.input_buffer.as_str())
        .block(Block::default().borders(Borders::ALL).title(title))
        .style(Style::default().fg(Color::White));
    frame.render_widget(input_widget, area);
    let cursor_x = (area.x + 1 + tui.input_buffer.chars().count() as u16)
        .min(area.x + area.width.saturating_sub(2));
    frame.set_cursor_position((cursor_x, area.y + 1));
}

/// The one-line statusline footer: connection health, then a `·`-joined run of
/// model · effort · git-branch · daemon pid.
#[cfg(feature = "acp")]
fn render_footer(frame: &mut Frame, area: Rect, tui: &AkhTui) {
    let disconnected = tui.remote.is_disconnected();
    let (status, conn_color) = if disconnected {
        ("⨯ disconnected", Color::Red)
    } else {
        ("● connected", Color::Green)
    };

    let mut parts: Vec<String> = Vec::new();
    if let Some(m) = &tui.model_label {
        parts.push(m.clone());
    }
    if let Some(e) = &tui.effort_label {
        parts.push(format!("effort:{e}"));
    }
    if let Some(b) = &tui.git_branch {
        parts.push(b.clone());
    }
    parts.push(format!("pid {}", tui.info.pid));

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {status}  "), Style::default().fg(conn_color)),
        Span::styled(parts.join(" · "), Style::default().fg(Color::DarkGray)),
    ]));
    frame.render_widget(footer, area);
}

/// The right sidebar: stacked Context / Goals / Capabilities panels.
///
/// Phase A renders the scaffold; the live data feed (token usage, goals, tools,
/// skills) arrives with the `_akh/status` poller in Phase C, at which point these
/// placeholders are replaced with real rows from the cached snapshot.
#[cfg(feature = "acp")]
fn render_sidebar(frame: &mut Frame, area: Rect, tui: &AkhTui) {
    let [ctx_area, goals_area, caps_area] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Fill(1),
        Constraint::Length(6),
    ])
    .areas(area);

    let dim = Style::default().fg(Color::DarkGray);
    let turn = if tui.awaiting {
        "● turn in flight"
    } else {
        "○ idle"
    };

    let context = Paragraph::new(vec![
        Line::from(Span::styled("tokens: —", dim)),
        Line::from(Span::styled(turn, dim)),
    ])
    .block(panel_block("Context"));
    frame.render_widget(context, ctx_area);

    let goals = Paragraph::new(vec![
        Line::from(Span::styled("no goals loaded", dim)),
        Line::from(Span::styled("(live in Phase C)", dim)),
    ])
    .block(panel_block("Goals"))
    .wrap(Wrap { trim: false });
    frame.render_widget(goals, goals_area);

    let caps = Paragraph::new(vec![
        Line::from(Span::styled("tools:  —", dim)),
        Line::from(Span::styled("skills: —", dim)),
        Line::from(Span::styled("· awaiting daemon ·", dim)),
    ])
    .block(panel_block("Capabilities"));
    frame.render_widget(caps, caps_area);
}

/// A bordered sidebar-panel block with a cyan, bold title.
#[cfg(feature = "acp")]
fn panel_block(title: &str) -> Block<'static> {
    Block::default().borders(Borders::ALL).title(Span::styled(
        format!(" {title} "),
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ))
}

/// Draw the centered model-picker overlay: a bordered, scrollable list of model
/// names with the selected row highlighted. The list is the picker's
/// type-ahead-*filtered* view, and the border title echoes the active filter so
/// the user can see what they're typing. The area underneath is cleared first
/// ([`Clear`]) so the chat does not bleed through.
#[cfg(feature = "acp")]
pub fn render_model_picker(frame: &mut Frame, picker: &ModelPicker) {
    let area = centered_rect(60, 60, frame.area());
    // Clear whatever is underneath before drawing the popup.
    frame.render_widget(Clear, area);

    let visible = picker.filtered();
    let items: Vec<ListItem> = if visible.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "(no models match)",
            Style::default().fg(Color::DarkGray),
        )))]
    } else {
        visible
            .iter()
            .map(|m| ListItem::new(Line::from(Span::raw((*m).clone()))))
            .collect()
    };

    let title = if picker.filter.is_empty() {
        " Select model (↑/↓, type to filter, Enter, Esc) ".to_string()
    } else {
        format!(" Select model — filter: {}▏ ", picker.filter)
    };

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");

    let mut state = ListState::default();
    if !visible.is_empty() {
        state.select(Some(picker.selected.min(visible.len() - 1)));
    }
    frame.render_stateful_widget(list, area, &mut state);
}

/// Draw the centered command-palette overlay (the `ctrl+p` popup): a bordered,
/// scrollable list of `/command — description` rows, type-ahead-filtered, with
/// the selected row highlighted. Mirrors [`render_model_picker`]; the area
/// underneath is cleared first ([`Clear`]).
#[cfg(feature = "acp")]
pub fn render_command_palette(frame: &mut Frame, palette: &CommandPalette) {
    let area = centered_rect(60, 60, frame.area());
    frame.render_widget(Clear, area);

    let visible = palette.filtered();
    let items: Vec<ListItem> = if visible.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "(no commands match)",
            Style::default().fg(Color::DarkGray),
        )))]
    } else {
        visible
            .iter()
            .map(|c| {
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("/{:<8}", c.name),
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!(" {}", c.desc), Style::default().fg(Color::Gray)),
                ]))
            })
            .collect()
    };

    let title = if palette.filter.is_empty() {
        " Commands (↑/↓, type to filter, Enter, Esc) ".to_string()
    } else {
        format!(" Commands — filter: {}▏ ", palette.filter)
    };

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");

    let mut state = ListState::default();
    if !visible.is_empty() {
        state.select(Some(palette.selected.min(visible.len() - 1)));
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

/// Render one message into one-or-more display lines.
///
/// Narrative answers are passed through the Markdown renderer (headings, lists,
/// code, emphasis) and so can expand to several lines; every other message kind
/// is a single line via [`message_to_line`].
#[cfg(feature = "acp")]
fn message_to_lines(msg: &AkhMessage) -> Vec<Line<'static>> {
    match msg {
        AkhMessage::Narrative { text, .. } => {
            let lines = crate::tui::markdown::render_markdown(text);
            if lines.is_empty() {
                vec![Line::default()]
            } else {
                lines
            }
        }
        other => vec![message_to_line(other)],
    }
}

/// The progress spinner shown in the input border while a turn is in flight.
///
/// A static ankh (☥ — the most universally rendered Egyptian glyph) paired with
/// a braille wheel for smooth motion across terminals. For a terminal with an
/// Egyptian Hieroglyphs font, swap [`SPINNER_WHEEL`] for a hieroglyph cycle such
/// as `["𓂀", "𓂁", "𓂃", "𓂀"]`.
#[cfg(feature = "acp")]
fn spinner_text(frame: usize) -> String {
    let tick = SPINNER_WHEEL[frame % SPINNER_WHEEL.len()];
    format!("☥ {tick} thinking…")
}

/// Braille wheel frames for the [`spinner_text`] animation.
#[cfg(feature = "acp")]
const SPINNER_WHEEL: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
