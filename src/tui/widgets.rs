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
            id, kind, summary, ..
        } => Line::from(vec![
            Span::styled(
                format!("[audit:{kind}] "),
                Style::default().fg(Color::Magenta),
            ),
            Span::styled(format!("#{id} "), Style::default().fg(Color::DarkGray)),
            Span::raw(summary.clone()),
        ]),
        AkhMessage::ToolCallStarted {
            tool_id,
            kind,
            title,
            locations,
        } => {
            use crate::message::ToolKind;
            let kind_color = match kind {
                ToolKind::Execute => Color::Red,
                ToolKind::Edit => Color::Yellow,
                ToolKind::Fetch => Color::Cyan,
                ToolKind::Search => Color::Blue,
                ToolKind::Other => Color::Magenta,
            };
            let mut spans = vec![
                Span::styled(format!("[{kind:?}] "), Style::default().fg(kind_color)),
                Span::styled(format!("{tool_id} "), Style::default().fg(Color::DarkGray)),
                Span::raw(title.clone()),
            ];
            if !locations.is_empty() {
                spans.push(Span::styled(
                    format!(" ({})", locations.join(", ")),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            Line::from(spans)
        }
        AkhMessage::ToolCallCompleted {
            tool_id,
            success,
            output,
            diff,
        } => {
            let status_color = if *success { Color::Green } else { Color::Red };
            let status = if *success { "ok" } else { "failed" };
            let mut spans = vec![
                Span::styled(format!("[{status}] "), Style::default().fg(status_color)),
                Span::styled(format!("{tool_id} "), Style::default().fg(Color::DarkGray)),
            ];
            if let Some(out) = output {
                spans.push(Span::raw(out.clone()));
            }
            if let Some(d) = diff {
                spans.push(Span::styled(
                    format!("\n{d}"),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            Line::from(spans)
        }
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
    let (messages_area, sidebar_area) = if tui.show_sidebar && body_area.width >= MIN_W_FOR_SIDEBAR
    {
        let [m, s] = Layout::horizontal([Constraint::Fill(1), Constraint::Length(SIDEBAR_W)])
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
    let title = tui
        .session_title
        .as_deref()
        .unwrap_or(tui.workspace.as_str());
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

/// The right sidebar: stacked Context / Goals / Capabilities panels, populated
/// from the latest `_akh/status` snapshot (the status poller, Phase C). Before
/// the first poll lands (`snapshot` is `None`) the panels show a "connecting"
/// placeholder; under a busy turn the snapshot is `stale` and agent-derived rows
/// fall back while engine-derived skills stay live.
#[cfg(feature = "acp")]
fn render_sidebar(frame: &mut Frame, area: Rect, tui: &AkhTui) {
    let [
        ctx_area,
        goals_area,
        reasoning_area,
        mind_area,
        evidence_area,
        caps_area,
    ] = Layout::vertical([
        Constraint::Length(4), // Context
        Constraint::Fill(1),   // Goals (flexes with the terminal)
        Constraint::Length(5), // Reasoning
        Constraint::Length(4), // Mind
        Constraint::Length(5), // Evidence
        Constraint::Length(5), // Capabilities
    ])
    .areas(area);

    let dim = Style::default().fg(Color::DarkGray);
    let snap = tui.snapshot.as_ref();

    // ── Context: token usage + cycle + activity ──
    let context_lines: Vec<Line> = match snap {
        Some(s) => {
            let tok_line = match s.usage.as_ref().filter(|u| u.total_tokens > 0) {
                Some(u) => match u.context_window.filter(|w| *w > 0) {
                    Some(win) => {
                        let pct = (u.context_tokens.min(win) * 100) / win;
                        format!("ctx {}/{win} {pct}%", u.context_tokens)
                    }
                    None => format!("ctx {} · {} tok", u.context_tokens, u.total_tokens),
                },
                None => "tokens: —".to_string(),
            };
            let cycle = s.cycle.map_or_else(|| "—".to_string(), |c| c.to_string());
            let activity = if s.stale {
                "● agent busy"
            } else if tui.awaiting {
                "● turn in flight"
            } else {
                "○ idle"
            };
            vec![
                Line::from(Span::styled(tok_line, dim)),
                Line::from(Span::styled(format!("cycle: {cycle}  {activity}"), dim)),
            ]
        }
        None => vec![
            Line::from(Span::styled("tokens: —", dim)),
            Line::from(Span::styled("· connecting ·", dim)),
        ],
    };
    frame.render_widget(
        Paragraph::new(context_lines).block(panel_block("Context")),
        ctx_area,
    );

    // ── Goals: active-plan progress header + status-marked goal rows ──
    let goal_lines: Vec<Line> = match snap {
        Some(s) if !s.goals.is_empty() => {
            let mut lines: Vec<Line> = Vec::new();
            if let Some(p) = &s.plan {
                let done = p
                    .steps
                    .iter()
                    .filter(|st| st.status.as_str() == "completed")
                    .count();
                lines.push(Line::from(Span::styled(
                    format!("▶ {}/{} {}", done, p.steps.len(), truncate(&p.strategy, 18)),
                    Style::default().fg(Color::Magenta),
                )));
            }
            let room = (goals_area.height.saturating_sub(2) as usize).saturating_sub(lines.len());
            for g in s.goals.iter().take(room) {
                let (marker, color) = goal_marker(g.status_label());
                lines.push(Line::from(vec![
                    Span::styled(format!("{marker} "), Style::default().fg(color)),
                    Span::raw(truncate(&g.description, 26)),
                ]));
            }
            lines
        }
        Some(s) if s.stale => vec![Line::from(Span::styled("agent busy…", dim))],
        Some(_) => vec![Line::from(Span::styled("no goals", dim))],
        None => vec![Line::from(Span::styled("· connecting ·", dim))],
    };
    frame.render_widget(
        Paragraph::new(goal_lines)
            .block(panel_block("Goals"))
            .wrap(Wrap { trim: false }),
        goals_area,
    );

    // ── Reasoning: recent working-memory trace (newest first) ──
    let reasoning_lines: Vec<Line> = match snap {
        Some(s) if !s.reasoning.is_empty() => s
            .reasoning
            .iter()
            .take(3)
            .map(|r| {
                Line::from(vec![
                    Span::styled(
                        format!("{} ", reason_tag(&r.kind)),
                        Style::default().fg(reason_color(&r.kind)),
                    ),
                    Span::raw(truncate(&r.content, 24)),
                ])
            })
            .collect(),
        Some(s) if s.stale => vec![Line::from(Span::styled("agent busy…", dim))],
        Some(_) => vec![Line::from(Span::styled("no recent steps", dim))],
        None => vec![Line::from(Span::styled("· connecting ·", dim))],
    };
    frame.render_widget(
        Paragraph::new(reasoning_lines).block(panel_block("Reasoning")),
        reasoning_area,
    );

    // ── Mind: knowledge-graph size + active microtheory (beyond OpenCode) ──
    let mind_lines: Vec<Line> = match snap {
        Some(s) => {
            let kg_line = match s.kg.as_ref() {
                Some(k) => format!("{} sym · {} tri", k.symbols, k.triples),
                None => "KG: —".to_string(),
            };
            let active = s.active_project.as_deref().unwrap_or("—");
            vec![
                Line::from(Span::styled(kg_line, dim)),
                Line::from(Span::styled(
                    format!("active: {}", truncate(active, 22)),
                    dim,
                )),
            ]
        }
        None => vec![Line::from(Span::styled("· connecting ·", dim))],
    };
    frame.render_widget(
        Paragraph::new(mind_lines).block(panel_block("Mind")),
        mind_area,
    );

    // ── Evidence: tracked claims + the most uncertain (belief interval) ──
    let evidence_lines: Vec<Line> = match snap {
        Some(s) => match &s.evidence {
            Some(e) => {
                let mut lines = vec![Line::from(Span::styled(
                    format!("{} claims", e.claims),
                    dim,
                ))];
                for c in e.uncertain.iter().take(2) {
                    let pct = |v: f32| (v * 100.0).round() as u32;
                    lines.push(Line::from(vec![
                        Span::raw(truncate(&c.claim, 14)),
                        Span::styled(
                            format!(" {}-{}%", pct(c.belief), pct(c.plausibility)),
                            Style::default().fg(verdict_color(&c.verdict)),
                        ),
                    ]));
                }
                lines
            }
            None if s.stale => vec![Line::from(Span::styled("agent busy…", dim))],
            None => vec![Line::from(Span::styled("no evidence", dim))],
        },
        None => vec![Line::from(Span::styled("· connecting ·", dim))],
    };
    frame.render_widget(
        Paragraph::new(evidence_lines).block(panel_block("Evidence")),
        evidence_area,
    );

    // ── Capabilities: tool + skill counts, then a few skill names ──
    let cap_lines: Vec<Line> = match snap {
        Some(s) => {
            let mut lines = vec![
                Line::from(Span::styled(format!("tools:  {}", s.tools.len()), dim)),
                Line::from(Span::styled(format!("skills: {}", s.skills.len()), dim)),
            ];
            for sk in s.skills.iter().take(3) {
                lines.push(Line::from(Span::styled(
                    format!("  · {}", truncate(&sk.name, 24)),
                    dim,
                )));
            }
            lines
        }
        None => vec![
            Line::from(Span::styled("tools:  —", dim)),
            Line::from(Span::styled("skills: —", dim)),
            Line::from(Span::styled("· connecting ·", dim)),
        ],
    };
    frame.render_widget(
        Paragraph::new(cap_lines).block(panel_block("Capabilities")),
        caps_area,
    );
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

/// Marker glyph + colour for a goal's status label (the serialized `GoalStatus`
/// variant name).
#[cfg(feature = "acp")]
fn goal_marker(status: &str) -> (&'static str, Color) {
    match status {
        "Active" => ("●", Color::Green),
        "Completed" => ("✓", Color::Cyan),
        "Failed" | "Cancelled" => ("✗", Color::Red),
        "Pending" | "Proposed" => ("○", Color::Yellow),
        "Suspended" | "Dormant" => ("◌", Color::DarkGray),
        _ => ("·", Color::Gray),
    }
}

/// Short tag for a reasoning entry's working-memory kind.
#[cfg(feature = "acp")]
fn reason_tag(kind: &str) -> &'static str {
    match kind {
        "Observation" => "obs",
        "Decision" => "dec",
        "GoalUpdate" => "goal",
        "Inference" => "inf",
        "ToolResult" => "tool",
        _ => "·",
    }
}

/// Colour for a reasoning entry's kind.
#[cfg(feature = "acp")]
fn reason_color(kind: &str) -> Color {
    match kind {
        "Decision" => Color::Yellow,
        "Inference" => Color::Magenta,
        "ToolResult" => Color::Green,
        "GoalUpdate" => Color::Blue,
        _ => Color::DarkGray,
    }
}

/// Colour for a claim's Dempster-Shafer verdict. Matches on substrings so a
/// change to the exact label wording can't silently drop the colour.
#[cfg(feature = "acp")]
fn verdict_color(verdict: &str) -> Color {
    let v = verdict.to_ascii_lowercase();
    if v.contains("well") || v.contains("support") {
        Color::Green
    } else if v.contains("false") {
        Color::Red
    } else if v.contains("conflict") {
        Color::Magenta
    } else if v.contains("insufficient") {
        Color::DarkGray
    } else {
        Color::Yellow // plausible / uncertain
    }
}

/// Truncate to `max` chars with a trailing ellipsis (char-aware, never splits a
/// multibyte boundary).
#[cfg(feature = "acp")]
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    } else {
        s.to_string()
    }
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
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
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
