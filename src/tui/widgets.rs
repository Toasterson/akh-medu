//! TUI widget rendering: message list, input area, status bar.

use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::message::AkhMessage;

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
        AkhMessage::Gap { entity, description } => Line::from(vec![
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
            Span::styled(
                format!("[{grammar}] "),
                Style::default().fg(Color::Magenta),
            ),
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
                Span::styled(
                    goal.clone(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
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
    }
}

/// Main TUI layout rendering.
pub fn render(
    frame: &mut Frame,
    workspace: &str,
    grammar: &str,
    messages: &[AkhMessage],
    input: &str,
    scroll_offset: usize,
    cycle_count: u64,
    symbol_count: usize,
    goal_count: usize,
) {
    let [header_area, messages_area, input_area, status_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(3),
        Constraint::Length(1),
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
        Span::raw(format!(" :: workspace: {workspace} :: grammar: {grammar} ")),
    ]));
    frame.render_widget(header, header_area);

    // Messages area.
    let lines: Vec<Line> = messages.iter().map(message_to_line).collect();
    let visible_lines = if lines.len() > scroll_offset {
        &lines[scroll_offset..]
    } else {
        &[]
    };
    let messages_widget = Paragraph::new(visible_lines.to_vec())
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(messages_widget, messages_area);

    // Input area.
    let input_widget = Paragraph::new(input)
        .block(Block::default().borders(Borders::ALL).title(" > "))
        .style(Style::default().fg(Color::White));
    frame.render_widget(input_widget, input_area);

    // Status bar.
    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            format!(" cycles: {cycle_count} "),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("| "),
        Span::styled(
            format!("symbols: {symbol_count} "),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("| "),
        Span::styled(
            format!("goals: {goal_count} active "),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("| "),
        Span::styled(
            format!("grammar: {grammar} "),
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    frame.render_widget(status, status_area);
}
