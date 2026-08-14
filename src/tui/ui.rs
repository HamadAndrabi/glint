//! Terminal UI rendering and widget layout via ratatui.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Gauge, List, ListItem, Paragraph, Tabs, Wrap,
};
use ratatui::Frame;

use super::app::{ActiveTab, App};

pub fn render(f: &mut Frame, app: &mut App) {
    let size = f.area();

    // Main vertical layout: Header, Content, Input & Status
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header & Tabs
            Constraint::Min(10),   // Active View
            Constraint::Length(4), // Input & Keybinds
        ])
        .split(size);

    render_header(f, app, chunks[0]);

    // Middle area: split if settings drawer is open
    if app.settings_open {
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(chunks[1]);

        render_main_tab(f, app, content_chunks[0]);
        render_settings_drawer(f, app, content_chunks[1]);
    } else {
        render_main_tab(f, app, chunks[1]);
    }

    render_bottom_input(f, app, chunks[2]);
}

fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let header_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(28), // Brand
            Constraint::Min(35),   // Tabs
            Constraint::Length(35), // Model Badges
        ])
        .split(area);

    // 1. Brand / Logo
    let brand = Paragraph::new(Line::from(vec![
        Span::styled("🐺 GLINT ", Style::default().fg(Color::Cyan).bold()),
        Span::styled("// CORE v0.2.0", Style::default().fg(Color::DarkGray)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(brand, header_chunks[0]);

    // 2. Tabs
    let tab_titles = vec![
        Line::from(" [1] 💬 DIALOGUE "),
        Line::from(" [2] 🧬 SCHEMA MATRIX "),
        Line::from(" [3] 📊 TELEMETRY "),
    ];
    let selected_tab = match app.active_tab {
        ActiveTab::Chat => 0,
        ActiveTab::StructuredLab => 1,
        ActiveTab::KvTelemetry => 2,
    };
    let tabs = Tabs::new(tab_titles)
        .select(selected_tab)
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .bold()
                .add_modifier(Modifier::UNDERLINED),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    f.render_widget(tabs, header_chunks[1]);

    // 3. Status Badges
    let status_text = format!(
        "{} | Ctx: {}",
        app.model_name, app.context_length
    );
    let badges = Paragraph::new(Line::from(vec![
        Span::styled("● ", Style::default().fg(if app.is_generating { Color::Yellow } else { Color::Cyan })),
        Span::styled(status_text, Style::default().fg(Color::Gray)),
    ]))
    .alignment(Alignment::Right)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(badges, header_chunks[2]);
}

fn render_main_tab(f: &mut Frame, app: &mut App, area: Rect) {
    match app.active_tab {
        ActiveTab::Chat => render_chat_view(f, app, area),
        ActiveTab::StructuredLab => render_lab_view(f, app, area),
        ActiveTab::KvTelemetry => render_telemetry_view(f, app, area),
    }
}

fn render_chat_view(f: &mut Frame, app: &mut App, area: Rect) {
    let mut items = Vec::new();

    for m in &app.messages {
        let (role_label, border_color, text_color) = match m.role.as_str() {
            "user" => (" [DIRECTIVE // USER] ", Color::Blue, Color::White),
            "assistant" => (" [GLINT // OPTIC-CORE] ", Color::Cyan, Color::White),
            "system" => (" [SYSTEM DIRECTIVE] ", Color::DarkGray, Color::Gray),
            _ => (" [PROMPT] ", Color::Gray, Color::White),
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled(role_label, Style::default().fg(border_color).bold()),
                if let (Some(tok_s), Some(ttft)) = (m.tok_per_sec, m.ttft_ms) {
                    Span::styled(
                        format!(" [⚡ {tok_s:.1} tok/s · ⏱ {ttft}ms TTFT]"),
                        Style::default().fg(Color::DarkGray),
                    )
                } else {
                    Span::raw("")
                },
            ]),
            Line::from(""),
        ];

        for text_line in m.content.lines() {
            if text_line.starts_with("```") {
                lines.push(Line::from(Span::styled(
                    text_line,
                    Style::default().fg(Color::Yellow),
                )));
            } else {
                lines.push(Line::from(Span::styled(text_line, Style::default().fg(text_color))));
            }
        }
        lines.push(Line::from(""));

        items.push(ListItem::new(lines));
    }

    // Streaming token buffer if currently generating
    if app.is_generating || !app.streaming_response.is_empty() {
        let mut lines = vec![
            Line::from(vec![
                Span::styled(" [GLINT // OPTIC-CORE] ", Style::default().fg(Color::Cyan).bold()),
                Span::styled(" [vector stream active...]", Style::default().fg(Color::Yellow)),
            ]),
            Line::from(""),
        ];
        for text_line in app.streaming_response.lines() {
            lines.push(Line::from(Span::styled(text_line, Style::default().fg(Color::White))));
        }
        lines.push(Line::from(Span::styled(" ▍", Style::default().fg(Color::Cyan))));
        items.push(ListItem::new(lines));
    }

    let list = List::new(items)
        .block(
            Block::default()
                .title(" Target Acquisition & Dialogue ")
                .title_style(Style::default().fg(Color::Cyan).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

    f.render_widget(list, area);
}

fn render_lab_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Mode selector
            Constraint::Min(8),    // Schema/Grammar Editor
            Constraint::Length(5), // Prompt input
        ])
        .split(chunks[0]);

    // 1. Lab Mode toggle
    let modes = vec![Line::from(" JSON Schema "), Line::from(" GBNF Grammar ")];
    let mode_tabs = Tabs::new(modes)
        .select(app.lab_mode)
        .highlight_style(Style::default().fg(Color::Magenta).bold())
        .block(
            Block::default()
                .title(" Constraint Type (Ctrl+T to toggle) ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        );
    f.render_widget(mode_tabs, left_chunks[0]);

    // 2. Schema / Grammar Editor
    let editor_title = if app.lab_mode == 0 {
        " JSON Schema Definition "
    } else {
        " GBNF Grammar Rules "
    };
    let editor_content = if app.lab_mode == 0 {
        &app.schema_input
    } else {
        &app.grammar_input
    };
    let editor = Paragraph::new(editor_content.as_str())
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .title(editor_title)
                .title_style(Style::default().fg(Color::Magenta).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(editor, left_chunks[1]);

    // 3. Lab Prompt
    let prompt_box = Paragraph::new(app.lab_prompt.as_str())
        .style(Style::default().fg(Color::Cyan))
        .block(
            Block::default()
                .title(" Lab Prompt (Press Ctrl+Enter to Run) ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(prompt_box, left_chunks[2]);

    // 4. Output Pane
    let output = Paragraph::new(app.lab_output.as_str())
        .style(Style::default().fg(Color::Green))
        .block(
            Block::default()
                .title(" Constrained Generation Stream ")
                .title_style(Style::default().fg(Color::Green).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(output, chunks[1]);
}

fn render_telemetry_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7), // Metric Gauges
            Constraint::Length(8), // KV Cache Grid & Memory
            Constraint::Min(6),    // Engine Spec Table
        ])
        .split(area);

    // 1. Performance Meters
    let meter_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(chunks[0]);

    // Tok/s speed gauge
    let tok_s_ratio = (app.current_tok_per_sec / 150.0).clamp(0.0, 1.0);
    let speed_gauge = Gauge::default()
        .block(
            Block::default()
                .title(" Generation Throughput ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(tok_s_ratio)
        .label(format!("{:.1} tok/s", app.current_tok_per_sec));
    f.render_widget(speed_gauge, meter_chunks[0]);

    // TTFT meter
    let ttft_label = format!("{} ms", app.current_ttft_ms);
    let ttft_p = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("⚡ TTFT: ", Style::default().fg(Color::Yellow).bold()),
            Span::styled(ttft_label, Style::default().fg(Color::White).bold()),
        ]),
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .title(" Time-to-First-Token ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(ttft_p, meter_chunks[1]);

    // Total tokens
    let total_p = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("📦 Tokens: ", Style::default().fg(Color::Green).bold()),
            Span::styled(
                format!("{}", app.total_tokens_generated),
                Style::default().fg(Color::White).bold(),
            ),
        ]),
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .title(" Session Tokens ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(total_p, meter_chunks[2]);

    // 2. KV Cache Heatmap / Memory
    let ctx_ratio = (app.kv_used_tokens as f64 / app.context_length as f64).min(1.0);
    let kv_gauge = Gauge::default()
        .block(
            Block::default()
                .title(" KV Cache Window Utilization ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .gauge_style(Style::default().fg(Color::Magenta))
        .ratio(ctx_ratio)
        .label(format!("{}/{} tokens ({:.1}%)", app.kv_used_tokens, app.context_length, ctx_ratio * 100.0));
    f.render_widget(kv_gauge, chunks[1]);

    // 3. Engine Info
    let info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Model Path:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(app.model_path.display().to_string(), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Architecture:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(app.architecture.clone(), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Context Window: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} tokens", app.context_length), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Layers / Heads: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} layers / {} heads", app.block_count, app.head_count), Style::default().fg(Color::White)),
        ]),
    ])
    .block(
        Block::default()
            .title(" Engine Specification ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(info, chunks[2]);
}

fn render_settings_drawer(f: &mut Frame, app: &App, area: Rect) {
    let settings = [
        format!("Temperature:     {:.2}", app.temperature),
        format!("Top-P:           {:.2}", app.top_p),
        format!("Top-K:           {}", app.top_k),
        format!("Repeat Penalty:  {:.2}", app.repeat_penalty),
        format!("Max Tokens:      {}", app.max_tokens),
        format!("System Prompt:   {}", if app.system_prompt.len() > 18 { format!("{}...", &app.system_prompt[..18]) } else { app.system_prompt.clone() }),
    ];

    let items: Vec<ListItem> = settings
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let is_sel = i == app.selected_setting;
            let style = if is_sel {
                Style::default().fg(Color::Cyan).bold().add_modifier(Modifier::REVERSED)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(vec![
                Span::styled(if is_sel { " ❯ " } else { "   " }, Style::default().fg(Color::Cyan)),
                Span::styled(s, style),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Parameters (Tab to close) ")
            .title_style(Style::default().fg(Color::Cyan).bold())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(list, area);
}

fn render_bottom_input(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(1)])
        .split(area);

    let input_title = if app.is_generating {
        " [Generating... press Esc to interrupt] "
    } else {
        " Prompt (Enter to submit, Shift+Enter for newline) "
    };

    let border_color = if app.is_generating {
        Color::Yellow
    } else {
        Color::Cyan
    };

    let input_widget = Paragraph::new(format!("❯ {}", app.input_text))
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .title(input_title)
                .title_style(Style::default().fg(border_color).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(border_color)),
        );
    f.render_widget(input_widget, chunks[0]);

    // Keybind Hints
    let hints = Line::from(vec![
        Span::styled("Enter ", Style::default().fg(Color::Cyan).bold()),
        Span::styled("Send | ", Style::default().fg(Color::DarkGray)),
        Span::styled("Tab ", Style::default().fg(Color::Cyan).bold()),
        Span::styled("Settings Drawer | ", Style::default().fg(Color::DarkGray)),
        Span::styled("F1-F3 ", Style::default().fg(Color::Cyan).bold()),
        Span::styled("Tabs | ", Style::default().fg(Color::DarkGray)),
        Span::styled("Esc ", Style::default().fg(Color::Cyan).bold()),
        Span::styled("Stop | ", Style::default().fg(Color::DarkGray)),
        Span::styled("Ctrl+C ", Style::default().fg(Color::Red).bold()),
        Span::styled("Quit", Style::default().fg(Color::DarkGray)),
    ]);
    let hints_widget = Paragraph::new(hints).alignment(Alignment::Center);
    f.render_widget(hints_widget, chunks[1]);
}
