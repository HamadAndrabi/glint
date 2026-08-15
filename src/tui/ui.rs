//! Terminal UI rendering and widget layout via ratatui.
//! Glint TUI — Obsidian / Amber / Mint Brutalist Aesthetic.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph, Tabs, Wrap};
use ratatui::Frame;

use super::app::{ActiveTab, App};

// ── Design Tokens & Color Palette ─────────────────────────────────────────────
pub const COLOR_BG: Color = Color::Rgb(24, 25, 30); // #18191E
pub const COLOR_SURFACE_DIM: Color = Color::Rgb(17, 19, 29); // #11131D
pub const COLOR_SURFACE_HIGH: Color = Color::Rgb(39, 41, 53); // #272935
pub const COLOR_SURFACE_VARIANT: Color = Color::Rgb(50, 52, 64); // #323440

pub const COLOR_PRIMARY: Color = Color::Rgb(255, 193, 116); // #ffc174 (Warm Gold)
pub const COLOR_PRIMARY_CONTAINER: Color = Color::Rgb(245, 158, 11); // #f59e0b (Amber)
pub const COLOR_SECONDARY: Color = Color::Rgb(78, 222, 163); // #4edea3 (Mint Green)

pub const COLOR_BORDER: Color = Color::Rgb(63, 65, 77); // #3F414D (Muted Slate)
pub const COLOR_BORDER_WARM: Color = Color::Rgb(83, 68, 52); // #534434 (Warm Slate)

pub const COLOR_TEXT: Color = Color::Rgb(212, 213, 220); // #D4D5DC (Soft White)
pub const COLOR_MUTED: Color = Color::Rgb(160, 142, 122); // #a08e7a (Warm Slate / Ochre)
pub const COLOR_DIM: Color = Color::Rgb(116, 116, 121); // #747479

pub const COLOR_CODE_BG: Color = Color::Rgb(31, 32, 42); // #1F202A

pub fn render(f: &mut Frame, app: &mut App) {
    let size = f.area();

    // Main vertical layout: Header, Content View, Input & Keybinds
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header & Tabs
            Constraint::Min(8),    // Active View (Chat / Lab / Telemetry)
            Constraint::Length(5), // Input Dock & Keybind Footer
        ])
        .split(size);

    render_header(f, app, chunks[0]);

    // Middle area: split if settings drawer is open
    if app.settings_open {
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(chunks[1]);

        render_main_tab(f, app, content_chunks[0]);
        render_settings_drawer(f, app, content_chunks[1]);
    } else {
        render_main_tab(f, app, chunks[1]);
    }

    render_bottom_input(f, app, chunks[2]);
}

// ── Top Navigation Header ─────────────────────────────────────────────────────
fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let header_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(22), // Brand Mark
            Constraint::Min(32),    // Centered Tabs
            Constraint::Length(40), // Hardware & Engine Telemetry Chips
        ])
        .split(area);

    // 1. Brand Mark
    let brand = Paragraph::new(Line::from(vec![
        Span::styled("✦ GLINT ", Style::default().fg(COLOR_PRIMARY).bold()),
        Span::styled(
            concat!("v", env!("CARGO_PKG_VERSION")),
            Style::default().fg(COLOR_MUTED),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_BORDER)),
    );
    f.render_widget(brand, header_chunks[0]);

    // 2. Center Tabs
    let tab_titles = vec![
        Line::from(" [1] CHAT "),
        Line::from(" [2] LAB "),
        Line::from(" [3] TELEMETRY "),
    ];
    let selected_tab = match app.active_tab {
        ActiveTab::Chat => 0,
        ActiveTab::StructuredLab => 1,
        ActiveTab::KvTelemetry => 2,
    };
    let tabs = Tabs::new(tab_titles)
        .select(selected_tab)
        .style(Style::default().fg(COLOR_MUTED))
        .highlight_style(
            Style::default()
                .fg(COLOR_PRIMARY)
                .bold()
                .add_modifier(Modifier::UNDERLINED),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(COLOR_BORDER_WARM)),
        );
    f.render_widget(tabs, header_chunks[1]);

    // 3. Right Status Chips
    let kv_used_mb = app.kv_bytes_used() as f64 / (1024.0 * 1024.0);
    let badges = Paragraph::new(Line::from(vec![
        Span::styled(
            "● ",
            Style::default().fg(if app.is_generating {
                COLOR_PRIMARY_CONTAINER
            } else {
                COLOR_SECONDARY
            }),
        ),
        Span::styled("[ LOCAL SIMD ] ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            format!("[ KV: {kv_used_mb:.1}M ]"),
            Style::default().fg(COLOR_TEXT),
        ),
    ]))
    .alignment(Alignment::Right)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_BORDER)),
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

// ── Chat View ─────────────────────────────────────────────────────────────────
fn render_chat_view(f: &mut Frame, app: &mut App, area: Rect) {
    let mut items = Vec::new();

    // If no user/assistant turns yet, render quick empty state guidance.
    // `App::new` seeds a system turn, so `messages` is never actually empty.
    let has_conversation = app.messages.iter().any(|m| m.role != "system");
    if !has_conversation && app.streaming_response.is_empty() {
        let empty_lines = vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                "✦ Glint Local Inference",
                Style::default().fg(COLOR_PRIMARY).bold(),
            )]),
            Line::from(vec![Span::styled(
                "Private, instant, zero-dependency autoregressive generation.",
                Style::default().fg(COLOR_MUTED),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("• Press ", Style::default().fg(COLOR_DIM)),
                Span::styled("Enter", Style::default().fg(COLOR_PRIMARY).bold()),
                Span::styled(" to submit prompt.", Style::default().fg(COLOR_DIM)),
            ]),
            Line::from(vec![
                Span::styled("• Press ", Style::default().fg(COLOR_DIM)),
                Span::styled("Tab / Ctrl+S", Style::default().fg(COLOR_PRIMARY).bold()),
                Span::styled(
                    " to open Parameters drawer.",
                    Style::default().fg(COLOR_DIM),
                ),
            ]),
            Line::from(vec![
                Span::styled("• Press ", Style::default().fg(COLOR_DIM)),
                Span::styled("F1 / F2 / F3", Style::default().fg(COLOR_PRIMARY).bold()),
                Span::styled(
                    " to switch between Chat, Structured Lab, and Telemetry.",
                    Style::default().fg(COLOR_DIM),
                ),
            ]),
        ];
        items.push(ListItem::new(empty_lines));
    }

    for m in &app.messages {
        let is_user = m.role == "user";
        let is_system = m.role == "system";

        if is_system {
            // Render concise system turn
            let sys_lines = vec![
                Line::from(vec![
                    Span::styled("⚙ System: ", Style::default().fg(COLOR_MUTED).bold()),
                    Span::styled(&m.content, Style::default().fg(COLOR_DIM)),
                ]),
                Line::from(""),
            ];
            items.push(ListItem::new(sys_lines));
            continue;
        }

        let mut lines = Vec::new();

        if is_user {
            // User Header
            lines.push(Line::from(vec![Span::styled(
                "You",
                Style::default().fg(COLOR_MUTED).bold(),
            )]));
            for line in m.content.lines() {
                lines.push(Line::from(vec![
                    Span::styled("  ", Style::default()),
                    Span::styled(line, Style::default().fg(COLOR_TEXT)),
                ]));
            }
            lines.push(Line::from(""));
        } else {
            // Assistant Header with Performance Pill
            let mut header_spans = vec![Span::styled(
                "✦ Glint",
                Style::default().fg(COLOR_PRIMARY).bold(),
            )];
            if let (Some(tok_s), Some(ttft)) = (m.tok_per_sec, m.ttft_ms) {
                let token_str = m
                    .token_count
                    .map(|c| format!(" · {c} tokens"))
                    .unwrap_or_default();
                header_spans.push(Span::styled(
                    format!("  [ {tok_s:.1} tok/s · {ttft}ms TTFT{token_str} ]"),
                    Style::default().fg(COLOR_MUTED),
                ));
            }
            lines.push(Line::from(header_spans));

            // Format body with syntax highlights & bullet points
            let mut in_code_block = false;
            for line in m.content.lines() {
                let trimmed = line.trim_start();
                // Strip a leading bullet marker by char, never by byte — `•` is multi-byte.
                let bullet_rest = ["•", "-", "*"]
                    .iter()
                    .find_map(|marker| trimmed.strip_prefix(marker));

                if trimmed.starts_with("```") {
                    in_code_block = !in_code_block;
                    lines.push(Line::from(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled(line, Style::default().fg(COLOR_MUTED)),
                    ]));
                } else if in_code_block {
                    // Code block line highlighting
                    lines.push(format_code_line(line));
                } else if let Some(rest) = bullet_rest {
                    // Bullet list item
                    lines.push(Line::from(vec![
                        Span::styled("  •", Style::default().fg(COLOR_PRIMARY).bold()),
                        Span::styled(rest, Style::default().fg(COLOR_TEXT)),
                    ]));
                } else {
                    lines.push(Line::from(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled(line, Style::default().fg(COLOR_TEXT)),
                    ]));
                }
            }
            lines.push(Line::from(""));
        }

        items.push(ListItem::new(lines));
    }

    // Streaming response buffer
    if app.is_generating || !app.streaming_response.is_empty() {
        let mut lines = vec![Line::from(vec![
            Span::styled("✦ Glint", Style::default().fg(COLOR_PRIMARY).bold()),
            Span::styled(
                "  [ generating... ]",
                Style::default().fg(COLOR_PRIMARY_CONTAINER),
            ),
        ])];
        for line in app.streaming_response.lines() {
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(line, Style::default().fg(COLOR_TEXT)),
            ]));
        }
        // Blinking amber cursor
        lines.push(Line::from(vec![Span::styled(
            "  ▍",
            Style::default().fg(COLOR_PRIMARY),
        )]));
        lines.push(Line::from(""));
        items.push(ListItem::new(lines));
    }

    let list = List::new(items).block(
        Block::default()
            .title(" Chat Session ")
            .title_style(Style::default().fg(COLOR_PRIMARY).bold())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_BORDER)),
    );

    f.render_widget(list, area);
}

/// Helper function to format code syntax in TUI
fn format_code_line(line: &str) -> Line<'_> {
    let mut spans = vec![Span::styled("    ", Style::default())];
    let trimmed = line.trim();

    if trimmed.starts_with("//") || trimmed.starts_with("#") {
        spans.push(Span::styled(line, Style::default().fg(COLOR_MUTED)));
    } else {
        // Highlight standard language keywords
        let words = line.split_inclusive(char::is_whitespace);
        for word in words {
            let pure_word = word.trim();
            if matches!(
                pure_word,
                "def"
                    | "fn"
                    | "let"
                    | "mut"
                    | "pub"
                    | "struct"
                    | "enum"
                    | "impl"
                    | "return"
                    | "import"
                    | "from"
                    | "class"
                    | "async"
                    | "await"
            ) {
                spans.push(Span::styled(
                    word,
                    Style::default().fg(COLOR_SECONDARY).bold(),
                ));
            } else if matches!(
                pure_word,
                "true" | "false" | "None" | "Some" | "Ok" | "Err" | "nil"
            ) {
                spans.push(Span::styled(
                    word,
                    Style::default().fg(COLOR_PRIMARY_CONTAINER),
                ));
            } else if pure_word.starts_with('"') || pure_word.starts_with('\'') {
                spans.push(Span::styled(word, Style::default().fg(COLOR_PRIMARY)));
            } else {
                spans.push(Span::styled(word, Style::default().fg(COLOR_TEXT)));
            }
        }
    }

    Line::from(spans)
}

// ── Structured Lab View ───────────────────────────────────────────────────────
fn render_lab_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Mode Selector
            Constraint::Min(8),    // Schema / Grammar Code
            Constraint::Length(4), // Prompt Box
        ])
        .split(chunks[0]);

    // 1. Lab Mode toggle
    let modes = vec![
        Line::from(" [1] JSON Schema "),
        Line::from(" [2] GBNF Grammar "),
    ];
    let mode_tabs = Tabs::new(modes)
        .select(app.lab_mode)
        .highlight_style(Style::default().fg(COLOR_PRIMARY).bold())
        .block(
            Block::default()
                .title(" Mode (Ctrl+T to toggle) ")
                .title_style(Style::default().fg(COLOR_MUTED))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(COLOR_BORDER)),
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
        .style(Style::default().fg(COLOR_TEXT))
        .block(
            Block::default()
                .title(editor_title)
                .title_style(Style::default().fg(COLOR_PRIMARY).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(COLOR_BORDER_WARM)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(editor, left_chunks[1]);

    // 3. Lab Prompt
    let prompt_box = Paragraph::new(format!("> {}", app.lab_prompt.as_str()))
        .style(Style::default().fg(COLOR_PRIMARY))
        .block(
            Block::default()
                .title(" Prompt (Ctrl+Enter to Execute) ")
                .title_style(Style::default().fg(COLOR_MUTED))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(COLOR_BORDER)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(prompt_box, left_chunks[2]);

    // 4. Output Pane (Guaranteed Structured Output)
    let output_text = if app.lab_output.is_empty() {
        "// Constrained tokens conforming strictly to schema will appear here..."
    } else {
        app.lab_output.as_str()
    };
    let output = Paragraph::new(output_text)
        .style(Style::default().fg(COLOR_SECONDARY))
        .block(
            Block::default()
                .title(" Guaranteed Structured Output ")
                .title_style(Style::default().fg(COLOR_SECONDARY).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(COLOR_BORDER)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(output, chunks[1]);
}

// ── Telemetry View ────────────────────────────────────────────────────────────
fn render_telemetry_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7), // Performance Metric Meters
            Constraint::Length(7), // KV Cache Window Bar
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
                .title(" Throughput (tok/s) ")
                .title_style(Style::default().fg(COLOR_PRIMARY).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(COLOR_BORDER)),
        )
        .gauge_style(Style::default().fg(COLOR_PRIMARY_CONTAINER))
        .ratio(tok_s_ratio)
        .label(format!("{:.1} tok/s", app.current_tok_per_sec));
    f.render_widget(speed_gauge, meter_chunks[0]);

    // TTFT meter
    let ttft_label = format!("{} ms", app.current_ttft_ms);
    let ttft_p = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("TTFT: ", Style::default().fg(COLOR_PRIMARY).bold()),
            Span::styled(ttft_label, Style::default().fg(COLOR_TEXT).bold()),
        ]),
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .title(" Time-to-First-Token ")
            .title_style(Style::default().fg(COLOR_MUTED))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_BORDER)),
    );
    f.render_widget(ttft_p, meter_chunks[1]);

    // Total tokens
    let total_p = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Tokens: ", Style::default().fg(COLOR_SECONDARY).bold()),
            Span::styled(
                format!("{}", app.total_tokens_generated),
                Style::default().fg(COLOR_TEXT).bold(),
            ),
        ]),
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .title(" Generated Tokens ")
            .title_style(Style::default().fg(COLOR_MUTED))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_BORDER)),
    );
    f.render_widget(total_p, meter_chunks[2]);

    // 2. KV Cache Window Utilization
    let ctx_ratio = (app.kv_used_tokens as f64 / app.context_length as f64).min(1.0);
    let kv_gauge = Gauge::default()
        .block(
            Block::default()
                .title(" KV Cache Window Residency ")
                .title_style(Style::default().fg(COLOR_SECONDARY).bold())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(COLOR_BORDER_WARM)),
        )
        .gauge_style(Style::default().fg(COLOR_SECONDARY))
        .ratio(ctx_ratio)
        .label(format!(
            "{}/{} tokens ({:.1}%)",
            app.kv_used_tokens,
            app.context_length,
            ctx_ratio * 100.0
        ));
    f.render_widget(kv_gauge, chunks[1]);

    // 3. Engine Info
    let info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Model:          ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                app.model_name.clone(),
                Style::default().fg(COLOR_PRIMARY).bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("Architecture:   ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                app.architecture.clone(),
                Style::default().fg(COLOR_SECONDARY),
            ),
        ]),
        Line::from(vec![
            Span::styled("Context Window: ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                format!("{} tokens", app.context_length),
                Style::default().fg(COLOR_TEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("Layers / Heads: ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                format!("{} layers / {} heads", app.block_count, app.head_count),
                Style::default().fg(COLOR_TEXT),
            ),
        ]),
    ])
    .block(
        Block::default()
            .title(" Engine Specification ")
            .title_style(Style::default().fg(COLOR_MUTED))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_BORDER)),
    );
    f.render_widget(info, chunks[2]);
}

/// Truncate to at most `max` characters, appending an ellipsis when clipped.
/// Counts chars rather than bytes so multi-byte text never splits mid-codepoint.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let head: String = s.chars().take(max).collect();
        format!("{head}...")
    } else {
        s.to_string()
    }
}

// ── Settings Drawer (When Open) ───────────────────────────────────────────────
fn render_settings_drawer(f: &mut Frame, app: &App, area: Rect) {
    let settings = [
        format!("Temperature:     {:.2}", app.temperature),
        format!("Top-P:           {:.2}", app.top_p),
        format!("Top-K:           {}", app.top_k),
        format!("Repeat Penalty:  {:.2}", app.repeat_penalty),
        format!("Max Tokens:      {}", app.max_tokens),
        format!(
            "System Prompt:   {}",
            truncate_chars(&app.system_prompt, 16)
        ),
    ];

    let items: Vec<ListItem> = settings
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let is_sel = i == app.selected_setting;
            let style = if is_sel {
                Style::default()
                    .fg(COLOR_PRIMARY)
                    .bold()
                    .add_modifier(Modifier::REVERSED)
            } else {
                Style::default().fg(COLOR_TEXT)
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    if is_sel { " ❯ " } else { "   " },
                    Style::default().fg(COLOR_PRIMARY),
                ),
                Span::styled(s, style),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Parameters (Tab to close) ")
            .title_style(Style::default().fg(COLOR_PRIMARY).bold())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(COLOR_PRIMARY)),
    );

    f.render_widget(list, area);
}

// ── Bottom Input Dock & Keybind Footer ────────────────────────────────────────
fn render_bottom_input(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(2)])
        .split(area);

    let input_title = if app.is_generating {
        " Message Glint (Generating... press Esc to stop) "
    } else {
        " Message Glint (Enter to send, Ctrl+S for Settings) "
    };

    let border_color = if app.is_generating {
        COLOR_PRIMARY_CONTAINER
    } else {
        COLOR_PRIMARY
    };

    // Calculate prompt token estimation
    let est_tok = (app.input_text.len() / 4).max(1);
    let max_ctx = app.context_length;

    let input_widget = Paragraph::new(Line::from(vec![
        Span::styled("> ", Style::default().fg(COLOR_PRIMARY).bold()),
        Span::styled(
            if app.input_text.is_empty() {
                "Type your prompt here..."
            } else {
                &app.input_text
            },
            if app.input_text.is_empty() {
                Style::default().fg(COLOR_BORDER_WARM)
            } else {
                Style::default().fg(COLOR_TEXT)
            },
        ),
    ]))
    .block(
        Block::default()
            .title(input_title)
            .title_style(Style::default().fg(border_color).bold())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color)),
    );
    f.render_widget(input_widget, chunks[0]);

    // Keybind Hints & Token Count Footer
    let hints = Line::from(vec![
        Span::styled(" [Enter] ", Style::default().fg(COLOR_PRIMARY).bold()),
        Span::styled("Send  ", Style::default().fg(COLOR_MUTED)),
        Span::styled("[Tab / Ctrl+S] ", Style::default().fg(COLOR_PRIMARY).bold()),
        Span::styled("Settings  ", Style::default().fg(COLOR_MUTED)),
        Span::styled("[F1-F3] ", Style::default().fg(COLOR_PRIMARY).bold()),
        Span::styled("Tabs  ", Style::default().fg(COLOR_MUTED)),
        Span::styled("[Esc] ", Style::default().fg(COLOR_PRIMARY).bold()),
        Span::styled("Stop  ", Style::default().fg(COLOR_MUTED)),
        Span::styled("[Ctrl+C] ", Style::default().fg(Color::Red).bold()),
        Span::styled("Quit  ", Style::default().fg(COLOR_MUTED)),
        Span::styled(
            format!(" [Tok: {est_tok} / {max_ctx}]"),
            Style::default().fg(COLOR_MUTED),
        ),
    ]);
    let hints_widget = Paragraph::new(hints).alignment(Alignment::Center);
    f.render_widget(hints_widget, chunks[1]);
}
