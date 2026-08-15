//! Event handling and crossterm keybinding loop for the TUI.

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use std::time::Duration;

use super::app::{ActiveTab, App};

pub enum AppAction {
    Continue,
    Quit,
}

pub fn handle_events(app: &mut App) -> Result<AppAction, Box<dyn std::error::Error>> {
    // Poll for events at 60 FPS (16ms timeout)
    if event::poll(Duration::from_millis(16))? {
        if let Event::Key(key) = event::read()? {
            return Ok(handle_key_event(app, key));
        }
    }

    Ok(AppAction::Continue)
}

fn handle_key_event(app: &mut App, key: KeyEvent) -> AppAction {
    // Global shortcuts
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return AppAction::Quit;
    }

    // Escape interrupts generation
    if key.code == KeyCode::Esc {
        if app.is_generating {
            app.cancel_generation();
            return AppAction::Continue;
        } else if app.settings_open {
            app.settings_open = false;
            return AppAction::Continue;
        }
    }

    // Toggle Settings Drawer (Tab or Ctrl+S)
    if key.code == KeyCode::Tab
        || (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s'))
    {
        app.settings_open = !app.settings_open;
        return AppAction::Continue;
    }

    // Tab Switching
    match key.code {
        KeyCode::F(1) => {
            app.active_tab = ActiveTab::Chat;
            return AppAction::Continue;
        }
        KeyCode::F(2) => {
            app.active_tab = ActiveTab::StructuredLab;
            return AppAction::Continue;
        }
        KeyCode::F(3) => {
            app.active_tab = ActiveTab::KvTelemetry;
            return AppAction::Continue;
        }
        _ => {}
    }

    // If Settings Drawer is Open, handle parameter adjustments
    if app.settings_open {
        match key.code {
            KeyCode::Up => {
                if app.selected_setting > 0 {
                    app.selected_setting -= 1;
                }
            }
            KeyCode::Down => {
                if app.selected_setting < 5 {
                    app.selected_setting += 1;
                }
            }
            KeyCode::Left => adjust_setting(app, -1.0),
            KeyCode::Right => adjust_setting(app, 1.0),
            _ => {}
        }
        return AppAction::Continue;
    }

    // Structured Lab Tab Specific Shortcuts
    if app.active_tab == ActiveTab::StructuredLab {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
            app.lab_mode = (app.lab_mode + 1) % 2;
            return AppAction::Continue;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Enter {
            app.submit_lab_request();
            return AppAction::Continue;
        }
    }

    // Input Handling for Main Prompt
    match key.code {
        KeyCode::Enter => {
            if app.active_tab == ActiveTab::StructuredLab {
                app.submit_lab_request();
            } else {
                app.submit_user_message();
            }
        }
        KeyCode::Char(c) => {
            app.input_text.push(c);
            app.input_cursor += 1;
        }
        KeyCode::Backspace => {
            if !app.input_text.is_empty() {
                app.input_text.pop();
                app.input_cursor = app.input_cursor.saturating_sub(1);
            }
        }
        KeyCode::PageUp => {
            app.chat_scroll = app.chat_scroll.saturating_add(5);
        }
        KeyCode::PageDown => {
            app.chat_scroll = app.chat_scroll.saturating_sub(5);
        }
        _ => {}
    }

    AppAction::Continue
}

fn adjust_setting(app: &mut App, delta: f32) {
    match app.selected_setting {
        0 => {
            // Temperature (0.0 .. 2.0)
            app.temperature = (app.temperature + delta * 0.05).clamp(0.0, 2.0);
        }
        1 => {
            // Top-P (0.0 .. 1.0)
            app.top_p = (app.top_p + delta * 0.05).clamp(0.0, 1.0);
        }
        2 => {
            // Top-K (0 .. 200)
            let step = if delta > 0.0 { 5 } else { -5 };
            app.top_k = (app.top_k as i32 + step).clamp(0, 200) as usize;
        }
        3 => {
            // Repeat penalty (0.8 .. 2.0)
            app.repeat_penalty = (app.repeat_penalty + delta * 0.05).clamp(0.8, 2.0);
        }
        4 => {
            // Max Tokens (16 .. 4096)
            let step = if delta > 0.0 { 64 } else { -64 };
            app.max_tokens = (app.max_tokens as i32 + step).clamp(16, 4096) as usize;
        }
        _ => {}
    }
}
