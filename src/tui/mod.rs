//! Terminal User Interface (TUI) module for Glint.

pub mod app;
pub mod events;
pub mod ui;

use std::io::stdout;
use std::path::PathBuf;

use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use app::App;
use events::{handle_events, AppAction};

/// Launch the interactive Glint Terminal User Interface.
pub fn run_tui(
    model_path: PathBuf,
    system_prompt: Option<String>,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    max_tokens: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(
        model_path,
        system_prompt,
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        max_tokens,
    )?;

    // Main run loop
    loop {
        app.poll_events();

        terminal.draw(|f| ui::render(f, &mut app))?;

        match handle_events(&mut app) {
            Ok(AppAction::Quit) => break,
            Ok(AppAction::Continue) => {}
            Err(err) => {
                eprintln!("TUI event error: {err}");
                break;
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
