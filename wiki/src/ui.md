# User Interfaces

Glint provides two rich interactive interfaces out of the box: a terminal user interface (TUI) and an embedded web dashboard.

---

## 1. Terminal UI (Ratatui)

The TUI provides a full-featured terminal workspace powered by `ratatui` and `crossterm`.

### Launching

```bash
# Via chat flag
glint chat -f model.gguf --tui

# Or direct subcommand
glint tui -f model.gguf
```

### Features

- **Split View Layout**: Live chat transcript, telemetry gauges, and generation parameters.
- **Real-Time Telemetry**: Live tokens/sec meter, context window memory occupancy, and real-time KV cache residency.
- **Interactive Drawer**: Toggle system prompts, temperature, top-p, and max tokens with `Ctrl+S`.
- **Keyboard Shortcuts**:
  - `Enter`: Send message / submit input
  - `Shift+Enter` / `Alt+Enter`: Multi-line prompt input
  - `Ctrl+C`: Interrupt generation or exit
  - `Ctrl+L`: Clear chat transcript

---

## 2. Embedded Web Dashboard

When running `glint serve`, an embedded single-page web dashboard is served directly by the server.

### Launching

```bash
glint serve -f model.Q4_K_M.gguf -p 8080
```

Open your browser to:
```
http://localhost:8080
```

### Features

- **No External Node/Build Step**: Zero dependencies, served directly from binary static assets.
- **Live SSE Streaming**: Fluid word-by-word generation with real-time cancel/stop button.
- **Dynamic Model Resolution**: Automatically detects model metadata from `/v1/models`.
- **Settings Drawer**: Adjust temperature, top-p, repeat penalty, and system instructions.
- **Markdown & Code Highlighting**: Formatted message output with syntax highlighting and copy buttons.
