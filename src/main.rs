//! Glint CLI — inspect, run, and serve GGUF models.

use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write as IoWrite};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glint::model::chat_template::{ChatTemplate, Message};
use glint::model::config::ModelConfig;
use glint::model::gguf::GgufModel;
use glint::model::tokenizer::Tokenizer;
use glint::sampling::{Sampler, SamplerConfig};
use glint::server::AppState;
use glint::transformer::{TransformerWeights, generate_cached, generate_greedy_cached, generate_streaming};

#[derive(Parser)]
#[command(name = "glint")]
#[command(version, about = "LLM inference engine built in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect a GGUF model file.
    Inspect {
        /// Path to the .gguf model file.
        #[arg(short, long)]
        file: PathBuf,

        /// Show all metadata key-value pairs.
        #[arg(long, default_value_t = false)]
        show_metadata: bool,

        /// Show all tensor names, shapes, and types.
        #[arg(long, default_value_t = false)]
        show_tensors: bool,
    },

    /// Generate text from a GGUF model with a text prompt.
    Run {
        /// Path to the .gguf model file.
        #[arg(short, long)]
        file: PathBuf,

        /// Text prompt.
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of new tokens to generate.
        #[arg(short, long, default_value_t = 50)]
        max_tokens: usize,

        /// Sampling temperature. 0.0 = greedy (default), higher = more random.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Top-k sampling. 0 = disabled (default).
        #[arg(long, default_value_t = 0)]
        top_k: usize,

        /// Top-p (nucleus) sampling. 1.0 = disabled (default).
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Repetition penalty. 1.0 = disabled (default).
        #[arg(long, default_value_t = 1.0)]
        repeat_penalty: f32,

        /// Random seed for reproducible sampling.
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Generate tokens from raw token IDs (for debugging).
    Generate {
        /// Path to the .gguf model file.
        #[arg(short, long)]
        file: PathBuf,

        /// Prompt as comma-separated token IDs.
        #[arg(short, long)]
        tokens: String,

        /// Maximum number of new tokens to generate.
        #[arg(short, long, default_value_t = 20)]
        max_tokens: usize,
    },

    /// Interactive multi-turn chat with a GGUF model.
    Chat {
        /// Path to the .gguf model file.
        #[arg(short, long)]
        file: PathBuf,

        /// Optional system prompt prepended to every conversation.
        #[arg(long)]
        system: Option<String>,

        /// Maximum number of new tokens to generate per response.
        #[arg(short, long, default_value_t = 256)]
        max_tokens: usize,

        /// Sampling temperature. 0.0 = greedy, higher = more random.
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Top-k sampling. 0 = disabled.
        #[arg(long, default_value_t = 0)]
        top_k: usize,

        /// Top-p (nucleus) sampling. 1.0 = disabled.
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Repetition penalty. 1.0 = disabled.
        #[arg(long, default_value_t = 1.1)]
        repeat_penalty: f32,

        /// Random seed for reproducible sampling.
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Start the OpenAI-compatible HTTP inference server.
    Serve {
        /// Path to the .gguf model file.
        #[arg(short, long)]
        file: PathBuf,

        /// Port to listen on.
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Host/IP to bind to. Use 0.0.0.0 to accept external connections.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect {
            file,
            show_metadata,
            show_tensors,
        } => {
            inspect_model(&file, show_metadata, show_tensors);
        }
        Commands::Run {
            file,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
        } => {
            run_model(&file, &prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, seed);
        }
        Commands::Generate {
            file,
            tokens,
            max_tokens,
        } => {
            generate_tokens(&file, &tokens, max_tokens);
        }
        Commands::Chat {
            file, system, max_tokens, temperature, top_k, top_p, repeat_penalty, seed,
        } => {
            chat_model(&file, system.as_deref(), max_tokens, temperature, top_k, top_p, repeat_penalty, seed);
        }
        Commands::Serve { file, port, host } => {
            serve_model(&file, &host, port).await;
        }
    }
}

fn inspect_model(path: &PathBuf, show_metadata: bool, show_tensors: bool) {
    println!("Loading GGUF file: {}", path.display());
    println!();

    let model = match GgufModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading GGUF file: {e}");
            std::process::exit(1);
        }
    };

    // Header
    println!("═══ Header ═══");
    println!("GGUF version:      {}", model.version);
    println!("Tensor count:      {}", model.tensor_count());
    println!("Metadata entries:  {}", model.metadata.len());
    if let Some(name) = model.model_name() {
        println!("Model name:        {name}");
    }
    if let Some(arch) = model.architecture() {
        println!("Architecture:      {arch}");
    }
    println!();

    // Model config
    if let Some(config) = ModelConfig::from_metadata(&model.metadata) {
        println!("═══ Model Configuration ═══");
        print!("{config}");
        println!();
    }

    // Metadata
    if show_metadata {
        println!("═══ Metadata ({} entries) ═══", model.metadata.len());
        let mut keys: Vec<&String> = model.metadata.keys().collect();
        keys.sort();
        for key in keys {
            let value = &model.metadata[key];
            println!("  {key} ({}) = {value}", value.type_name());
        }
        println!();
    }

    // Tensors
    if show_tensors {
        println!("═══ Tensors ({}) ═══", model.tensor_count());
        for info in &model.tensor_infos {
            let shape: Vec<String> = info.dimensions.iter().map(|d| d.to_string()).collect();
            let shape_str = shape.join(" × ");
            println!(
                "  {:50} {:>8} [{shape_str}]  ({} bytes)",
                info.name,
                info.ggml_type,
                info.data_size(),
            );
        }
        println!();
    }

    // Summary
    println!("═══ Summary ═══");
    let total_bytes = model.total_tensor_bytes();
    let total_params = model.total_parameters();
    println!(
        "Total parameters:  {} ({:.1}M)",
        format_number(total_params),
        total_params as f64 / 1_000_000.0,
    );
    println!(
        "Total tensor data: {} ({:.1} MB)",
        format_bytes(total_bytes as u64),
        total_bytes as f64 / (1024.0 * 1024.0),
    );

    let mut type_counts: std::collections::HashMap<&str, (usize, usize)> =
        std::collections::HashMap::new();
    for info in &model.tensor_infos {
        let entry = type_counts.entry(info.ggml_type.name()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += info.data_size();
    }
    println!("\nTensors by type:");
    let mut types: Vec<_> = type_counts.into_iter().collect();
    types.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    for (type_name, (count, bytes)) in types {
        println!(
            "  {type_name:>8}: {count:>4} tensors, {:.1} MB",
            bytes as f64 / (1024.0 * 1024.0),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_model(
    path: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: Option<u64>,
) {
    let model = match GgufModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading GGUF file: {e}");
            std::process::exit(1);
        }
    };

    let config = match ModelConfig::from_metadata(&model.metadata) {
        Some(c) => c,
        None => { eprintln!("Error: could not extract model configuration from GGUF metadata"); std::process::exit(1); }
    };

    eprintln!("Loading tokenizer...");
    let tokenizer = match Tokenizer::from_gguf(&model) {
        Ok(t) => t,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    let mut prompt_tokens = tokenizer.encode(prompt);
    // Prepend BOS token
    prompt_tokens.insert(0, tokenizer.bos_token_id);

    eprintln!("Prompt: {:?}", prompt);
    eprintln!("Tokens: {:?} ({} tokens)", prompt_tokens, prompt_tokens.len());

    eprintln!("Loading weights...");
    let weights = match TransformerWeights::load(&model, &config) {
        Ok(w) => w,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };

    let use_sampling = temperature > 0.0;
    if use_sampling {
        eprintln!(
            "Sampling: temp={temperature}, top_k={top_k}, top_p={top_p}, repeat_penalty={repeat_penalty}{}",
            seed.map_or(String::new(), |s| format!(", seed={s}"))
        );
    } else {
        eprintln!("Sampling: greedy (temperature=0)");
    }

    eprintln!("Generating...\n");
    let start = Instant::now();

    let output = if use_sampling {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            ..Default::default()
        });
        generate_cached(&weights, &config, &prompt_tokens, max_tokens, &mut sampler, tokenizer.eos_token_id)
    } else {
        generate_greedy_cached(&weights, &config, &prompt_tokens, max_tokens)
    };

    let elapsed = start.elapsed();

    // Decode and print the generated tokens (only the new ones)
    let generated = &output[prompt_tokens.len()..];
    let text = tokenizer.decode(generated);
    println!("Prompt: {prompt}");
    println!("Output: {text}");

    let n_gen = generated.len();
    let secs = elapsed.as_secs_f64();
    let tok_per_sec = if secs > 0.0 { n_gen as f64 / secs } else { 0.0 };
    println!("\n({n_gen} tokens in {secs:.2}s — {tok_per_sec:.1} tok/s)");
}

fn generate_tokens(path: &PathBuf, tokens_str: &str, max_tokens: usize) {
    let model = match GgufModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading GGUF file: {e}");
            std::process::exit(1);
        }
    };

    let config = match ModelConfig::from_metadata(&model.metadata) {
        Some(c) => c,
        None => { eprintln!("Error: could not extract model configuration from GGUF metadata"); std::process::exit(1); }
    };
    println!("Model: {} ({})", config.architecture, path.display());
    println!("{config}");

    let prompt_tokens: Vec<u32> = tokens_str
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid token ID"))
        .collect();
    println!("Prompt tokens: {:?}", prompt_tokens);
    println!();

    eprintln!("Loading weights...");
    let weights = match TransformerWeights::load(&model, &config) {
        Ok(w) => w,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };

    eprintln!("Generating...");
    let output = generate_greedy_cached(&weights, &config, &prompt_tokens, max_tokens);

    println!("\n═══ Output Tokens ═══");
    println!("{:?}", output);
    println!("\nGenerated {} new tokens", output.len() - prompt_tokens.len());
}

#[allow(clippy::too_many_arguments)]
fn chat_model(
    path: &PathBuf,
    system_prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: Option<u64>,
) {
    let model = match GgufModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading GGUF file: {e}");
            std::process::exit(1);
        }
    };

    let config = match ModelConfig::from_metadata(&model.metadata) {
        Some(c) => c,
        None => { eprintln!("Error: could not extract model configuration from GGUF metadata"); std::process::exit(1); }
    };

    eprintln!("Loading tokenizer...");
    let tokenizer = match Tokenizer::from_gguf(&model) {
        Ok(t) => t,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    eprintln!("Loading weights...");
    let weights = match TransformerWeights::load(&model, &config) {
        Ok(w) => w,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };

    let chat_template = config.chat_template.as_deref()
        .map(ChatTemplate::detect)
        .unwrap_or(ChatTemplate::Generic);
    eprintln!("Chat template:  {}", chat_template.name());

    // Conversation history as owned (role, content) pairs
    let mut history: Vec<(String, String)> = Vec::new();
    if let Some(sys) = system_prompt {
        history.push(("system".to_string(), sys.to_string()));
    }

    eprintln!("\nGlint Chat — type your message and press Enter. Ctrl+D to exit.\n");

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("> ");
        io::stdout().flush().ok();

        input.clear();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => break, // EOF (Ctrl+D)
            Ok(_) => {}
            Err(_) => break,
        }

        let user_text = input.trim();
        if user_text.is_empty() {
            continue;
        }

        history.push(("user".to_string(), user_text.to_string()));

        // Build prompt, trimming oldest non-system messages if over context budget
        let context_budget = config.context_length as usize;
        let prompt_tokens = loop {
            let msgs: Vec<Message<'_>> = history.iter()
                .map(|(role, content)| Message { role, content })
                .collect();
            let prompt = chat_template.apply(&msgs);
            let mut tokens = tokenizer.encode(&prompt);
            tokens.insert(0, tokenizer.bos_token_id);

            if tokens.len() + max_tokens <= context_budget {
                break tokens;
            }

            // Drop the oldest non-system message to free space
            let drop_idx = history.iter().position(|(r, _)| r != "system");
            match drop_idx {
                Some(idx) => {
                    eprintln!("[Context limit reached — dropping oldest message to make room]");
                    history.remove(idx);
                }
                None => {
                    // Only system message remains and still too long — truncate
                    eprintln!("[Warning: prompt still exceeds context window after trimming, truncating]");
                    tokens.truncate(context_budget.saturating_sub(max_tokens));
                    break tokens;
                }
            }
        };

        let sampler_cfg = SamplerConfig {
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            ..Default::default()
        };
        let mut sampler = Sampler::new(sampler_cfg);

        // Stream tokens to stdout
        let output = generate_streaming(
            &weights,
            &config,
            &prompt_tokens,
            max_tokens,
            &mut sampler,
            tokenizer.eos_token_id,
            |token_id| {
                let text = tokenizer.decode(&[token_id]);
                print!("{text}");
                io::stdout().flush().ok();
                true
            },
        );
        println!();

        // Add assistant response to history for multi-turn
        let generated = &output[prompt_tokens.len()..];
        let assistant_text = tokenizer.decode(generated);
        history.push(("assistant".to_string(), assistant_text));
    }

    eprintln!("\nGoodbye!");
}

async fn serve_model(path: &PathBuf, host: &str, port: u16) {
    let model = match GgufModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading GGUF file: {e}");
            std::process::exit(1);
        }
    };

    let config = match ModelConfig::from_metadata(&model.metadata) {
        Some(c) => c,
        None => { eprintln!("Error: could not extract model configuration from GGUF metadata"); std::process::exit(1); }
    };

    eprintln!("Loading tokenizer...");
    let tokenizer = match Tokenizer::from_gguf(&model) {
        Ok(t) => t,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    eprintln!("Loading weights...");
    let weights = match TransformerWeights::load(&model, &config) {
        Ok(w) => w,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };
    eprintln!("Weights loaded.");

    // Derive a model name from the file stem (e.g. "smollm-135m-instruct.Q8_0")
    let model_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("glint-model")
        .to_string();

    // Detect chat template from GGUF metadata
    let chat_template = config.chat_template.as_deref()
        .map(ChatTemplate::detect)
        .unwrap_or(ChatTemplate::Generic);
    eprintln!("Chat template:  {}", chat_template.name());

    let state = AppState {
        weights: Arc::new(weights),
        tokenizer: Arc::new(tokenizer),
        config: Arc::new(config),
        model_name,
        chat_template,
    };

    glint::server::run_server(state, host, port).await;
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
