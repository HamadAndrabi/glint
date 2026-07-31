//! Glint CLI — inspect, run, and serve GGUF models.

use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};
#[cfg(feature = "server")]
use std::sync::Arc;
use std::time::Instant;

use glint::api::Model as GlintModel;
use glint::bench;
#[cfg(feature = "server")]
use glint::cache::{PagePool, PAGE_SIZE};
#[cfg(feature = "server")]
use glint::constrained::VocabIndex;
use glint::model::chat_template::{ChatTemplate, Message};
use glint::model::config::ModelConfig;
use glint::model::gguf::GgufModel;
#[cfg(feature = "server")]
use glint::model::lora_registry::AdapterRegistry;
#[cfg(feature = "server")]
use glint::model::pull::{pull_model, search_huggingface};
use glint::model::tokenizer::Tokenizer;
use glint::sampling::{Sampler, SamplerConfig};
#[cfg(feature = "server")]
use glint::server::{AppState, InferenceEngine, Metrics};
#[cfg(feature = "server")]
use glint::session::CacheFormat;
use glint::transformer::{
    generate_cached, generate_greedy_cached, generate_streaming, speculative_decode,
    TransformerWeights,
};

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

        /// Path to a small draft model for speculative decoding (optional).
        #[arg(long)]
        draft_model: Option<PathBuf>,

        /// Number of tokens the draft model generates per verification round.
        #[arg(long, default_value_t = 4)]
        lookahead: usize,

        /// Path to a LoRA adapter GGUF file (optional).
        #[arg(long)]
        lora: Option<PathBuf>,

        /// Use GPU acceleration (requires `vulkan` feature).
        #[arg(long, default_value_t = false)]
        gpu: bool,
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

        /// Path to a LoRA adapter GGUF file (optional).
        #[arg(long)]
        lora: Option<PathBuf>,

        /// Use GPU acceleration (requires `vulkan` feature).
        #[arg(long, default_value_t = false)]
        gpu: bool,
    },

    /// Start the OpenAI-compatible HTTP inference server.
    #[cfg(feature = "server")]
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

        /// Use GPU acceleration (requires `vulkan` feature).
        #[arg(long, default_value_t = false)]
        gpu: bool,

        /// KV-cache storage format: "f32" (default, full precision),
        /// "q8" (~3.8× smaller), or "paged" (f32 in on-demand 16-token pages
        /// shared by all requests — memory follows real usage).
        #[arg(long, default_value = "f32")]
        kv_cache: String,
    },

    /// Download a GGUF model from HuggingFace Hub.
    ///
    /// Example: glint pull bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
    #[cfg(feature = "server")]
    Pull {
        /// HuggingFace repository in "owner/repo" format.
        repo: String,

        /// GGUF filename to download (e.g. "SmolLM2-135M-Instruct-Q8_0.gguf").
        file: String,

        /// Directory to save the model to (default: platform cache dir / glint / models).
        #[arg(long)]
        dir: Option<PathBuf>,
    },

    /// Benchmark inference performance (prefill, decode, concurrency, cache formats).
    Bench {
        /// Path to the .gguf model file.
        #[arg(short, long)]
        file: PathBuf,

        /// Which benchmark mode(s) to run: "all", "prefill", "decode", "concurrency", "cache-format".
        #[arg(long, default_value = "all")]
        mode: String,

        /// Number of prompt tokens to use for all benchmarks.
        #[arg(long, default_value_t = 512)]
        prompt_len: usize,

        /// Number of new tokens to decode in decode/concurrency benchmarks.
        #[arg(long, default_value_t = 128)]
        decode_tokens: usize,

        /// Maximum number of concurrent sessions for the concurrency benchmark.
        #[arg(long, default_value_t = 8)]
        max_concurrent: usize,

        /// Number of warm-up rounds (discarded).
        #[arg(long, default_value_t = 3)]
        warmup: usize,

        /// Number of timed measurement rounds.
        #[arg(long, default_value_t = 10)]
        iters: usize,

        /// If set, write results as JSON to this file path.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[cfg(feature = "server")]
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
            draft_model,
            lookahead,
            lora,
            gpu,
        } => {
            let file = maybe_download(&file).await;
            run_model(
                &file,
                &prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
                seed,
                draft_model.as_deref(),
                lookahead,
                lora.as_deref(),
                gpu,
            );
        }
        Commands::Generate {
            file,
            tokens,
            max_tokens,
        } => {
            generate_tokens(&file, &tokens, max_tokens);
        }
        Commands::Chat {
            file,
            system,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            lora,
            gpu,
        } => {
            let file = maybe_download(&file).await;
            chat_model(
                &file,
                system.as_deref(),
                max_tokens,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
                seed,
                lora.as_deref(),
                gpu,
            );
        }
        #[cfg(feature = "server")]
        Commands::Serve {
            file,
            port,
            host,
            gpu,
            kv_cache,
        } => {
            let file = maybe_download(&file).await;
            serve_model(&file, &host, port, gpu, &kv_cache).await;
        }
        #[cfg(feature = "server")]
        Commands::Pull { repo, file, dir } => {
            pull_model_cmd(&repo, &file, dir.as_deref()).await;
        }
        Commands::Bench {
            file,
            mode,
            prompt_len,
            decode_tokens,
            max_concurrent,
            warmup,
            iters,
            output,
        } => {
            bench_model(
                &file,
                &mode,
                prompt_len,
                decode_tokens,
                max_concurrent,
                warmup,
                iters,
                output.as_deref(),
            );
        }
    }
}

#[cfg(not(feature = "server"))]
fn main() {
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
            draft_model,
            lookahead,
            lora,
            gpu,
        } => {
            run_model(
                &file,
                &prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
                seed,
                draft_model.as_deref(),
                lookahead,
                lora.as_deref(),
                gpu,
            );
        }
        Commands::Generate {
            file,
            tokens,
            max_tokens,
        } => {
            generate_tokens(&file, &tokens, max_tokens);
        }
        Commands::Chat {
            file,
            system,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            lora,
            gpu,
        } => {
            chat_model(
                &file,
                system.as_deref(),
                max_tokens,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
                seed,
                lora.as_deref(),
                gpu,
            );
        }
        Commands::Bench {
            file,
            mode,
            prompt_len,
            decode_tokens,
            max_concurrent,
            warmup,
            iters,
            output,
        } => {
            bench_model(
                &file,
                &mode,
                prompt_len,
                decode_tokens,
                max_concurrent,
                warmup,
                iters,
                output.as_deref(),
            );
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

/// Initialize GPU backend if requested and the `vulkan` feature is enabled.
///
/// Returns `Some(GpuBackend)` with weights uploaded, or `None` if GPU is not
/// requested / not available. Prints a warning if `--gpu` is passed without
/// the `vulkan` feature compiled in.
fn init_gpu(
    use_gpu: bool,
    _weights: &mut TransformerWeights,
) -> Option<glint::backend::GpuBackend> {
    if !use_gpu {
        return None;
    }

    #[cfg(feature = "vulkan")]
    {
        match glint::backend::GpuBackend::new() {
            Ok(mut gpu) => {
                _weights.upload_all_to_gpu(&mut gpu);
                eprintln!("GPU backend initialized.");
                Some(gpu)
            }
            Err(e) => {
                eprintln!("Warning: GPU initialization failed ({e}), falling back to CPU.");
                None
            }
        }
    }

    #[cfg(not(feature = "vulkan"))]
    {
        eprintln!("Warning: --gpu requires the `vulkan` feature. Build with: cargo build --features vulkan");
        eprintln!("Continuing on CPU.");
        None
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
    draft_path: Option<&Path>,
    lookahead: usize,
    lora_path: Option<&Path>,
    use_gpu: bool,
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
        None => {
            eprintln!("Error: could not extract model configuration from GGUF metadata");
            std::process::exit(1);
        }
    };

    eprintln!("Loading tokenizer...");
    let tokenizer = match Tokenizer::from_gguf(&model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    let prompt_tokens = tokenizer.encode_prompt(prompt);

    eprintln!("Prompt: {:?}", prompt);
    eprintln!(
        "Tokens: {:?} ({} tokens)",
        prompt_tokens,
        prompt_tokens.len()
    );

    eprintln!("Loading weights...");
    let weights = match TransformerWeights::load(&model, &config) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let mut weights = if let Some(lp) = lora_path {
        let lora_model = match GgufModel::load(lp) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error loading LoRA file: {e}");
                std::process::exit(1);
            }
        };
        weights.with_lora(&lora_model)
    } else {
        weights
    };

    // GPU initialization
    let mut gpu_backend = init_gpu(use_gpu, &mut weights);
    let mut gpu: Option<&mut glint::backend::GpuBackend> = gpu_backend.as_mut();

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

    let output = if let Some(draft_path) = draft_path {
        // Speculative decoding path
        let draft_model = match GgufModel::load(draft_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error loading draft model: {e}");
                std::process::exit(1);
            }
        };
        let draft_config = match ModelConfig::from_metadata(&draft_model.metadata) {
            Some(c) => c,
            None => {
                eprintln!("Error: could not extract draft model config");
                std::process::exit(1);
            }
        };
        let draft_weights = match TransformerWeights::load(&draft_model, &draft_config) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Error loading draft weights: {e}");
                std::process::exit(1);
            }
        };
        eprintln!("Speculative decoding: lookahead={lookahead}");
        speculative_decode(
            &draft_weights,
            &draft_config,
            &weights,
            &config,
            &prompt_tokens,
            max_tokens,
            lookahead,
            temperature,
            tokenizer.eos_token_id,
            seed,
            &mut gpu,
        )
    } else if use_sampling {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            ..Default::default()
        });
        generate_cached(
            &weights,
            &config,
            &prompt_tokens,
            max_tokens,
            &mut sampler,
            tokenizer.eos_token_id,
            &mut gpu,
        )
    } else {
        generate_greedy_cached(&weights, &config, &prompt_tokens, max_tokens, &mut gpu)
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
        None => {
            eprintln!("Error: could not extract model configuration from GGUF metadata");
            std::process::exit(1);
        }
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
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("Generating...");
    let output = generate_greedy_cached(&weights, &config, &prompt_tokens, max_tokens, &mut None);

    println!("\n═══ Output Tokens ═══");
    println!("{:?}", output);
    println!(
        "\nGenerated {} new tokens",
        output.len() - prompt_tokens.len()
    );
}

/// Run the model on a summarization prompt and return the decoded summary string.
///
/// Used by the chat loop to compress old conversation turns before they're evicted
/// from the context window. Runs greedy sampling for deterministic output.
fn summarize_messages(
    weights: &TransformerWeights,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
    messages: &[(String, String)],
    context_budget: usize,
) -> String {
    let mut transcript =
        String::from("Summarize the following conversation briefly in 2-3 sentences:\n\n");
    for (role, content) in messages {
        let label = match role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            _ => continue,
        };
        transcript.push_str(&format!("{label}: {content}\n"));
    }
    transcript.push_str("\nSummary:");

    let prompt_tokens = tokenizer.encode_prompt(&transcript);

    let max_summary_tokens = 120usize;
    let available = context_budget.saturating_sub(prompt_tokens.len());
    let gen_tokens = max_summary_tokens.min(available);
    if gen_tokens == 0 {
        return String::from("[prior conversation]");
    }

    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    });
    let output = generate_cached(
        weights,
        config,
        &prompt_tokens,
        gen_tokens,
        &mut sampler,
        tokenizer.eos_token_id,
        &mut None,
    );
    tokenizer
        .decode(&output[prompt_tokens.len()..])
        .trim()
        .to_string()
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
    lora_path: Option<&Path>,
    use_gpu: bool,
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
        None => {
            eprintln!("Error: could not extract model configuration from GGUF metadata");
            std::process::exit(1);
        }
    };

    eprintln!("Loading tokenizer...");
    let tokenizer = match Tokenizer::from_gguf(&model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    eprintln!("Loading weights...");
    let weights = match TransformerWeights::load(&model, &config) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let mut weights = if let Some(lp) = lora_path {
        let lora_model = match GgufModel::load(lp) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error loading LoRA file: {e}");
                std::process::exit(1);
            }
        };
        weights.with_lora(&lora_model)
    } else {
        weights
    };

    let mut gpu_backend = init_gpu(use_gpu, &mut weights);
    let mut gpu: Option<&mut glint::backend::GpuBackend> = gpu_backend.as_mut();

    let chat_template = config
        .chat_template
        .as_deref()
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

        // Build prompt, summarizing or trimming if over context budget
        let context_budget = config.context_length as usize;
        let prompt_tokens = loop {
            let msgs: Vec<Message<'_>> = history
                .iter()
                .map(|(role, content)| Message { role, content })
                .collect();
            let prompt = chat_template.apply(&msgs);
            let mut tokens = tokenizer.encode_prompt(&prompt);

            if tokens.len() + max_tokens <= context_budget {
                break tokens;
            }

            // Collect indices of non-system messages
            let non_sys: Vec<usize> = history
                .iter()
                .enumerate()
                .filter(|(_, (r, _))| r != "system")
                .map(|(i, _)| i)
                .collect();

            // Summarize if we have enough old messages (keep 2 recent, summarize the rest)
            if non_sys.len() >= 3 {
                let keep_from = non_sys.len() - 2;
                let to_summarize: Vec<(String, String)> = non_sys[..keep_from]
                    .iter()
                    .map(|&i| history[i].clone())
                    .collect();
                eprintln!("[Context limit reached — summarizing earlier conversation...]");
                let summary = summarize_messages(
                    &weights,
                    &config,
                    &tokenizer,
                    &to_summarize,
                    context_budget,
                );
                // Remove summarized messages (reverse order to keep indices valid)
                for &i in non_sys[..keep_from].iter().rev() {
                    history.remove(i);
                }
                // Insert summary after system messages
                let insert_at = history.iter().position(|(r, _)| r != "system").unwrap_or(0);
                history.insert(
                    insert_at,
                    (
                        "system".to_string(),
                        format!("Summary of earlier conversation: {summary}"),
                    ),
                );
                continue;
            }

            // Fall back: drop the oldest non-system message
            if let Some(idx) = history.iter().position(|(r, _)| r != "system") {
                eprintln!("[Context limit reached — dropping oldest message to make room]");
                history.remove(idx);
            } else {
                // Only system message remains and still too long — truncate
                eprintln!("[Warning: prompt still exceeds context window, truncating]");
                tokens.truncate(context_budget.saturating_sub(max_tokens));
                break tokens;
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
            &mut gpu,
        );
        println!();

        // Add assistant response to history for multi-turn
        let generated = &output[prompt_tokens.len()..];
        let assistant_text = tokenizer.decode(generated);
        history.push(("assistant".to_string(), assistant_text));
    }

    eprintln!("\nGoodbye!");
}

#[cfg(feature = "server")]
async fn serve_model(path: &PathBuf, host: &str, port: u16, use_gpu: bool, kv_cache: &str) {
    let model = match GgufModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading GGUF file: {e}");
            std::process::exit(1);
        }
    };

    let config = match ModelConfig::from_metadata(&model.metadata) {
        Some(c) => c,
        None => {
            eprintln!("Error: could not extract model configuration from GGUF metadata");
            std::process::exit(1);
        }
    };

    eprintln!("Loading tokenizer...");
    let tokenizer = match Tokenizer::from_gguf(&model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    eprintln!("Loading weights...");
    let weights = match TransformerWeights::load(&model, &config) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Weights loaded.");

    let mut weights = weights; // make mutable for GPU upload
    let gpu_backend = init_gpu(use_gpu, &mut weights);

    // Derive a model name from the file stem (e.g. "smollm-135m-instruct.Q8_0")
    let model_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("glint-model")
        .to_string();

    // Detect chat template from GGUF metadata
    let chat_template = config
        .chat_template
        .as_deref()
        .map(ChatTemplate::detect)
        .unwrap_or(ChatTemplate::Generic);
    eprintln!("Chat template:  {}", chat_template.name());

    let weights = Arc::new(weights);
    let config_arc = Arc::new(config);

    // Parse cache format from CLI arg. "paged" keeps f32 storage but hands it
    // out in pages from one shared pool instead of pre-allocating a full
    // context per request.
    let mut limits = glint::server::EngineLimits::default();
    let cache_format = match kv_cache {
        "q8" => CacheFormat::Q8,
        _ => CacheFormat::F32,
    };
    if kv_cache == "paged" {
        // Sized for the worst case (every active sequence filling the context),
        // so paging never rejects work the contiguous cache would have taken;
        // pages are allocated lazily, so idle capacity costs nothing.
        limits.kv_pool_pages = Some(
            PagePool::pages_for(
                config_arc.context_length as usize,
                config_arc.block_count as usize,
            ) * limits.max_active,
        );
    }
    eprintln!(
        "KV-cache format: {}",
        match (cache_format, limits.kv_pool_pages) {
            (CacheFormat::Q8, _) => "Q8 (quantised)".to_string(),
            (CacheFormat::F32, Some(pages)) =>
                format!("F32 paged ({pages} pages × {PAGE_SIZE} tokens)"),
            (CacheFormat::F32, None) => "F32 (full precision)".to_string(),
        }
    );

    // Start the concurrent round-robin inference engine on a dedicated OS
    // thread. The engine owns the GPU backend (if any) and all active KV
    // caches.
    // Pre-build the vocabulary index for constraint-based generation (JSON mode, etc.).
    let vocab_strings: Vec<String> = (0..tokenizer.vocab_size())
        .map(|i| tokenizer.decode_token(i as u32).to_owned())
        .collect();
    let vocab_index = VocabIndex::from_vocab(&vocab_strings);

    // Build an empty adapter registry (adapters can be pre-loaded here in future
    // via a --lora-adapter flag; for now the registry starts empty).
    let adapter_registry = Arc::new(std::sync::RwLock::new(AdapterRegistry::new()));
    let engine_registry = Arc::new(AdapterRegistry::new()); // immutable snapshot for engine

    let engine = Arc::new(InferenceEngine::start(
        Arc::clone(&weights),
        Arc::clone(&config_arc),
        gpu_backend,
        cache_format,
        Arc::clone(&vocab_index),
        engine_registry,
        limits,
    ));

    let state = AppState {
        weights,
        tokenizer: Arc::new(tokenizer),
        config: config_arc,
        model_name,
        chat_template,
        metrics: Metrics::new(),
        engine,
        vocab_index,
        adapter_registry,
    };

    glint::server::run_server(state, host, port).await;
}

fn bench_model(
    path: &PathBuf,
    mode: &str,
    prompt_len: usize,
    decode_tokens: usize,
    max_concurrent: usize,
    warmup: usize,
    iters: usize,
    output: Option<&Path>,
) {
    eprintln!("Loading model: {}", path.display());
    let model = match GlintModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Model loaded. Running benchmark mode: {mode}");
    eprintln!(
        "  prompt_len={prompt_len}  decode_tokens={decode_tokens}  warmup={warmup}  iters={iters}"
    );
    eprintln!();

    let mut results: Vec<bench::BenchResult> = Vec::new();

    let run_prefill = matches!(mode, "all" | "prefill");
    let run_decode = matches!(mode, "all" | "decode");
    let run_conc = matches!(mode, "all" | "concurrency");
    let run_cache = matches!(mode, "all" | "cache-format");

    if run_prefill {
        eprintln!("  [1/4] Prefill benchmark...");
        results.push(bench::run_prefill_bench(&model, prompt_len, warmup, iters));
    }
    if run_decode {
        eprintln!("  [2/4] Decode benchmark...");
        results.push(bench::run_decode_bench(
            &model,
            prompt_len,
            decode_tokens,
            warmup,
            iters,
        ));
    }
    if run_conc {
        eprintln!("  [3/4] Concurrency benchmark (1..={max_concurrent} sessions)...");
        let levels: Vec<usize> = (0..)
            .map(|i| 1usize << i)
            .take_while(|&n| n <= max_concurrent)
            .collect();
        for n in levels {
            eprint!("    n={n}... ");
            results.push(bench::run_concurrency_bench(
                &model,
                n,
                prompt_len,
                decode_tokens,
                warmup,
                iters,
            ));
            eprintln!("done");
        }
    }
    if run_cache {
        eprintln!("  [4/4] Cache-format benchmark (f32 vs q8)...");
        results.extend(bench::run_cache_format_bench(
            &model,
            prompt_len,
            decode_tokens,
            warmup,
            iters,
        ));
    }

    eprintln!();
    bench::print_results(&results);

    if let Some(out_path) = output {
        let json = bench::results_to_json(&results);
        match std::fs::write(out_path, &json) {
            Ok(()) => eprintln!("\nResults written to: {}", out_path.display()),
            Err(e) => eprintln!("Warning: could not write output file: {e}"),
        }
    }
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

// ── Model download helpers ────────────────────────────────────────────────────

/// Returns the platform-specific default model cache directory.
///
/// - Windows: `%LOCALAPPDATA%\glint\models`
/// - Linux/macOS: `~/.cache/glint/models`
#[cfg(feature = "server")]
fn default_cache_dir(override_dir: Option<&Path>) -> PathBuf {
    match override_dir {
        Some(d) => d.to_path_buf(),
        None => dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("glint")
            .join("models"),
    }
}

/// If `path` doesn't exist and looks like a `.gguf` filename, search HuggingFace
/// and offer to download it. Returns the resolved path (either the original or the
/// freshly-downloaded one).
#[cfg(feature = "server")]
async fn maybe_download(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }

    let filename = match path.file_name().and_then(|n| n.to_str()) {
        Some(f) if f.ends_with(".gguf") => f,
        _ => return path.to_path_buf(), // let GgufModel::load produce the normal error
    };

    // Build a search query by stripping the quantization suffix
    // e.g. "SmolLM2-135M-Instruct-Q8_0" → "SmolLM2-135M-Instruct"
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);
    let query = stem
        .split('-')
        .take_while(|p| !p.starts_with('Q') && !p.starts_with('I'))
        .collect::<Vec<_>>()
        .join("-");

    eprintln!("File not found: {}", path.display());
    eprint!("Searching HuggingFace for \"{query}\"... ");
    io::stdout().flush().ok();

    let repos = match search_huggingface(&query).await {
        Ok(r) if !r.is_empty() => r,
        _ => {
            eprintln!("no results.");
            eprintln!("Download manually with:  glint pull <repo> {filename}");
            return path.to_path_buf();
        }
    };

    eprintln!("found {} match(es):", repos.len());
    for (i, repo) in repos.iter().enumerate() {
        eprintln!("  [{}] {}", i + 1, repo);
    }

    eprint!("Download which? [1-{}/N]: ", repos.len());
    io::stdout().flush().ok();

    let mut choice = String::new();
    io::stdin().lock().read_line(&mut choice).ok();

    if let Ok(n) = choice.trim().parse::<usize>() {
        if n >= 1 && n <= repos.len() {
            let repo = &repos[n - 1];
            let dest_dir = default_cache_dir(None);
            eprintln!("Downloading from {repo}...");
            match pull_model(repo, filename, &dest_dir).await {
                Ok(downloaded) => {
                    eprintln!("Saved to: {}", downloaded.display());
                    return downloaded;
                }
                Err(e) => eprintln!("Download failed: {e}"),
            }
        }
    }

    eprintln!("Cancelled.");
    path.to_path_buf()
}

/// Handle the `glint pull` subcommand.
#[cfg(feature = "server")]
async fn pull_model_cmd(repo: &str, filename: &str, dir: Option<&Path>) {
    let dest_dir = default_cache_dir(dir);
    eprintln!("Repository: {repo}");
    eprintln!("File:       {filename}");
    eprintln!("Saving to:  {}", dest_dir.display());
    eprintln!();

    match pull_model(repo, filename, &dest_dir).await {
        Ok(path) => {
            eprintln!("\nSaved to: {}", path.display());
            eprintln!("\nRun with:");
            eprintln!(
                "  glint run --file \"{}\" --prompt \"Your prompt here\"",
                path.display()
            );
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
