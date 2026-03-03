//! Ferrite CLI — inspect and run GGUF models.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use ferrite::model::config::ModelConfig;
use ferrite::model::gguf::GgufModel;

#[derive(Parser)]
#[command(name = "ferrite")]
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
}

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
