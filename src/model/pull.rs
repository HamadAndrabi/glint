//! HuggingFace Hub model downloader.
//!
//! Provides two public functions:
//!
//! - [`pull_model`] — download a single GGUF file with resume support and a
//!   progress bar.
//! - [`search_huggingface`] — search the HF Hub API for repos matching a query,
//!   filtered to GGUF repos.

use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header;
use tokio_stream::StreamExt as _;

/// Download a GGUF file from HuggingFace Hub.
///
/// URL pattern: `https://huggingface.co/{repo}/resolve/main/{filename}`
///
/// Behaviour:
/// - Creates `dest_dir` if it doesn't already exist.
/// - If the file already exists at its full size, prints a message and returns
///   immediately (idempotent).
/// - If a partial file exists, sends a `Range` header to resume the download.
/// - Shows a live progress bar (bytes / total, rate, ETA) while downloading.
///
/// Returns the path of the saved file on success.
pub async fn pull_model(
    repo: &str,
    filename: &str,
    dest_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    let dest_path = dest_dir.join(filename);

    std::fs::create_dir_all(dest_dir)?;

    // Check existing file size for potential resume
    let existing_size: u64 = if dest_path.exists() {
        dest_path.metadata()?.len()
    } else {
        0
    };

    // HEAD request to get the total file size
    let client = reqwest::Client::new();
    let head = client.head(&url).send().await?;
    if !head.status().is_success() {
        return Err(format!("Server returned {} for HEAD {url}", head.status()).into());
    }
    let total_size: u64 = head
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // Already complete — skip
    if existing_size > 0 && total_size > 0 && existing_size == total_size {
        eprintln!("Already downloaded: {}", dest_path.display());
        return Ok(dest_path);
    }

    // Build GET request, possibly with a Range header for resume
    let mut get = client.get(&url);
    let file = if existing_size > 0 && total_size > 0 {
        eprintln!("Resuming at {} bytes...", existing_size);
        get = get.header(header::RANGE, format!("bytes={existing_size}-"));
        std::fs::OpenOptions::new().append(true).open(&dest_path)?
    } else {
        std::fs::File::create(&dest_path)?
    };

    let resp = get.send().await?;
    if !resp.status().is_success() {
        return Err(format!("Server returned {} for GET {url}", resp.status()).into());
    }

    // Progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg}\n[{bar:40.cyan/blue}] {bytes}/{total_bytes} @ {bytes_per_sec} (ETA {eta})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    pb.set_message(format!("Downloading {filename}"));
    pb.set_position(existing_size);

    let mut writer = std::io::BufWriter::new(file);
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        writer.write_all(&chunk)?;
        pb.inc(chunk.len() as u64);
    }
    writer.flush()?;
    pb.finish_with_message(format!("Downloaded {filename}"));

    Ok(dest_path)
}

/// Search HuggingFace Hub for GGUF model repositories matching `query`.
///
/// Returns up to 5 repository IDs (e.g. `"bartowski/SmolLM2-135M-Instruct-GGUF"`)
/// sorted by download count (the API's default).
pub async fn search_huggingface(query: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://huggingface.co/api/models?search={query}&filter=gguf&limit=5",
        query = urlencoded(query),
    );

    let resp = reqwest::get(&url).await?;
    if !resp.status().is_success() {
        return Err(format!("HuggingFace API returned {}", resp.status()).into());
    }

    let json: serde_json::Value = resp.json().await?;
    let repos = json
        .as_array()
        .ok_or("Unexpected API response format")?
        .iter()
        .filter_map(|entry| entry["modelId"].as_str().map(|s| s.to_owned()))
        .collect();

    Ok(repos)
}

/// Minimal percent-encoding for URL query parameters (encodes spaces as `%20`).
fn urlencoded(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            ' ' => vec!['%', '2', '0'],
            '+' => vec!['%', '2', 'B'],
            '&' => vec!['%', '2', '6'],
            _ => vec![c],
        })
        .collect()
}
