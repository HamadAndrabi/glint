//! Benchmark runner for Glint inference performance.
//!
//! Measures prefill throughput, decode throughput, and concurrency scaling.
//! Driven by the `glint bench` CLI subcommand.

use std::time::Instant;

use crate::api::{GenerationOptions, Model};
use crate::backend::GpuBackend;
use crate::sampling::SamplerConfig;
use crate::session::CacheFormat;

// ── BenchResult ───────────────────────────────────────────────────────────────

/// Results for a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Human-readable mode label (e.g. "prefill", "decode", "concurrency").
    pub mode: String,
    /// Number of prompt tokens fed.
    pub prompt_tokens: usize,
    /// Number of new tokens decoded.
    pub decode_tokens: usize,
    /// Concurrency level (number of parallel sessions / requests).
    pub n_concurrent: usize,
    /// KV-cache format used.
    pub cache_format: String,
    /// Prefill latency in milliseconds.
    pub prefill_ms: f64,
    /// Average per-token decode latency in milliseconds.
    pub decode_ms_per_token: f64,
    /// Decode tokens per second.
    pub tokens_per_sec: f64,
    /// Time-to-first-token in milliseconds (= prefill_ms for single session).
    pub ttft_ms: f64,
}

// ── Synthetic prompt helpers ──────────────────────────────────────────────────

/// Build a synthetic prompt of `len` tokens using BOS token + cycling vocabulary.
fn synthetic_prompt(model: &Model, len: usize) -> Vec<u32> {
    let vocab = model.tokenizer.vocab_size() as u32;
    // Start with BOS, then cycle through mid-range vocabulary tokens.
    let mut tokens = Vec::with_capacity(len);
    tokens.push(model.tokenizer.bos_token_id);
    let mut t: u32 = 3; // skip special tokens 0-2
    while tokens.len() < len {
        tokens.push(t % vocab);
        t += 1;
    }
    tokens.truncate(len);
    tokens
}

// ── Individual benchmark functions ───────────────────────────────────────────

/// Measure prefill throughput: how fast the model processes prompt tokens.
///
/// `warmup` rounds are discarded; throughput is averaged over `iters` rounds.
pub fn run_prefill_bench(
    model: &Model,
    prompt_len: usize,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let prompt = synthetic_prompt(model, prompt_len);
    let opts = GenerationOptions {
        max_new_tokens: 1,
        sampler_cfg: SamplerConfig { temperature: 0.0, seed: Some(1), ..Default::default() },
        cache_format: CacheFormat::F32,
        ..Default::default()
    };
    let gpu: &mut Option<&mut GpuBackend> = &mut None;

    // Warm up.
    for _ in 0..warmup {
        let mut s = model.new_session(&opts);
        let _ = model.prefill_tokens(&mut s, &prompt, gpu);
    }

    // Timed iterations.
    let mut total_ms = 0.0f64;
    for _ in 0..iters {
        let mut s = model.new_session(&opts);
        let t0 = Instant::now();
        let _ = model.prefill_tokens(&mut s, &prompt, gpu);
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    let prefill_ms = total_ms / iters as f64;
    let tokens_per_sec = (prompt_len as f64) / (prefill_ms / 1000.0);

    BenchResult {
        mode: "prefill".to_string(),
        prompt_tokens: prompt_len,
        decode_tokens: 0,
        n_concurrent: 1,
        cache_format: "f32".to_string(),
        prefill_ms,
        decode_ms_per_token: 0.0,
        tokens_per_sec,
        ttft_ms: prefill_ms,
    }
}

/// Measure decode throughput: tokens/sec after prefill.
pub fn run_decode_bench(
    model: &Model,
    prompt_len: usize,
    decode_tokens: usize,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let prompt = synthetic_prompt(model, prompt_len);
    let opts = GenerationOptions {
        max_new_tokens: decode_tokens,
        sampler_cfg: SamplerConfig { temperature: 0.0, seed: Some(1), ..Default::default() },
        cache_format: CacheFormat::F32,
        ..Default::default()
    };
    let gpu: &mut Option<&mut GpuBackend> = &mut None;

    // Warm up.
    for _ in 0..warmup {
        let mut s = model.new_session(&opts);
        let _ = model.prefill_tokens(&mut s, &prompt, gpu);
        let mut remaining = decode_tokens;
        while remaining > 0 {
            if model.decode_one(&mut s, gpu).is_none() { break; }
            remaining -= 1;
        }
    }

    // Timed iterations.
    let mut prefill_sum_ms = 0.0f64;
    let mut decode_sum_ms = 0.0f64;
    let mut ttft_sum_ms = 0.0f64;
    for _ in 0..iters {
        let mut s = model.new_session(&opts);
        let t_start = Instant::now();
        let _ = model.prefill_tokens(&mut s, &prompt, gpu);
        let t_after_prefill = Instant::now();
        ttft_sum_ms += t_after_prefill.duration_since(t_start).as_secs_f64() * 1000.0;
        prefill_sum_ms += t_after_prefill.duration_since(t_start).as_secs_f64() * 1000.0;

        let t_decode_start = Instant::now();
        let mut decoded = 0usize;
        while decoded < decode_tokens {
            if model.decode_one(&mut s, gpu).is_none() { break; }
            decoded += 1;
        }
        decode_sum_ms += t_decode_start.elapsed().as_secs_f64() * 1000.0;
    }
    let prefill_ms = prefill_sum_ms / iters as f64;
    let total_decode_ms = decode_sum_ms / iters as f64;
    let decode_ms_per_token = if decode_tokens > 0 {
        total_decode_ms / decode_tokens as f64
    } else {
        0.0
    };
    let tokens_per_sec = if decode_ms_per_token > 0.0 {
        1000.0 / decode_ms_per_token
    } else {
        0.0
    };

    BenchResult {
        mode: "decode".to_string(),
        prompt_tokens: prompt_len,
        decode_tokens,
        n_concurrent: 1,
        cache_format: "f32".to_string(),
        prefill_ms,
        decode_ms_per_token,
        tokens_per_sec,
        ttft_ms: ttft_sum_ms / iters as f64,
    }
}

/// Measure decode throughput with multiple concurrent sessions.
///
/// Sessions are run round-robin (same as the server engine): each loop
/// iteration advances every session by one decode step.
pub fn run_concurrency_bench(
    model: &Model,
    n_seqs: usize,
    prompt_len: usize,
    decode_tokens: usize,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let prompt = synthetic_prompt(model, prompt_len);
    let opts = GenerationOptions {
        max_new_tokens: decode_tokens,
        sampler_cfg: SamplerConfig { temperature: 0.0, seed: Some(1), ..Default::default() },
        cache_format: CacheFormat::F32,
        ..Default::default()
    };
    let gpu: &mut Option<&mut GpuBackend> = &mut None;

    let mut run_one = |_| {
        // Prefill all sessions.
        let mut sessions: Vec<_> = (0..n_seqs)
            .map(|_| {
                let mut s = model.new_session(&opts);
                let _ = model.prefill_tokens(&mut s, &prompt, gpu);
                s
            })
            .collect();
        // Round-robin decode.
        let mut step = 0usize;
        let t0 = Instant::now();
        while step < decode_tokens {
            for s in sessions.iter_mut() {
                if !s.is_finished() {
                    let _ = model.decode_one(s, gpu);
                }
            }
            step += 1;
        }
        t0.elapsed().as_secs_f64() * 1000.0
    };

    // Warm up.
    for i in 0..warmup { let _ = run_one(i); }

    // Timed.
    let mut total_ms = 0.0f64;
    for i in 0..iters { total_ms += run_one(i); }
    let avg_ms = total_ms / iters as f64;
    let total_tokens = n_seqs * decode_tokens;
    let tokens_per_sec = (total_tokens as f64) / (avg_ms / 1000.0);
    let decode_ms_per_token = avg_ms / total_tokens as f64;

    BenchResult {
        mode: "concurrency".to_string(),
        prompt_tokens: prompt_len,
        decode_tokens,
        n_concurrent: n_seqs,
        cache_format: "f32".to_string(),
        prefill_ms: 0.0,
        decode_ms_per_token,
        tokens_per_sec,
        ttft_ms: 0.0,
    }
}

/// Compare F32 vs Q8 KV-cache decode throughput.
pub fn run_cache_format_bench(
    model: &Model,
    prompt_len: usize,
    decode_tokens: usize,
    warmup: usize,
    iters: usize,
) -> Vec<BenchResult> {
    let formats = [(CacheFormat::F32, "f32"), (CacheFormat::Q8, "q8")];
    formats.iter().map(|(fmt, name)| {
        let prompt = synthetic_prompt(model, prompt_len);
        let opts = GenerationOptions {
            max_new_tokens: decode_tokens,
            sampler_cfg: SamplerConfig { temperature: 0.0, seed: Some(1), ..Default::default() },
            cache_format: *fmt,
            ..Default::default()
        };
        let gpu: &mut Option<&mut GpuBackend> = &mut None;

        for _ in 0..warmup {
            let mut s = model.new_session(&opts);
            let _ = model.prefill_tokens(&mut s, &prompt, gpu);
            for _ in 0..decode_tokens {
                if model.decode_one(&mut s, gpu).is_none() { break; }
            }
        }

        let mut total_ms = 0.0f64;
        for _ in 0..iters {
            let mut s = model.new_session(&opts);
            let _ = model.prefill_tokens(&mut s, &prompt, gpu);
            let t0 = Instant::now();
            for _ in 0..decode_tokens {
                if model.decode_one(&mut s, gpu).is_none() { break; }
            }
            total_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
        let decode_ms = total_ms / iters as f64;
        let ms_per_token = if decode_tokens > 0 { decode_ms / decode_tokens as f64 } else { 0.0 };
        let tps = if ms_per_token > 0.0 { 1000.0 / ms_per_token } else { 0.0 };

        BenchResult {
            mode: format!("cache-format/{}", name),
            prompt_tokens: prompt_len,
            decode_tokens,
            n_concurrent: 1,
            cache_format: name.to_string(),
            prefill_ms: 0.0,
            decode_ms_per_token: ms_per_token,
            tokens_per_sec: tps,
            ttft_ms: 0.0,
        }
    }).collect()
}

// ── Formatting helpers ────────────────────────────────────────────────────────

/// Print a table of results to stdout.
pub fn print_results(results: &[BenchResult]) {
    println!(
        "{:<22} {:>8} {:>8} {:>6} {:>8} {:>10} {:>12} {:>10}",
        "Mode", "Prompt", "Decode", "Conc", "Cache",
        "Prefill ms", "ms/token", "tok/s"
    );
    println!("{}", "-".repeat(92));
    for r in results {
        let prompt = if r.prompt_tokens > 0 { format!("{}", r.prompt_tokens) } else { "-".to_string() };
        let decode = if r.decode_tokens > 0 { format!("{}", r.decode_tokens) } else { "-".to_string() };
        let prefill = if r.prefill_ms > 0.0 { format!("{:.1}", r.prefill_ms) } else { "-".to_string() };
        let mpt = if r.decode_ms_per_token > 0.0 { format!("{:.2}", r.decode_ms_per_token) } else { "-".to_string() };
        let tps = if r.tokens_per_sec > 0.0 { format!("{:.1}", r.tokens_per_sec) } else { "-".to_string() };
        println!(
            "{:<22} {:>8} {:>8} {:>6} {:>8} {:>10} {:>12} {:>10}",
            r.mode, prompt, decode, r.n_concurrent, r.cache_format,
            prefill, mpt, tps
        );
    }
}

/// Serialize results to JSON.
pub fn results_to_json(results: &[BenchResult]) -> String {
    let entries: Vec<String> = results.iter().map(|r| {
        format!(
            r#"  {{"mode":"{mode}","prompt_tokens":{pt},"decode_tokens":{dt},"n_concurrent":{nc},"cache_format":"{cf}","prefill_ms":{pm:.3},"decode_ms_per_token":{mpt:.3},"tokens_per_sec":{tps:.3},"ttft_ms":{ttft:.3}}}"#,
            mode = r.mode, pt = r.prompt_tokens, dt = r.decode_tokens,
            nc = r.n_concurrent, cf = r.cache_format,
            pm = r.prefill_ms, mpt = r.decode_ms_per_token,
            tps = r.tokens_per_sec, ttft = r.ttft_ms,
        )
    }).collect();
    format!("[\n{}\n]", entries.join(",\n"))
}
