use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::api::Model as GlintModel;
use crate::cache::KvCache;
use crate::constrained::{ConstraintSpec, VocabIndex};
use crate::model::chat_template::{ChatTemplate, Message};
use crate::model::config::ModelConfig;
use crate::model::tokenizer::Tokenizer;
use crate::sampling::{Sampler, SamplerConfig};
use crate::transformer::{forward_one, forward_prefill, TransformerWeights};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ActiveTab {
    Chat,
    StructuredLab,
    KvTelemetry,
}

#[derive(Clone, Debug)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
    pub tok_per_sec: Option<f64>,
    pub ttft_ms: Option<u64>,
    pub token_count: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum InferenceEvent {
    FirstToken { ttft_ms: u64 },
    Token(String),
    Done {
        total_tokens: usize,
        elapsed_secs: f64,
        tok_per_sec: f64,
    },
    Error(String),
}

pub struct App {
    // Model metadata
    pub model_path: PathBuf,
    pub model_name: String,
    pub architecture: String,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub head_count: usize,

    // Navigation & View
    pub active_tab: ActiveTab,
    pub settings_open: bool,
    pub selected_setting: usize,

    // Chat Tab
    pub messages: Vec<ChatTurn>,
    pub streaming_response: String,
    pub input_text: String,
    pub input_cursor: usize,
    pub chat_scroll: usize,

    // Structured Output Lab Tab
    pub lab_mode: usize, // 0 = JSON Schema, 1 = GBNF Grammar
    pub schema_input: String,
    pub grammar_input: String,
    pub lab_prompt: String,
    pub lab_output: String,
    pub lab_focus: usize, // 0 = Schema/Grammar, 1 = Prompt, 2 = Output

    // KV Cache & Performance Telemetry
    pub total_tokens_generated: usize,
    pub current_tok_per_sec: f64,
    pub current_ttft_ms: u64,
    pub last_elapsed_secs: f64,
    pub kv_used_tokens: usize,

    // Sampling Settings
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
    pub system_prompt: String,

    // Inference worker channels
    pub is_generating: bool,
    cancel_signal: Arc<AtomicBool>,
    event_rx: Option<Receiver<InferenceEvent>>,
    prompt_tx: Option<Sender<InferenceRequest>>,
}

struct InferenceRequest {
    messages: Vec<(String, String)>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    constraint: Option<ConstraintSpec>,
}

impl App {
    pub fn new(
        path: PathBuf,
        system_prompt: Option<String>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repeat_penalty: f32,
        max_tokens: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (config, tokenizer, weights) = load_raw_model(&path)?;
        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Model")
            .to_string();

        let (prompt_tx, prompt_rx) = channel::<InferenceRequest>();
        let (event_tx, event_rx) = channel::<InferenceEvent>();
        let cancel_signal = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel_signal);

        // Spawn dedicated inference thread
        let cfg_clone = Arc::clone(&config);
        let tok_clone = Arc::clone(&tokenizer);
        let w_clone = Arc::clone(&weights);
        thread::spawn(move || {
            run_inference_worker(cfg_clone, tok_clone, w_clone, prompt_rx, event_tx, cancel_clone);
        });

        let mut app = Self {
            model_path: path,
            model_name,
            architecture: config.architecture.clone(),
            context_length: config.context_length as usize,
            embedding_length: config.embedding_length as usize,
            block_count: config.block_count as usize,
            head_count: config.head_count as usize,

            active_tab: ActiveTab::Chat,
            settings_open: false,
            selected_setting: 0,

            messages: Vec::new(),
            streaming_response: String::new(),
            input_text: String::new(),
            input_cursor: 0,
            chat_scroll: 0,

            lab_mode: 0,
            schema_input: r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#.to_string(),
            grammar_input: r#"root ::= "{" ws "\"status\":" ws ("\"ok\"" | "\"error\"") ws "}"
ws   ::= [ \t\n]*"#.to_string(),
            lab_prompt: "Output user profile data:".to_string(),
            lab_output: String::new(),
            lab_focus: 0,

            total_tokens_generated: 0,
            current_tok_per_sec: 0.0,
            current_ttft_ms: 0,
            last_elapsed_secs: 0.0,
            kv_used_tokens: 0,

            temperature,
            top_p,
            top_k,
            repeat_penalty,
            max_tokens,
            system_prompt: system_prompt.unwrap_or_else(|| {
                "You are Glint, an ultra-fast, intelligent AI pair programmer and assistant."
                    .to_string()
            }),

            is_generating: false,
            cancel_signal,
            event_rx: Some(event_rx),
            prompt_tx: Some(prompt_tx),
        };

        // Initialize with default system turn if present
        if !app.system_prompt.is_empty() {
            app.messages.push(ChatTurn {
                role: "system".to_string(),
                content: app.system_prompt.clone(),
                tok_per_sec: None,
                ttft_ms: None,
                token_count: None,
            });
        }

        Ok(app)
    }

    /// Submit current user input for generation.
    pub fn submit_user_message(&mut self) {
        if self.is_generating {
            return;
        }
        let text = self.input_text.trim().to_string();
        if text.is_empty() {
            return;
        }

        self.messages.push(ChatTurn {
            role: "user".to_string(),
            content: text,
            tok_per_sec: None,
            ttft_ms: None,
            token_count: None,
        });
        self.input_text.clear();
        self.input_cursor = 0;
        self.streaming_response.clear();

        let req_messages: Vec<(String, String)> = self
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

        self.send_inference_request(req_messages, None);
    }

    /// Submit structured output lab generation.
    pub fn submit_lab_request(&mut self) {
        if self.is_generating {
            return;
        }
        self.lab_output.clear();

        let constraint = if self.lab_mode == 0 {
            match serde_json::from_str::<serde_json::Value>(&self.schema_input) {
                Ok(schema) => Some(ConstraintSpec::JsonSchema(schema)),
                Err(e) => {
                    self.lab_output = format!("Invalid JSON Schema: {e}");
                    return;
                }
            }
        } else {
            Some(ConstraintSpec::Grammar(self.grammar_input.clone()))
        };

        let req_messages = vec![
            ("system".to_string(), self.system_prompt.clone()),
            ("user".to_string(), self.lab_prompt.clone()),
        ];

        self.send_inference_request(req_messages, constraint);
    }

    fn send_inference_request(
        &mut self,
        messages: Vec<(String, String)>,
        constraint: Option<ConstraintSpec>,
    ) {
        self.is_generating = true;
        self.cancel_signal.store(false, Ordering::SeqCst);

        let req = InferenceRequest {
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty,
            constraint,
        };

        if let Some(tx) = &self.prompt_tx {
            let _ = tx.send(req);
        }
    }

    /// Cancel active generation.
    pub fn cancel_generation(&mut self) {
        if self.is_generating {
            self.cancel_signal.store(true, Ordering::SeqCst);
        }
    }

    /// Poll incoming token events from the inference worker.
    pub fn poll_events(&mut self) {
        let mut finished = false;
        let mut final_tok_s = 0.0;
        let mut final_ttft = 0u64;
        let mut final_tokens = 0usize;

        if let Some(rx) = &self.event_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    InferenceEvent::FirstToken { ttft_ms } => {
                        self.current_ttft_ms = ttft_ms;
                        final_ttft = ttft_ms;
                    }
                    InferenceEvent::Token(tok) => {
                        if self.active_tab == ActiveTab::StructuredLab {
                            self.lab_output.push_str(&tok);
                        } else {
                            self.streaming_response.push_str(&tok);
                        }
                    }
                    InferenceEvent::Done {
                        total_tokens,
                        elapsed_secs,
                        tok_per_sec,
                    } => {
                        finished = true;
                        self.total_tokens_generated += total_tokens;
                        self.current_tok_per_sec = tok_per_sec;
                        self.last_elapsed_secs = elapsed_secs;
                        final_tok_s = tok_per_sec;
                        final_tokens = total_tokens;
                    }
                    InferenceEvent::Error(err) => {
                        finished = true;
                        if self.active_tab == ActiveTab::StructuredLab {
                            self.lab_output.push_str(&format!("\n[Error: {err}]"));
                        } else {
                            self.streaming_response.push_str(&format!("\n[Error: {err}]"));
                        }
                    }
                }
            }
        }

        if finished {
            self.is_generating = false;
            if self.active_tab == ActiveTab::Chat && !self.streaming_response.is_empty() {
                let content = std::mem::take(&mut self.streaming_response);
                self.messages.push(ChatTurn {
                    role: "assistant".to_string(),
                    content,
                    tok_per_sec: Some(final_tok_s),
                    ttft_ms: Some(if final_ttft > 0 { final_ttft } else { self.current_ttft_ms }),
                    token_count: Some(final_tokens),
                });
            }
        }
    }
}

type LoadedModelParts = (Arc<ModelConfig>, Arc<Tokenizer>, Arc<TransformerWeights>);

fn load_raw_model(
    path: &Path,
) -> Result<LoadedModelParts, Box<dyn std::error::Error>> {
    let model = GlintModel::load(path)?;
    Ok((model.config, model.tokenizer, model.weights))
}

fn run_inference_worker(
    config: Arc<ModelConfig>,
    tokenizer: Arc<Tokenizer>,
    weights: Arc<TransformerWeights>,
    rx: Receiver<InferenceRequest>,
    tx: Sender<InferenceEvent>,
    cancel: Arc<AtomicBool>,
) {
    let chat_template = config
        .chat_template
        .as_deref()
        .map(ChatTemplate::detect)
        .unwrap_or(ChatTemplate::Generic);

    let vocab_strings: Vec<String> = (0..tokenizer.vocab_size())
        .map(|i| tokenizer.decode_token(i as u32).to_owned())
        .collect();
    let vocab_index = VocabIndex::from_vocab(&vocab_strings);

    while let Ok(req) = rx.recv() {
        let msgs: Vec<Message<'_>> = req
            .messages
            .iter()
            .map(|(role, content)| Message { role, content })
            .collect();
        let prompt_str = chat_template.apply(&msgs);
        let prompt_tokens = tokenizer.encode_prompt(&prompt_str);

        let mut constraint = req.constraint.as_ref().and_then(|spec| {
            crate::constrained::build_constraint(spec, Arc::clone(&vocab_index)).ok()
        });

        let mut sampler = Sampler::new(SamplerConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            repeat_penalty: req.repeat_penalty,
            ..Default::default()
        });

        let mut cache = KvCache::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );

        let t0 = Instant::now();
        let mut first_token_sent = false;

        // Prefill
        let prefill_logits = forward_prefill(
            &weights,
            &config,
            &prompt_tokens,
            &mut cache,
            0,
            &mut None,
        );

        let mut generated_tokens = Vec::new();
        let mut all_history = prompt_tokens.clone();

        let mut next_token = if let Some(c) = constraint.as_mut() {
            let mask = c.allowed_tokens(&all_history, &vocab_index);
            sampler.sample_constrained(prefill_logits.data(), &all_history, &mask)
        } else {
            sampler.sample(prefill_logits.data(), &all_history)
        };

        if let Some(c) = constraint.as_mut() {
            c.advance(next_token);
        }

        while generated_tokens.len() < req.max_tokens {
            if cancel.load(Ordering::SeqCst) {
                break;
            }

            if !first_token_sent {
                let ttft = t0.elapsed().as_millis() as u64;
                let _ = tx.send(InferenceEvent::FirstToken { ttft_ms: ttft });
                first_token_sent = true;
            }

            if next_token == tokenizer.eos_token_id {
                break;
            }

            generated_tokens.push(next_token);
            all_history.push(next_token);

            let token_str = tokenizer.decode(&[next_token]);
            let _ = tx.send(InferenceEvent::Token(token_str));

            let pos = prompt_tokens.len() + generated_tokens.len() - 1;
            let logits = forward_one(&weights, &config, next_token, pos, &mut cache, &mut None);

            next_token = if let Some(c) = constraint.as_mut() {
                let mask = c.allowed_tokens(&all_history, &vocab_index);
                sampler.sample_constrained(logits.data(), &all_history, &mask)
            } else {
                sampler.sample(logits.data(), &all_history)
            };

            if let Some(c) = constraint.as_mut() {
                c.advance(next_token);
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let tok_s = if elapsed > 0.0 {
            generated_tokens.len() as f64 / elapsed
        } else {
            0.0
        };

        let _ = tx.send(InferenceEvent::Done {
            total_tokens: generated_tokens.len(),
            elapsed_secs: elapsed,
            tok_per_sec: tok_s,
        });
    }
}
