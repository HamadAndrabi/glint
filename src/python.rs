//! Python bindings for Glint via PyO3.
//!
//! Exposes a single `GlintLLM` class that loads a GGUF model and generates
//! text. Enable with `cargo build --features python` or build via maturin.
//!
//! # Example (Python)
//! ```python
//! import glint
//! llm = glint.GlintLLM("mistral-7b-q4_k.gguf")
//! print(llm.generate("The meaning of life is", max_tokens=100, temperature=0.7))
//! print(llm.model_info())
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::model::tokenizer::Tokenizer;
use crate::sampling::{Sampler, SamplerConfig};
use crate::transformer::{generate_cached, TransformerWeights};

/// A loaded Glint LLM, ready to generate text.
///
/// Loads the full GGUF model on construction; weights are kept in RAM and
/// generation is synchronous (runs on the calling thread).
#[pyclass]
pub struct GlintLLM {
    weights: TransformerWeights,
    config: ModelConfig,
    tokenizer: Tokenizer,
}

#[pymethods]
impl GlintLLM {
    /// Load a GGUF model from `model_path`.
    ///
    /// Raises `ValueError` if the file cannot be read or the metadata is
    /// incomplete.
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let model =
            GgufModel::load(model_path).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let config = ModelConfig::from_metadata(&model.metadata)
            .ok_or_else(|| PyValueError::new_err("Failed to parse model config from metadata"))?;

        let tokenizer =
            Tokenizer::from_gguf(&model).map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Load weights — copies data out of the mmap so the model handle can drop.
        let weights = TransformerWeights::load(&model, &config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            weights,
            config,
            tokenizer,
        })
    }

    /// Generate text continuing `prompt`.
    ///
    /// Returns only the newly generated tokens (not the prompt itself).
    ///
    /// Parameters
    /// ----------
    /// prompt : str
    /// max_tokens : int   (default 256)
    /// temperature : float  0.0 = greedy, >0 = stochastic (default 0.0)
    /// top_k : int          0 = disabled (default)
    /// top_p : float        1.0 = disabled (default)
    /// repeat_penalty : float  1.0 = disabled (default)
    /// seed : int | None    RNG seed; None = random (default)
    #[pyo3(signature = (
        prompt,
        max_tokens = 256,
        temperature = 0.0,
        top_k = 0,
        top_p = 1.0,
        repeat_penalty = 1.0,
        seed = None
    ))]
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repeat_penalty: f32,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let prompt_tokens = self.tokenizer.encode(prompt);
        let eos = self.tokenizer.eos_token_id;

        let mut sampler = Sampler::new(SamplerConfig {
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            ..Default::default()
        });

        let all_tokens = generate_cached(
            &self.weights,
            &self.config,
            &prompt_tokens,
            max_tokens,
            &mut sampler,
            eos,
            &mut None,
        );

        // Return only newly generated tokens (skip the prompt)
        let new_tokens = &all_tokens[prompt_tokens.len()..];
        Ok(self.tokenizer.decode(new_tokens))
    }

    /// Return a dict of model hyperparameters.
    fn model_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("architecture", &self.config.architecture)?;
        d.set_item("context_length", self.config.context_length)?;
        d.set_item("embedding_length", self.config.embedding_length)?;
        d.set_item("block_count", self.config.block_count)?;
        d.set_item("head_count", self.config.head_count)?;
        d.set_item("head_count_kv", self.config.head_count_kv)?;
        d.set_item("vocab_size", self.config.vocab_size)?;
        d.set_item("head_dim", self.config.head_dim())?;
        if let Some(w) = self.config.sliding_window {
            d.set_item("sliding_window", w)?;
        }
        Ok(d)
    }
}

/// Register the `glint` Python module.
#[pymodule]
pub fn glint(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<GlintLLM>()?;
    Ok(())
}
