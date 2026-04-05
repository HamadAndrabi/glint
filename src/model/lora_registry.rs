//! Registry of named LoRA adapters for per-request selection.
//!
//! Adapters are loaded once at startup (or via [`AdapterRegistry::register`])
//! and looked up by name at inference time.  Each registered adapter is wrapped
//! in an `Arc` so it can be shared cheaply across sessions.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::GlintError;
use crate::model::gguf::GgufModel;
use crate::model::lora::LoraWeights;

/// Registry of named LoRA adapters.
///
/// Build once and share behind `Arc<RwLock<_>>` for concurrent access in
/// server contexts, or own directly inside [`crate::api::Model`] for
/// single-threaded library use.
pub struct AdapterRegistry {
    adapters: HashMap<String, Arc<LoraWeights>>,
}

impl AdapterRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { adapters: HashMap::new() }
    }

    /// Load a GGUF LoRA adapter file and add it to the registry under `name`.
    ///
    /// `n_layers` must match the base model's layer count so that per-layer
    /// adapter arrays are allocated at the right size.
    pub fn register(
        &mut self,
        name: &str,
        path: &Path,
        n_layers: usize,
    ) -> Result<(), GlintError> {
        let gguf = GgufModel::load(path)
            .map_err(|e| GlintError::TensorReadError {
                name: name.to_string(),
                detail: e.to_string(),
            })?;
        let weights = LoraWeights::load(&gguf, n_layers)
            .ok_or_else(|| GlintError::TensorReadError {
                name: name.to_string(),
                detail: "no lora_a/lora_b tensors found in adapter file".to_string(),
            })?;
        self.adapters.insert(name.to_string(), Arc::new(weights));
        Ok(())
    }

    /// Look up an adapter by name.  Returns `None` if not registered.
    pub fn get(&self, name: &str) -> Option<Arc<LoraWeights>> {
        self.adapters.get(name).cloned()
    }

    /// Number of registered adapters.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// True when no adapters have been registered.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_empty() {
        let reg = AdapterRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn test_registry_missing_returns_none() {
        let mut reg = AdapterRegistry::new();
        // Only way to test get() without a real GGUF file is to confirm None.
        assert!(reg.get("not-registered").is_none());
        // Silence unused mut warning.
        let _ = &mut reg;
    }
}
