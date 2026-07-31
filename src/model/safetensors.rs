//! SafeTensors file format parser and HuggingFace model-directory loader.
//!
//! The format is deliberately trivial:
//!
//! ```text
//! [u64 LE header_len][header_len bytes of UTF-8 JSON][raw tensor data]
//! ```
//!
//! The JSON header maps each tensor name to `{dtype, shape, data_offsets}`,
//! where `data_offsets` is a `[begin, end)` byte range **relative to the start
//! of the data region** (i.e. relative to `8 + header_len`). One reserved key,
//! `__metadata__`, holds a string→string map instead of a tensor.
//!
//! Like [`crate::model::gguf`], the file is memory-mapped so tensor bytes are
//! never copied during parsing, and every field read out of it is treated as
//! hostile: a model file is untrusted input. See [`SafeTensorsFile::parse`] for
//! the specific bounds that are enforced.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use serde_json::Value;

use crate::error::GlintError;
use crate::model::config::{HfConfig, ModelConfig};
use crate::model::gguf::GgmlType;
use crate::model::tokenizer::Tokenizer;
use crate::tensor::dequantize;

// ── Limits ───────────────────────────────────────────────────────────────────

/// Hard cap on the JSON header, matching the reference implementation.
///
/// The header length is the very first field in the file and is fully
/// attacker-controlled; it is checked against this bound *and* against the real
/// file length before anything is allocated or sliced.
const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;

/// Largest rank Glint will look at. Real LLM weights are 1-D or 2-D; this only
/// exists so a header claiming a million dimensions is rejected outright
/// instead of driving a large allocation while multiplying out its shape.
const MAX_TENSOR_DIMS: usize = 8;

/// Files considered when a directory is opened without an index.
const SAFETENSORS_EXT: &str = "safetensors";

/// Shard map emitted by HF for models split across multiple files.
const INDEX_FILE: &str = "model.safetensors.index.json";

// ── Dtypes ───────────────────────────────────────────────────────────────────

/// A dtype named in a SafeTensors header.
///
/// Every dtype in the specification is recognised so that a *valid* file is
/// never rejected wholesale, but only the three float formats Glint's kernels
/// understand can actually be loaded — [`Dtype::to_ggml`] returns `None` for
/// the rest and the caller reports which tensor was at fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E4M3" => Self::F8E4M3,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "I64" => Self::I64,
            "U64" => Self::U64,
            _ => return None,
        })
    }

    /// Bytes per element.
    pub fn size(&self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E5M2 | Self::F8E4M3 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E4M3 => "F8_E4M3",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::I64 => "I64",
            Self::U64 => "U64",
        }
    }

    /// The equivalent ggml type, when Glint's kernels can consume the bytes
    /// directly. `None` for every non-float dtype.
    ///
    /// F16/BF16 weights are kept in their native width rather than expanded to
    /// f32: `QuantizedTensor`'s fallback matvec dequantizes one row at a time
    /// (exactly as it does for the GGUF quant formats), so keeping the narrow
    /// form halves resident memory instead of doubling it.
    pub fn to_ggml(self) -> Option<GgmlType> {
        match self {
            Self::F32 => Some(GgmlType::F32),
            Self::F16 => Some(GgmlType::F16),
            Self::Bf16 => Some(GgmlType::BF16),
            _ => None,
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── Tensor descriptors ───────────────────────────────────────────────────────

/// One tensor entry from the JSON header, validated against the data region.
#[derive(Debug, Clone)]
pub struct TensorView {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Start of the tensor, relative to the data region.
    pub begin: usize,
    /// End (exclusive) of the tensor, relative to the data region.
    pub end: usize,
}

impl TensorView {
    /// Product of the shape. Always exact: the parser rejects any shape whose
    /// product overflows `usize`.
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.end - self.begin
    }
}

// ── SafeTensorsFile ──────────────────────────────────────────────────────────

/// Backing storage — memory-mapped (native) or heap-owned (in-memory / WASM),
/// mirroring `GgufData`.
enum StData {
    Mmap(Arc<Mmap>),
    Owned(Box<[u8]>),
}

impl StData {
    fn as_slice(&self) -> &[u8] {
        match self {
            StData::Mmap(m) => m,
            StData::Owned(b) => b,
        }
    }
}

/// Header fields shared by the mmap and in-memory constructors.
type ParsedHeader = (
    Vec<TensorView>,
    HashMap<String, usize>,
    usize,
    HashMap<String, String>,
);

/// A single parsed `.safetensors` file.
pub struct SafeTensorsFile {
    data: StData,
    tensors: Vec<TensorView>,
    index: HashMap<String, usize>,
    /// `8 + header_len` — where `TensorView::begin` is measured from.
    data_offset: usize,
    /// The optional `__metadata__` string map (e.g. `{"format": "pt"}`).
    pub metadata: HashMap<String, String>,
}

impl std::fmt::Debug for SafeTensorsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensorsFile")
            .field("file_size", &self.data.as_slice().len())
            .field("tensor_count", &self.tensors.len())
            .field("data_offset", &self.data_offset)
            .finish()
    }
}

fn malformed(detail: impl Into<String>) -> GlintError {
    GlintError::SafeTensorsMalformed(detail.into())
}

impl SafeTensorsFile {
    /// Parse and validate the 8-byte length prefix plus the JSON header.
    ///
    /// Every quantity here comes from the file, so each one is bounded before
    /// it is used:
    ///
    /// * `header_len` is checked against [`MAX_HEADER_LEN`] and against the
    ///   real file length (in u64 arithmetic, so `8 + header_len` cannot wrap)
    ///   before the header slice is taken or handed to `serde_json`.
    /// * a shape's element count is accumulated with `checked_mul`, so a
    ///   hostile `[u64::MAX, u64::MAX]` cannot wrap to a small number that
    ///   would then agree with a small byte range.
    /// * `data_offsets` must satisfy `begin <= end <= data_len` and must span
    ///   exactly `n_elements * dtype.size()` bytes.
    /// * tensors may not overlap, which the spec forbids and which would
    ///   otherwise let one tensor alias another's bytes.
    fn parse(bytes: &[u8]) -> Result<ParsedHeader, GlintError> {
        if bytes.len() < 8 {
            return Err(malformed(format!(
                "file is {} bytes — too short for the 8-byte header length prefix",
                bytes.len()
            )));
        }
        let header_len = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        if header_len > MAX_HEADER_LEN {
            return Err(malformed(format!(
                "JSON header claims {header_len} bytes (limit {MAX_HEADER_LEN})"
            )));
        }
        // u64 arithmetic on both sides: `8 + header_len` is unwrappable here
        // because `header_len` is already bounded well below u64::MAX.
        let header_end = 8 + header_len;
        if header_end > bytes.len() as u64 {
            return Err(malformed(format!(
                "JSON header claims {header_len} bytes but only {} follow the prefix",
                bytes.len() - 8
            )));
        }
        let header_end = header_end as usize;

        let header: Value =
            serde_json::from_slice(&bytes[8..header_end]).map_err(|e| malformed(e.to_string()))?;
        let header = header
            .as_object()
            .ok_or_else(|| malformed("JSON header is not an object"))?;

        let data_len = bytes.len() - header_end;

        let mut metadata = HashMap::new();
        let mut tensors: Vec<TensorView> = Vec::with_capacity(header.len());

        for (name, entry) in header {
            if name == "__metadata__" {
                // Free-form string map; anything else is a malformed header.
                let map = entry
                    .as_object()
                    .ok_or_else(|| malformed("__metadata__ is not an object"))?;
                for (k, v) in map {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_string());
                    }
                }
                continue;
            }

            let entry = entry
                .as_object()
                .ok_or_else(|| malformed(format!("tensor '{name}': entry is not an object")))?;

            let dtype_str = entry
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| malformed(format!("tensor '{name}': missing 'dtype'")))?;
            let dtype = Dtype::parse(dtype_str).ok_or_else(|| {
                malformed(format!("tensor '{name}': unknown dtype '{dtype_str}'"))
            })?;

            let shape_json = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| malformed(format!("tensor '{name}': missing 'shape'")))?;
            if shape_json.len() > MAX_TENSOR_DIMS {
                return Err(malformed(format!(
                    "tensor '{name}': {}-D shape exceeds the {MAX_TENSOR_DIMS}-D limit",
                    shape_json.len()
                )));
            }
            let mut shape = Vec::with_capacity(shape_json.len());
            let mut n_elements: usize = 1;
            for dim in shape_json {
                let dim = dim
                    .as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| {
                        malformed(format!("tensor '{name}': shape entry is not a u64"))
                    })?;
                n_elements = n_elements.checked_mul(dim).ok_or_else(|| {
                    malformed(format!(
                        "tensor '{name}': shape {shape_json:?} overflows usize"
                    ))
                })?;
                shape.push(dim);
            }

            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .filter(|a| a.len() == 2)
                .ok_or_else(|| {
                    malformed(format!(
                        "tensor '{name}': 'data_offsets' must be a 2-element array"
                    ))
                })?;
            let read_offset = |v: &Value| -> Result<usize, GlintError> {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| malformed(format!("tensor '{name}': bad data_offsets entry")))
            };
            let begin = read_offset(&offsets[0])?;
            let end = read_offset(&offsets[1])?;

            if begin > end || end > data_len {
                return Err(malformed(format!(
                    "tensor '{name}': data_offsets [{begin}, {end}) is outside the \
                     {data_len}-byte data region"
                )));
            }
            let expected = n_elements.checked_mul(dtype.size()).ok_or_else(|| {
                malformed(format!("tensor '{name}': size in bytes overflows usize"))
            })?;
            if end - begin != expected {
                return Err(malformed(format!(
                    "tensor '{name}': shape {shape:?} of {dtype} needs {expected} bytes \
                     but data_offsets spans {}",
                    end - begin
                )));
            }

            tensors.push(TensorView {
                name: name.clone(),
                dtype,
                shape,
                begin,
                end,
            });
        }

        // The spec forbids overlapping tensors. Gaps (alignment padding) are
        // tolerated; aliasing is not.
        let mut order: Vec<usize> = (0..tensors.len()).collect();
        order.sort_by_key(|&i| tensors[i].begin);
        for pair in order.windows(2) {
            let (a, b) = (&tensors[pair[0]], &tensors[pair[1]]);
            if b.begin < a.end {
                return Err(malformed(format!(
                    "tensors '{}' and '{}' overlap in the data region",
                    a.name, b.name
                )));
            }
        }

        let index = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok((tensors, index, header_end, metadata))
    }

    /// Memory-map and parse a `.safetensors` file (zero-copy).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GlintError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| GlintError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        // SAFETY: identical contract to `GgufModel::load` — `Mmap::map` is
        // unsafe because the mapped pages alias a file that another process
        // could truncate or rewrite while we hold it, which would fault on
        // access. We accept that standard mmap contract for read-only model
        // artifacts. Everything read out of the map goes through the
        // bounds-checked accessors below (and the header validation in
        // `parse`), so no in-bounds access can escape the mapping.
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| GlintError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?);
        let (tensors, index, data_offset, metadata) = Self::parse(&mmap)?;
        Ok(Self {
            data: StData::Mmap(mmap),
            tensors,
            index,
            data_offset,
            metadata,
        })
    }

    /// Parse a `.safetensors` image already in memory (tests, WASM).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GlintError> {
        let (tensors, index, data_offset, metadata) = Self::parse(&bytes)?;
        Ok(Self {
            data: StData::Owned(bytes.into_boxed_slice()),
            tensors,
            index,
            data_offset,
            metadata,
        })
    }

    pub fn tensor_infos(&self) -> &[TensorView] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&TensorView> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Raw bytes of one tensor.
    ///
    /// The range was validated at parse time; it is re-derived with checked
    /// arithmetic here so that a future change to `data_offset` cannot turn
    /// into an out-of-bounds slice.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GlintError> {
        let view = self
            .get(name)
            .ok_or_else(|| GlintError::TensorNotFound(name.to_string()))?;
        let raw = self.data.as_slice();
        let start = self
            .data_offset
            .checked_add(view.begin)
            .filter(|&s| s <= raw.len())
            .ok_or_else(|| malformed(format!("tensor '{name}': start offset out of range")))?;
        let end = start
            .checked_add(view.nbytes())
            .filter(|&e| e <= raw.len())
            .ok_or_else(|| malformed(format!("tensor '{name}': end offset out of range")))?;
        Ok(&raw[start..end])
    }

    /// Total mapped size in bytes.
    pub fn byte_len(&self) -> usize {
        self.data.as_slice().len()
    }
}

// ── SafeTensorsModel (one or more shards) ────────────────────────────────────

/// One or more `.safetensors` files presented as a single tensor namespace.
///
/// HF splits large checkpoints into `model-00001-of-000NN.safetensors` shards
/// listed by `model.safetensors.index.json`; a small model is a single
/// `model.safetensors`. Both are handled here.
pub struct SafeTensorsModel {
    shards: Vec<SafeTensorsFile>,
    /// tensor name → index into `shards`.
    index: HashMap<String, usize>,
}

impl std::fmt::Debug for SafeTensorsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensorsModel")
            .field("shards", &self.shards.len())
            .field("tensors", &self.index.len())
            .field("bytes", &self.byte_len())
            .finish()
    }
}

impl SafeTensorsModel {
    /// Open a `.safetensors` file, or a directory holding one or more of them.
    ///
    /// Passing a single shard of a sharded checkpoint opens the whole set: a
    /// shard on its own is never a complete model.
    pub fn open(path: &Path) -> Result<Self, GlintError> {
        if path.is_dir() {
            return Self::open_dir(path);
        }
        if let Some(parent) = path.parent() {
            if parent.join(INDEX_FILE).is_file() {
                return Self::open_dir(parent);
            }
        }
        Self::from_files(vec![SafeTensorsFile::load(path)?])
    }

    /// Open every shard in a HuggingFace model directory.
    pub fn open_dir(dir: &Path) -> Result<Self, GlintError> {
        let index_path = dir.join(INDEX_FILE);
        let files = if index_path.is_file() {
            shard_files_from_index(&index_path)?
                .into_iter()
                .map(|name| dir.join(name))
                .collect()
        } else {
            let mut found: Vec<PathBuf> = read_dir_sorted(dir)?
                .into_iter()
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(SAFETENSORS_EXT))
                .collect();
            found.sort();
            found
        };

        if files.is_empty() {
            return Err(GlintError::HfMissingFile {
                dir: dir.display().to_string(),
                file: format!("*.{SAFETENSORS_EXT}"),
            });
        }

        let mut shards = Vec::with_capacity(files.len());
        for path in files {
            shards.push(SafeTensorsFile::load(&path)?);
        }
        Self::from_files(shards)
    }

    /// Build the shared name→shard index, rejecting names claimed twice.
    pub fn from_files(shards: Vec<SafeTensorsFile>) -> Result<Self, GlintError> {
        let mut index = HashMap::new();
        for (shard_idx, shard) in shards.iter().enumerate() {
            for view in &shard.tensors {
                if index.insert(view.name.clone(), shard_idx).is_some() {
                    return Err(malformed(format!(
                        "tensor '{}' appears in more than one shard",
                        view.name
                    )));
                }
            }
        }
        Ok(Self { shards, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&TensorView> {
        let &shard = self.index.get(name)?;
        self.shards[shard].get(name)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GlintError> {
        let &shard = self
            .index
            .get(name)
            .ok_or_else(|| GlintError::TensorNotFound(name.to_string()))?;
        self.shards[shard].tensor_bytes(name)
    }

    /// Dequantize one tensor to f32 regardless of its stored width.
    ///
    /// Used for the small norm vectors, which the forward pass consumes as
    /// plain f32 `Tensor`s.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, GlintError> {
        let view = self
            .get(name)
            .ok_or_else(|| GlintError::TensorNotFound(name.to_string()))?;
        let ggml_type =
            view.dtype
                .to_ggml()
                .ok_or_else(|| GlintError::SafeTensorsUnsupportedDtype {
                    name: name.to_string(),
                    dtype: view.dtype.name().to_string(),
                })?;
        let bytes = self.tensor_bytes(name)?;
        Ok(dequantize(bytes, ggml_type, view.n_elements()))
    }

    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Sum of the shard file sizes.
    pub fn byte_len(&self) -> usize {
        self.shards.iter().map(|s| s.byte_len()).sum()
    }

    /// Tensor names, sorted — stable output for `glint inspect`.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.index.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }
}

/// Read the shard file names out of `model.safetensors.index.json`.
///
/// The names come from the file, so they are constrained to bare file names:
/// a `weight_map` entry of `../../etc/passwd` must not be turned into a path
/// this loader then opens.
fn shard_files_from_index(index_path: &Path) -> Result<Vec<String>, GlintError> {
    let text = read_to_string(index_path)?;
    let value: Value = serde_json::from_str(&text).map_err(|e| GlintError::HfInvalidJson {
        file: index_path.display().to_string(),
        detail: e.to_string(),
    })?;
    let weight_map = value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| GlintError::HfInvalidJson {
            file: index_path.display().to_string(),
            detail: "missing 'weight_map' object".to_string(),
        })?;

    let mut files: Vec<String> = Vec::new();
    for name in weight_map.values() {
        let name = name.as_str().ok_or_else(|| GlintError::HfInvalidJson {
            file: index_path.display().to_string(),
            detail: "weight_map values must be file names".to_string(),
        })?;
        if Path::new(name).file_name().and_then(|n| n.to_str()) != Some(name) {
            return Err(GlintError::HfInvalidJson {
                file: index_path.display().to_string(),
                detail: format!("weight_map entry '{name}' is not a plain file name"),
            });
        }
        if !files.iter().any(|f| f == name) {
            files.push(name.to_string());
        }
    }
    files.sort();
    Ok(files)
}

fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>, GlintError> {
    let entries = std::fs::read_dir(dir).map_err(|e| GlintError::Io {
        path: dir.display().to_string(),
        detail: e.to_string(),
    })?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| GlintError::Io {
            path: dir.display().to_string(),
            detail: e.to_string(),
        })?;
        paths.push(entry.path());
    }
    paths.sort();
    Ok(paths)
}

fn read_to_string(path: &Path) -> Result<String, GlintError> {
    std::fs::read_to_string(path).map_err(|e| GlintError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

// ── HuggingFace model directory ──────────────────────────────────────────────

/// Everything a HuggingFace model directory contributes to a Glint model:
/// hyperparameters, tokenizer, and the mapped weight files.
///
/// Turning the weights into [`crate::transformer::TransformerWeights`] is the
/// caller's next step (`TransformerWeights::from_safetensors`), which keeps
/// this module free of any dependency on the transformer layer.
pub struct HfModelDir {
    /// Directory the files were read from.
    pub root: PathBuf,
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub weights: SafeTensorsModel,
    /// `tie_word_embeddings` from `config.json` — when true, `lm_head.weight`
    /// is absent and the embedding table doubles as the output projection.
    pub tie_word_embeddings: bool,
}

impl HfModelDir {
    /// Load `config.json`, `tokenizer.json`, and the weight shards.
    ///
    /// `path` may be the directory itself or any `.safetensors` file inside it.
    pub fn open(path: &Path) -> Result<Self, GlintError> {
        let root = model_root(path)?;

        let config_path = root.join("config.json");
        if !config_path.is_file() {
            return Err(GlintError::HfMissingFile {
                dir: root.display().to_string(),
                file: "config.json".to_string(),
            });
        }
        let hf = HfConfig::from_json(&read_to_string(&config_path)?)?;
        let mut config = hf.config;

        let tokenizer_path = root.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(GlintError::HfMissingFile {
                dir: root.display().to_string(),
                file: "tokenizer.json".to_string(),
            });
        }
        let tokenizer_config = optional_file(&root.join("tokenizer_config.json"))?;
        let tokenizer = Tokenizer::from_hf_json(
            &read_to_string(&tokenizer_path)?,
            tokenizer_config.as_deref(),
            hf.bos_token_id,
            hf.eos_token_id,
        )?;

        config.chat_template = chat_template(&root, tokenizer_config.as_deref());

        let weights = SafeTensorsModel::open(path)?;

        Ok(Self {
            root,
            config,
            tokenizer,
            weights,
            tie_word_embeddings: hf.tie_word_embeddings,
        })
    }
}

/// Directory holding a model's JSON sidecars, given a path to the directory or
/// to one of its `.safetensors` files.
fn model_root(path: &Path) -> Result<PathBuf, GlintError> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    match path.parent() {
        // `Path::parent` of a bare file name is the empty path.
        Some(p) if !p.as_os_str().is_empty() => Ok(p.to_path_buf()),
        _ => Ok(PathBuf::from(".")),
    }
}

/// Read a file that is allowed to be absent.
fn optional_file(path: &Path) -> Result<Option<String>, GlintError> {
    if path.is_file() {
        Ok(Some(read_to_string(path)?))
    } else {
        Ok(None)
    }
}

/// Locate the raw Jinja chat template.
///
/// Newer repos ship `chat_template.jinja` next to the tokenizer; older ones
/// embed the string in `tokenizer_config.json` (occasionally as a list of
/// named templates, in which case the `default` entry — or the first — wins).
fn chat_template(root: &Path, tokenizer_config: Option<&str>) -> Option<String> {
    if let Ok(text) = std::fs::read_to_string(root.join("chat_template.jinja")) {
        return Some(text);
    }
    let value: Value = serde_json::from_str(tokenizer_config?).ok()?;
    let template = value.get("chat_template")?;
    if let Some(s) = template.as_str() {
        return Some(s.to_string());
    }
    let entries = template.as_array()?;
    let chosen = entries
        .iter()
        .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
        .or_else(|| entries.first())?;
    Some(chosen.get("template")?.as_str()?.to_string())
}

/// Does this path look like a HuggingFace safetensors model rather than a GGUF
/// file? Used by the CLI and [`crate::api::Model::load`] to pick a loader.
pub fn is_safetensors_path(path: &Path) -> bool {
    if path.extension().and_then(|e| e.to_str()) == Some(SAFETENSORS_EXT) {
        return true;
    }
    if !path.is_dir() {
        return false;
    }
    if path.join(INDEX_FILE).is_file() || path.join("model.safetensors").is_file() {
        return true;
    }
    read_dir_sorted(path)
        .map(|entries| {
            entries
                .iter()
                .any(|p| p.extension().and_then(|e| e.to_str()) == Some(SAFETENSORS_EXT))
        })
        .unwrap_or(false)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod test_support {
    //! Helpers for building `.safetensors` images in tests. Also used by the
    //! weight-loading tests in `transformer::weights`.

    /// One tensor to serialise: `(name, shape, f32 values)`.
    pub struct TensorSpec {
        pub name: String,
        pub shape: Vec<usize>,
        pub data: Vec<f32>,
    }

    pub fn spec(name: &str, shape: &[usize], data: Vec<f32>) -> TensorSpec {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        TensorSpec {
            name: name.to_string(),
            shape: shape.to_vec(),
            data,
        }
    }

    /// Hand-write a `.safetensors` image: length prefix, JSON header, data.
    ///
    /// Deliberately does not share code with the parser — the tests should
    /// fail if either side drifts.
    pub fn build_f32(specs: &[TensorSpec]) -> Vec<u8> {
        let mut entries: Vec<String> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for s in specs {
            let begin = data.len();
            for v in &s.data {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let shape: Vec<String> = s.shape.iter().map(|d| d.to_string()).collect();
            entries.push(format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                s.name,
                shape.join(","),
                begin,
                data.len()
            ));
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut out = Vec::with_capacity(8 + header.len() + data.len());
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        out
    }

    /// Deterministic pseudo-random values in `[-0.5, 0.5)` — a tiny LCG so the
    /// fixtures need no `rand` dependency.
    pub fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    fn tiny_image() -> Vec<u8> {
        build_f32(&[
            spec("a", &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            spec("b", &[2], vec![-1.5, 0.25]),
        ])
    }

    #[test]
    fn test_parse_tensors_shapes_and_values() {
        let st = SafeTensorsFile::from_bytes(tiny_image()).unwrap();
        assert_eq!(st.tensor_infos().len(), 2);

        let a = st.get("a").unwrap();
        assert_eq!(a.dtype, Dtype::F32);
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.n_elements(), 6);
        assert_eq!(a.nbytes(), 24);
        assert_eq!(a.begin, 0);

        let model = SafeTensorsModel::from_files(vec![st]).unwrap();
        assert_eq!(
            model.tensor_f32("a").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(model.tensor_f32("b").unwrap(), vec![-1.5, 0.25]);
        assert_eq!(model.names(), vec!["a", "b"]);
        assert!(model.contains("a"));
        assert!(!model.contains("missing"));
        assert!(matches!(
            model.tensor_f32("missing"),
            Err(GlintError::TensorNotFound(_))
        ));
    }

    #[test]
    fn test_load_from_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, tiny_image()).unwrap();

        let st = SafeTensorsFile::load(&path).unwrap();
        assert_eq!(st.tensor_infos().len(), 2);
        assert_eq!(st.tensor_bytes("b").unwrap().len(), 8);
    }

    #[test]
    fn test_metadata_map_is_parsed_and_skipped_as_a_tensor() {
        let header = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        let st = SafeTensorsFile::from_bytes(bytes).unwrap();
        assert_eq!(st.tensor_infos().len(), 1);
        assert_eq!(st.metadata.get("format").map(|s| s.as_str()), Some("pt"));
    }

    #[test]
    fn test_f16_and_bf16_dequantize_to_f32() {
        // 1.5 and -2.0 in both half formats, written by hand.
        let header = r#"{"h":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(1.5).to_le_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(-2.0).to_le_bytes());
        // BF16 = the top 16 bits of the f32 bit pattern.
        bytes.extend_from_slice(&1.5f32.to_le_bytes()[2..4]);
        bytes.extend_from_slice(&(-2.0f32).to_le_bytes()[2..4]);

        let model = SafeTensorsModel::from_files(vec![SafeTensorsFile::from_bytes(bytes).unwrap()])
            .unwrap();
        assert_eq!(model.tensor_f32("h").unwrap(), vec![1.5, -2.0]);
        assert_eq!(model.tensor_f32("b").unwrap(), vec![1.5, -2.0]);
    }

    #[test]
    fn test_unsupported_dtype_names_the_tensor() {
        let header = r#"{"ids":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 16]);

        // A valid I64 tensor parses (the file is well-formed) but cannot load.
        let model = SafeTensorsModel::from_files(vec![SafeTensorsFile::from_bytes(bytes).unwrap()])
            .unwrap();
        let err = model.tensor_f32("ids").unwrap_err();
        assert!(err.to_string().contains("I64"), "got: {err}");
    }

    // ── Adversarial input ────────────────────────────────────────────────────
    // A model file is untrusted input. Every one of these must return an error
    // — never panic, never allocate on an unchecked length.

    #[test]
    fn test_empty_and_short_files_are_rejected() {
        assert!(SafeTensorsFile::from_bytes(Vec::new()).is_err());
        assert!(SafeTensorsFile::from_bytes(vec![0u8; 7]).is_err());
    }

    #[test]
    fn test_header_length_larger_than_file_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&64u64.to_le_bytes());
        bytes.extend_from_slice(b"{}");
        let err = SafeTensorsFile::from_bytes(bytes).unwrap_err();
        assert!(err.to_string().contains("only 2 follow"), "got: {err}");
    }

    #[test]
    fn test_absurd_header_length_does_not_allocate() {
        // u64::MAX would wrap `8 + header_len` on a 64-bit add; the bound
        // check must run first.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(b"{}");
        let err = SafeTensorsFile::from_bytes(bytes).unwrap_err();
        assert!(err.to_string().contains("limit"), "got: {err}");
    }

    #[test]
    fn test_truncated_data_region_is_rejected() {
        let mut bytes = tiny_image();
        bytes.truncate(bytes.len() - 4);
        assert!(SafeTensorsFile::from_bytes(bytes).is_err());
    }

    #[test]
    fn test_out_of_bounds_data_offsets_are_rejected() {
        let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 8]); // only half the promised bytes
        let err = SafeTensorsFile::from_bytes(bytes).unwrap_err();
        assert!(err.to_string().contains("outside the"), "got: {err}");
    }

    #[test]
    fn test_reversed_data_offsets_are_rejected() {
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        assert!(SafeTensorsFile::from_bytes(bytes).is_err());
    }

    #[test]
    fn test_shape_disagreeing_with_byte_range_is_rejected() {
        let header = r#"{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let err = SafeTensorsFile::from_bytes(bytes).unwrap_err();
        assert!(err.to_string().contains("needs 24 bytes"), "got: {err}");
    }

    #[test]
    fn test_overflowing_shape_is_rejected() {
        let header = format!(
            r#"{{"a":{{"dtype":"F32","shape":[{m},{m}],"data_offsets":[0,4]}}}}"#,
            m = u64::MAX
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        assert!(SafeTensorsFile::from_bytes(bytes).is_err());
    }

    #[test]
    fn test_absurd_rank_is_rejected() {
        let dims = ["1"; MAX_TENSOR_DIMS + 1].join(",");
        let header = format!(r#"{{"a":{{"dtype":"F32","shape":[{dims}],"data_offsets":[0,4]}}}}"#);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        let err = SafeTensorsFile::from_bytes(bytes).unwrap_err();
        assert!(err.to_string().contains("limit"), "got: {err}");
    }

    #[test]
    fn test_unknown_dtype_is_rejected() {
        let header = r#"{"a":{"dtype":"FP4","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        let err = SafeTensorsFile::from_bytes(bytes).unwrap_err();
        assert!(
            err.to_string().contains("unknown dtype 'FP4'"),
            "got: {err}"
        );
    }

    #[test]
    fn test_overlapping_tensors_are_rejected() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 12]);
        let err = SafeTensorsFile::from_bytes(bytes).unwrap_err();
        assert!(err.to_string().contains("overlap"), "got: {err}");
    }

    #[test]
    fn test_non_object_header_is_rejected() {
        for header in ["[]", "12", "\"x\"", "not json"] {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            assert!(
                SafeTensorsFile::from_bytes(bytes).is_err(),
                "header {header:?} should be rejected"
            );
        }
    }

    #[test]
    fn test_malformed_entries_are_rejected() {
        let headers = [
            r#"{"a":5}"#,
            r#"{"a":{"shape":[1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
            r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,"4"]}}"#,
        ];
        for header in headers {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0u8; 8]);
            assert!(
                SafeTensorsFile::from_bytes(bytes).is_err(),
                "header {header:?} should be rejected"
            );
        }
    }

    // ── Sharding ─────────────────────────────────────────────────────────────

    #[test]
    fn test_sharded_directory_is_loaded_through_the_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            build_f32(&[spec("a", &[2], vec![1.0, 2.0])]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            build_f32(&[spec("b", &[2], vec![3.0, 4.0])]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join(INDEX_FILE),
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let model = SafeTensorsModel::open(dir.path()).unwrap();
        assert_eq!(model.shard_count(), 2);
        assert_eq!(model.tensor_count(), 2);
        assert_eq!(model.tensor_f32("a").unwrap(), vec![1.0, 2.0]);
        assert_eq!(model.tensor_f32("b").unwrap(), vec![3.0, 4.0]);

        // Opening a single shard pulls in the whole set.
        let via_shard =
            SafeTensorsModel::open(&dir.path().join("model-00001-of-00002.safetensors")).unwrap();
        assert_eq!(via_shard.tensor_count(), 2);
    }

    #[test]
    fn test_sharded_directory_without_an_index_globs_the_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            build_f32(&[spec("a", &[1], vec![7.0])]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            build_f32(&[spec("b", &[1], vec![8.0])]),
        )
        .unwrap();

        let model = SafeTensorsModel::open(dir.path()).unwrap();
        assert_eq!(model.tensor_count(), 2);
    }

    #[test]
    fn test_duplicate_tensor_across_shards_is_rejected() {
        let a = SafeTensorsFile::from_bytes(build_f32(&[spec("a", &[1], vec![1.0])])).unwrap();
        let b = SafeTensorsFile::from_bytes(build_f32(&[spec("a", &[1], vec![2.0])])).unwrap();
        let err = SafeTensorsModel::from_files(vec![a, b]).unwrap_err();
        assert!(
            err.to_string().contains("more than one shard"),
            "got: {err}"
        );
    }

    #[test]
    fn test_index_with_a_traversing_file_name_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(INDEX_FILE),
            r#"{"weight_map":{"a":"../escape.safetensors"}}"#,
        )
        .unwrap();
        let err = SafeTensorsModel::open(dir.path()).unwrap_err();
        assert!(err.to_string().contains("plain file name"), "got: {err}");
    }

    #[test]
    fn test_directory_without_weights_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let err = SafeTensorsModel::open(dir.path()).unwrap_err();
        assert!(err.to_string().contains("safetensors"), "got: {err}");
    }

    #[test]
    fn test_is_safetensors_path() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_safetensors_path(dir.path()));
        std::fs::write(
            dir.path().join("model.safetensors"),
            build_f32(&[spec("a", &[1], vec![1.0])]),
        )
        .unwrap();
        assert!(is_safetensors_path(dir.path()));
        assert!(is_safetensors_path(&dir.path().join("model.safetensors")));
        assert!(!is_safetensors_path(Path::new("model.gguf")));
    }
}
