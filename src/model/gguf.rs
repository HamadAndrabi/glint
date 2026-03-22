//! GGUF file format parser.
//!
//! Parses GGUF v2/v3 model files using memory-mapped I/O and provides
//! access to metadata and tensor data by name.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use thiserror::Error;

// ── Error Types ─────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid GGUF magic number: expected 0x46554747 ('GGUF'), got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("Unsupported GGUF version: {0} (we support version 2 and 3)")]
    UnsupportedVersion(u32),

    #[error("Unknown metadata value type: {0}")]
    UnknownValueType(u32),

    #[error("Unknown tensor type (ggml_type): {0}")]
    UnknownGgmlType(u32),

    #[error("Invalid UTF-8 in string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    #[error("Unexpected end of data at offset {offset} (need {needed} bytes, have {available})")]
    UnexpectedEof {
        offset: usize,
        needed: usize,
        available: usize,
    },

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;

// ── GGML Tensor Types ───────────────────────────────────────────────────────

/// Tensor element data type / quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
}

impl GgmlType {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1M),
            30 => Ok(Self::BF16),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            _ => Err(GgufError::UnknownGgmlType(value)),
        }
    }

    /// Number of elements per quantization block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 | Self::IQ4NL | Self::IQ4XS => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 256,
            Self::IQ3XXS | Self::IQ3S => 256,
            Self::IQ1S | Self::IQ1M => 256,
            Self::TQ1_0 | Self::TQ2_0 => 256,
        }
    }

    /// Bytes consumed per block of elements.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 260,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ1S => 50,
            Self::IQ1M => 56,
            Self::IQ4NL | Self::IQ4XS => 18,
            Self::TQ1_0 => 54,
            Self::TQ2_0 => 66,
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 |
            Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3S => "IQ3_S",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── Metadata Types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataValueType {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufError::UnknownValueType(value)),
        }
    }
}

/// A parsed metadata value from the GGUF key-value store.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::UInt8(_) => "uint8",
            Self::Int8(_) => "int8",
            Self::UInt16(_) => "uint16",
            Self::Int16(_) => "int16",
            Self::UInt32(_) => "uint32",
            Self::Int32(_) => "int32",
            Self::Float32(_) => "float32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::UInt64(_) => "uint64",
            Self::Int64(_) => "int64",
            Self::Float64(_) => "float64",
        }
    }
}

impl std::fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UInt8(v) => write!(f, "{v}"),
            Self::Int8(v) => write!(f, "{v}"),
            Self::UInt16(v) => write!(f, "{v}"),
            Self::Int16(v) => write!(f, "{v}"),
            Self::UInt32(v) => write!(f, "{v}"),
            Self::Int32(v) => write!(f, "{v}"),
            Self::Float32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
            Self::UInt64(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::Float64(v) => write!(f, "{v}"),
            Self::Array(arr) => write!(f, "[array of {} elements]", arr.len()),
        }
    }
}

// ── Tensor Info ─────────────────────────────────────────────────────────────

/// Describes a single tensor in the model file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements (product of dimensions).
    pub fn n_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Total byte size of this tensor's data on disk.
    pub fn data_size(&self) -> usize {
        let n_elements = self.n_elements() as usize;
        let block_size = self.ggml_type.block_size();
        let type_size = self.ggml_type.type_size();
        let n_blocks = n_elements.div_ceil(block_size);
        n_blocks * type_size
    }
}

// ── Cursor (Binary Reader) ─────────────────────────────────────────────────

/// Sequential little-endian reader over a byte slice.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn check_remaining(&self, needed: usize) -> Result<()> {
        if self.pos + needed > self.data.len() {
            Err(GgufError::UnexpectedEof {
                offset: self.pos,
                needed,
                available: self.data.len() - self.pos,
            })
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.check_remaining(1)?;
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    fn read_u16(&mut self) -> Result<u16> {
        self.check_remaining(2)?;
        let val = (&self.data[self.pos..]).read_u16::<LittleEndian>()?;
        self.pos += 2;
        Ok(val)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.check_remaining(4)?;
        let val = (&self.data[self.pos..]).read_u32::<LittleEndian>()?;
        self.pos += 4;
        Ok(val)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.check_remaining(8)?;
        let val = (&self.data[self.pos..]).read_u64::<LittleEndian>()?;
        self.pos += 8;
        Ok(val)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Result<i16> {
        self.check_remaining(2)?;
        let val = (&self.data[self.pos..]).read_i16::<LittleEndian>()?;
        self.pos += 2;
        Ok(val)
    }

    fn read_i32(&mut self) -> Result<i32> {
        self.check_remaining(4)?;
        let val = (&self.data[self.pos..]).read_i32::<LittleEndian>()?;
        self.pos += 4;
        Ok(val)
    }

    fn read_i64(&mut self) -> Result<i64> {
        self.check_remaining(8)?;
        let val = (&self.data[self.pos..]).read_i64::<LittleEndian>()?;
        self.pos += 8;
        Ok(val)
    }

    fn read_f32(&mut self) -> Result<f32> {
        self.check_remaining(4)?;
        let val = (&self.data[self.pos..]).read_f32::<LittleEndian>()?;
        self.pos += 4;
        Ok(val)
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.check_remaining(8)?;
        let val = (&self.data[self.pos..]).read_f64::<LittleEndian>()?;
        self.pos += 8;
        Ok(val)
    }

    /// Read a GGUF string: `[u64 length][UTF-8 bytes]`.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        self.check_remaining(len)?;
        let bytes = self.data[self.pos..self.pos + len].to_vec();
        self.pos += len;
        Ok(String::from_utf8(bytes)?)
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_metadata_value(&mut self, value_type: MetadataValueType) -> Result<MetadataValue> {
        match value_type {
            MetadataValueType::UInt8 => Ok(MetadataValue::UInt8(self.read_u8()?)),
            MetadataValueType::Int8 => Ok(MetadataValue::Int8(self.read_i8()?)),
            MetadataValueType::UInt16 => Ok(MetadataValue::UInt16(self.read_u16()?)),
            MetadataValueType::Int16 => Ok(MetadataValue::Int16(self.read_i16()?)),
            MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(self.read_u32()?)),
            MetadataValueType::Int32 => Ok(MetadataValue::Int32(self.read_i32()?)),
            MetadataValueType::Float32 => Ok(MetadataValue::Float32(self.read_f32()?)),
            MetadataValueType::Bool => Ok(MetadataValue::Bool(self.read_bool()?)),
            MetadataValueType::String => Ok(MetadataValue::String(self.read_string()?)),
            MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(self.read_u64()?)),
            MetadataValueType::Int64 => Ok(MetadataValue::Int64(self.read_i64()?)),
            MetadataValueType::Float64 => Ok(MetadataValue::Float64(self.read_f64()?)),
            MetadataValueType::Array => {
                let element_type = MetadataValueType::from_u32(self.read_u32()?)?;
                let count = self.read_u64()? as usize;
                let mut elements = Vec::with_capacity(count.min(1024 * 1024));
                for _ in 0..count {
                    elements.push(self.read_metadata_value(element_type)?);
                }
                Ok(MetadataValue::Array(elements))
            }
        }
    }

    /// Read a metadata key-value pair: `[string key][u32 type][value]`.
    fn read_metadata_kv(&mut self) -> Result<(String, MetadataValue)> {
        let key = self.read_string()?;
        let value_type = MetadataValueType::from_u32(self.read_u32()?)?;
        let value = self.read_metadata_value(value_type)?;
        Ok((key, value))
    }

    /// Read tensor info: `[string name][u32 n_dims][dims...][u32 type][u64 offset]`.
    fn read_tensor_info(&mut self) -> Result<TensorInfo> {
        let name = self.read_string()?;
        let n_dims = self.read_u32()? as usize;
        let mut dimensions = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dimensions.push(self.read_u64()?);
        }
        let ggml_type = GgmlType::from_u32(self.read_u32()?)?;
        let offset = self.read_u64()?;
        Ok(TensorInfo {
            name,
            dimensions,
            ggml_type,
            offset,
        })
    }
}

// ── GgufModel ───────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x4655_4747; // 'G','G','U','F' as little-endian u32
const DEFAULT_ALIGNMENT: u64 = 32;

/// Backing storage for a `GgufModel` — either memory-mapped (native) or
/// heap-owned (WASM / in-memory usage via `GgufModel::from_bytes`).
enum GgufData {
    Mmap(Mmap),
    Owned(Box<[u8]>),
}

impl GgufData {
    fn as_slice(&self) -> &[u8] {
        match self {
            GgufData::Mmap(m)  => m,
            GgufData::Owned(b) => b,
        }
    }
}

/// A loaded GGUF model.  Backed by a memory map on native platforms or
/// heap-owned bytes when loaded via `GgufModel::from_bytes` (e.g. WASM).
pub struct GgufModel {
    data: GgufData,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensor_infos: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    tensor_data_offset: usize,
    pub version: u32,
}

impl std::fmt::Debug for GgufModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufModel")
            .field("version", &self.version)
            .field("data_size", &self.data.as_slice().len())
            .field("tensor_count", &self.tensor_infos.len())
            .field("metadata_count", &self.metadata.len())
            .field("tensor_data_offset", &self.tensor_data_offset)
            .finish()
    }
}

impl GgufModel {
    /// Shared header + metadata + tensor-info parsing from any byte slice.
    fn parse(bytes: &[u8]) -> Result<(u32, HashMap<String, MetadataValue>, Vec<TensorInfo>, HashMap<String, usize>, usize)> {
        let mut cursor = Cursor::new(bytes);

        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = cursor.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count      = cursor.read_u64()? as usize;
        let metadata_kv_count = cursor.read_u64()? as usize;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let (key, value) = cursor.read_metadata_kv()?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);
        for i in 0..tensor_count {
            let info = cursor.read_tensor_info()?;
            tensor_index.insert(info.name.clone(), i);
            tensor_infos.push(info);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(DEFAULT_ALIGNMENT as u32) as u64;

        let tensor_data_offset = align_offset(cursor.pos as u64, alignment) as usize;
        Ok((version, metadata, tensor_infos, tensor_index, tensor_data_offset))
    }

    /// Load and parse a GGUF model file via memory-mapped I/O (zero-copy).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (version, metadata, tensor_infos, tensor_index, tensor_data_offset) =
            Self::parse(&mmap)?;
        Ok(Self {
            data: GgufData::Mmap(mmap),
            metadata, tensor_infos, tensor_index, tensor_data_offset, version,
        })
    }

    /// Parse a GGUF model from an in-memory byte buffer.
    ///
    /// Use this on WASM where filesystem access is unavailable: fetch the
    /// `.gguf` file as an `ArrayBuffer` in JS, pass it to Rust as `Vec<u8>`.
    ///
    /// The data is moved into the model struct — no extra copy is made.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let (version, metadata, tensor_infos, tensor_index, tensor_data_offset) =
            Self::parse(&bytes)?;
        Ok(Self {
            data: GgufData::Owned(bytes.into_boxed_slice()),
            metadata, tensor_infos, tensor_index, tensor_data_offset, version,
        })
    }

    pub fn alignment(&self) -> u64 {
        self.metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(DEFAULT_ALIGNMENT as u32) as u64
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&idx| &self.tensor_infos[idx])
    }

    /// Get a tensor's raw data as a byte slice from the mmap'd file.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .get_tensor_info(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;

        let start = self.tensor_data_offset + info.offset as usize;
        let size = info.data_size();
        let end = start + size;

        let raw = self.data.as_slice();
        if end > raw.len() {
            return Err(GgufError::UnexpectedEof {
                offset: start,
                needed: size,
                available: raw.len() - start,
            });
        }

        Ok(&raw[start..end])
    }

    pub fn tensor_count(&self) -> usize {
        self.tensor_infos.len()
    }

    pub fn total_tensor_bytes(&self) -> usize {
        self.tensor_infos.iter().map(|t| t.data_size()).sum()
    }

    pub fn total_parameters(&self) -> u64 {
        self.tensor_infos.iter().map(|t| t.n_elements()).sum()
    }
}

/// Align `offset` up to the next multiple of `alignment`.
fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset + (alignment - (offset % alignment)) % alignment
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(63, 32), 64);
        assert_eq!(align_offset(64, 32), 64);
    }

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(8).unwrap(), GgmlType::Q8_0);
        assert!(GgmlType::from_u32(999).is_err());
    }

    #[test]
    fn test_ggml_type_properties() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F32.type_size(), 4);
        assert!(!GgmlType::F32.is_quantized());

        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
        assert!(GgmlType::Q8_0.is_quantized());

        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
        assert!(GgmlType::Q4_0.is_quantized());
    }

    #[test]
    fn test_tensor_info_data_size() {
        let info = TensorInfo {
            name: "test".to_string(),
            dimensions: vec![4096],
            ggml_type: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(info.data_size(), 16384);
        assert_eq!(info.n_elements(), 4096);

        let info_q8 = TensorInfo {
            name: "test_q8".to_string(),
            dimensions: vec![4096],
            ggml_type: GgmlType::Q8_0,
            offset: 0,
        };
        assert_eq!(info_q8.data_size(), 4352);
    }

    #[test]
    fn test_metadata_value_type_from_u32() {
        assert_eq!(MetadataValueType::from_u32(0).unwrap(), MetadataValueType::UInt8);
        assert_eq!(MetadataValueType::from_u32(8).unwrap(), MetadataValueType::String);
        assert_eq!(MetadataValueType::from_u32(9).unwrap(), MetadataValueType::Array);
        assert!(MetadataValueType::from_u32(99).is_err());
    }

    #[test]
    fn test_cursor_read_string() {
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"hello");

        let mut cursor = Cursor::new(&data);
        let s = cursor.read_string().unwrap();
        assert_eq!(s, "hello");
        assert_eq!(cursor.pos, 13);
    }

    #[test]
    fn test_cursor_read_integers() {
        let mut data = Vec::new();
        data.extend_from_slice(&42u32.to_le_bytes());
        data.extend_from_slice(&0xDEADBEEFu64.to_le_bytes());

        let mut cursor = Cursor::new(&data);
        assert_eq!(cursor.read_u32().unwrap(), 42);
        assert_eq!(cursor.read_u64().unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn test_cursor_bounds_checking() {
        let data = [0u8; 2];
        let mut cursor = Cursor::new(&data);
        assert!(cursor.read_u32().is_err());
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, &data).unwrap();

        let result = GgufModel::load(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid GGUF magic"));
    }

    #[test]
    fn test_minimal_valid_gguf() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minimal.gguf");
        std::fs::write(&path, &data).unwrap();

        let model = GgufModel::load(&path).unwrap();
        assert_eq!(model.version, 3);
        assert_eq!(model.tensor_count(), 0);
        assert!(model.metadata.is_empty());
        assert_eq!(model.architecture(), None);
    }

    #[test]
    fn test_gguf_with_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = b"general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&(MetadataValueType::String as u32).to_le_bytes());
        let val = b"llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.gguf");
        std::fs::write(&path, &data).unwrap();

        let model = GgufModel::load(&path).unwrap();
        assert_eq!(model.architecture(), Some("llama"));
    }
}
