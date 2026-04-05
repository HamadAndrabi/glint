//! Tensor operations module — data structures, math ops, and dequantization.

#[allow(clippy::module_inception)]
mod tensor;
mod ops;
mod dequantize;
pub mod quantized;
#[cfg(all(target_arch = "x86_64", feature = "rayon"))]
mod simd;
pub mod flash;

pub use tensor::Tensor;
pub use ops::*;
pub use dequantize::{dequantize, load_tensor_f32};
pub use quantized::{QuantizedStorage, QuantizedTensor, WeightLoadMode};
pub use flash::flash_attn_1d;
