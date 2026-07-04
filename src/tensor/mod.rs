//! Tensor operations module — data structures, math ops, and dequantization.

mod dequantize;
pub mod flash;
mod ops;
pub mod quantized;
#[cfg(all(target_arch = "x86_64", feature = "rayon"))]
mod simd;
#[allow(clippy::module_inception)]
mod tensor;

pub use dequantize::{dequantize, load_tensor_f32};
pub use flash::flash_attn_1d;
pub use ops::*;
pub use quantized::{QuantizedStorage, QuantizedTensor, WeightLoadMode};
pub use tensor::Tensor;
