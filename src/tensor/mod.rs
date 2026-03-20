//! Tensor operations module — data structures, math ops, and dequantization.

#[allow(clippy::module_inception)]
mod tensor;
mod ops;
mod dequantize;
mod quantized;
#[cfg(target_arch = "x86_64")]
mod simd;
pub mod flash;

pub use tensor::Tensor;
pub use ops::*;
pub use dequantize::{dequantize, load_tensor_f32};
pub use quantized::QuantizedTensor;
pub use flash::flash_attn_1d;
