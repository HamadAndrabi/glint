//! Compute pipeline compilation and management.
//!
//! Each WGSL shader is embedded at compile time via `include_str!` and
//! compiled into a `wgpu::ComputePipeline` during `GpuBackend::new()`.

use std::collections::HashMap;

/// Identifies which compute pipeline to use.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum PipelineKind {
    MatvecQ8_0,
    MatvecQ4_0,
    MatvecQ4K,
    MatvecQ5K,
    MatvecQ6K,
    MatvecF32,
    Attention,
    /// Attention with GPU-resident K/V buffers (GpuKvCache path).
    AttentionResident,
    RmsNorm,
    SiluMul,
    Rope,
    Softmax,
    Add,
}

/// A compiled compute pipeline.
pub struct Pipeline {
    pub pipeline: wgpu::ComputePipeline,
}

impl Pipeline {
    /// Compile all WGSL shaders into compute pipelines.
    pub fn compile_all(device: &wgpu::Device) -> HashMap<PipelineKind, Pipeline> {
        let shaders: &[(PipelineKind, &str, &str)] = &[
            (
                PipelineKind::MatvecQ8_0,
                include_str!("shaders/matvec_q8_0.wgsl"),
                "main",
            ),
            (
                PipelineKind::MatvecQ4_0,
                include_str!("shaders/matvec_q4_0.wgsl"),
                "main",
            ),
            (
                PipelineKind::MatvecQ4K,
                include_str!("shaders/matvec_q4_k.wgsl"),
                "main",
            ),
            (
                PipelineKind::MatvecQ5K,
                include_str!("shaders/matvec_q5_k.wgsl"),
                "main",
            ),
            (
                PipelineKind::MatvecQ6K,
                include_str!("shaders/matvec_q6_k.wgsl"),
                "main",
            ),
            (
                PipelineKind::MatvecF32,
                include_str!("shaders/matvec_f32.wgsl"),
                "main",
            ),
            (
                PipelineKind::RmsNorm,
                include_str!("shaders/rms_norm.wgsl"),
                "reduce_ss",
            ),
            (
                PipelineKind::SiluMul,
                include_str!("shaders/silu_mul.wgsl"),
                "main",
            ),
            (
                PipelineKind::Rope,
                include_str!("shaders/rope.wgsl"),
                "main",
            ),
            (
                PipelineKind::Softmax,
                include_str!("shaders/softmax.wgsl"),
                "main",
            ),
            (
                PipelineKind::Add,
                include_str!("shaders/add.wgsl"),
                "main",
            ),
            (
                PipelineKind::Attention,
                include_str!("shaders/attention.wgsl"),
                "main",
            ),
            (
                PipelineKind::AttentionResident,
                include_str!("shaders/attention_resident.wgsl"),
                "main",
            ),
        ];

        let mut pipelines = HashMap::new();
        for &(kind, source, entry_point) in shaders {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{kind:?}")),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{kind:?}")),
                layout: None, // auto-layout from shader
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

            pipelines.insert(kind, Pipeline { pipeline });
        }

        pipelines
    }
}
