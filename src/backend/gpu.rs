//! GPU compute backend via wgpu (Vulkan / Metal / DX12).
//!
//! `GpuBackend` owns the wgpu device and queue, manages GPU buffers for
//! weight matrices and activation vectors, and dispatches WGSL compute
//! shaders for the hot-path operations (quantized matvec, RMSNorm, RoPE,
//! SiLU, softmax, element-wise add).
//!
//! # Lifecycle
//!
//! 1. `GpuBackend::new()` — request a high-performance adapter, create
//!    device + queue, compile all compute pipelines.
//! 2. `upload_weights()` — copy quantized weight bytes into GPU storage
//!    buffers (done once at model load time).
//! 3. `matvec_q8_0()` / `rms_norm()` / … — enqueue compute dispatches
//!    and read results back to CPU.

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::error::GlintError;
use super::pipeline::{Pipeline, PipelineKind};

/// A handle to a GPU-resident buffer, identified by a user-chosen string key.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct GpuBufferHandle(pub String);

/// GPU compute backend.
pub struct GpuBackend {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: HashMap<PipelineKind, Pipeline>,
    /// Named GPU storage buffers (weights, scratch activations).
    pub(crate) buffers: HashMap<String, wgpu::Buffer>,
    /// Device limits — stored for diagnostic / future validation.
    #[allow(dead_code)]
    pub(crate) limits: wgpu::Limits,
}

impl GpuBackend {
    /// Initialise wgpu: request a high-performance GPU adapter, create the
    /// device + queue, and compile all WGSL compute pipelines.
    pub fn new() -> Result<Self, GlintError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GlintError::GpuAdapterNotFound)?;

        let adapter_info = adapter.get_info();
        eprintln!(
            "[gpu] adapter: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30, // 1 GiB
            max_buffer_size: 2 << 30,                 // 2 GiB
            // Attention shader uses 16 384 B (scores) + 512 B (partial) = 16 896 B
            // of workgroup memory — above the 16 384 B WebGPU baseline.
            max_compute_workgroup_storage_size: 32768, // 32 KiB; supported by all Vulkan/DX12/Metal GPUs
            ..wgpu::Limits::downlevel_defaults()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("glint-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: required_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GlintError::GpuDeviceError(e.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Compile all compute pipelines
        let pipelines = Pipeline::compile_all(&device);

        eprintln!("[gpu] {} compute pipelines compiled", pipelines.len());

        Ok(Self {
            limits: required_limits,
            device,
            queue,
            pipelines,
            buffers: HashMap::new(),
        })
    }

    /// Print adapter info for diagnostics.
    pub fn info_string(&self) -> String {
        format!("GpuBackend {{ pipelines: {} }}", self.pipelines.len())
    }

    // ── Buffer management ─────────────────────────────────────────────

    /// Upload raw bytes into a GPU storage buffer.
    ///
    /// Used at model-load time to transfer quantized weight matrices to the
    /// GPU. The buffer is created with STORAGE | COPY_SRC usage.
    pub fn upload_buffer(&mut self, name: &str, data: &[u8]) {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        self.buffers.insert(name.to_string(), buffer);
    }

    /// Upload f32 data into a GPU storage buffer.
    pub fn upload_f32(&mut self, name: &str, data: &[f32]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        self.buffers.insert(name.to_string(), buffer);
    }

    /// Create a zero-initialised output buffer of `n` f32 elements.
    pub fn create_output_buffer(&mut self, name: &str, n: usize) {
        let size = (n * std::mem::size_of::<f32>()) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.buffers.insert(name.to_string(), buffer);
    }

    /// Read back f32 values from a GPU buffer to CPU.
    pub fn download_f32(&self, name: &str, n: usize) -> Result<Vec<f32>, GlintError> {
        let src_buf = self
            .buffers
            .get(name)
            .ok_or_else(|| GlintError::GpuBufferError(format!("buffer '{name}' not found")))?;

        let size = (n * std::mem::size_of::<f32>()) as u64;

        // Create a staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(src_buf, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GlintError::GpuBufferError(format!("map channel error: {e}")))?
            .map_err(|e| GlintError::GpuBufferError(format!("map error: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    // ── Compute dispatch helpers ──────────────────────────────────────

    /// Dispatch a quantized matvec (Q8_0): `[rows, cols] × [cols] → [rows]`.
    ///
    /// Expects:
    /// - `weight_buf`: name of the GPU buffer holding raw Q8_0 bytes.
    /// - `input`: f32 input vector (uploaded to a temp buffer).
    ///
    /// Returns the result as a CPU Vec<f32>.
    pub fn matvec_q8_0(
        &mut self,
        weight_buf: &str,
        input: &[f32],
        rows: u32,
        cols: u32,
    ) -> Result<Vec<f32>, GlintError> {
        self.dispatch_matvec(PipelineKind::MatvecQ8_0, weight_buf, input, rows, cols)
    }

    /// Dispatch a quantized matvec (Q4_0).
    pub fn matvec_q4_0(
        &mut self,
        weight_buf: &str,
        input: &[f32],
        rows: u32,
        cols: u32,
    ) -> Result<Vec<f32>, GlintError> {
        self.dispatch_matvec(PipelineKind::MatvecQ4_0, weight_buf, input, rows, cols)
    }

    /// Dispatch a quantized matvec (Q4_K, 256-element super-blocks).
    pub fn matvec_q4_k(
        &mut self,
        weight_buf: &str,
        input: &[f32],
        rows: u32,
        cols: u32,
    ) -> Result<Vec<f32>, GlintError> {
        self.dispatch_matvec(PipelineKind::MatvecQ4K, weight_buf, input, rows, cols)
    }

    /// Dispatch a quantized matvec (Q5_K, 256-element super-blocks).
    pub fn matvec_q5_k(
        &mut self,
        weight_buf: &str,
        input: &[f32],
        rows: u32,
        cols: u32,
    ) -> Result<Vec<f32>, GlintError> {
        self.dispatch_matvec(PipelineKind::MatvecQ5K, weight_buf, input, rows, cols)
    }

    /// Dispatch a quantized matvec (Q6_K, 256-element super-blocks).
    pub fn matvec_q6_k(
        &mut self,
        weight_buf: &str,
        input: &[f32],
        rows: u32,
        cols: u32,
    ) -> Result<Vec<f32>, GlintError> {
        self.dispatch_matvec(PipelineKind::MatvecQ6K, weight_buf, input, rows, cols)
    }

    /// Dispatch single-query multi-head attention for the decode step.
    ///
    /// * `q`          — flat query [n_heads * head_dim]
    /// * `k`          — flat keys  [seq_len * n_kv_heads * head_dim]
    /// * `v`          — flat values[seq_len * n_kv_heads * head_dim]
    /// * Returns      — flat attention output [n_heads * head_dim]
    ///
    /// Falls back (returns `Err`) if seq_len > 4096 (shader's MAX_SEQ).
    pub fn attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        scale: f32,
    ) -> Result<Vec<f32>, GlintError> {
        const MAX_SEQ: u32 = 4096;
        if seq_len > MAX_SEQ {
            return Err(GlintError::GpuShaderError(format!(
                "attention: seq_len {seq_len} > MAX_SEQ {MAX_SEQ}"
            )));
        }

        let pipeline = self
            .pipelines
            .get(&PipelineKind::Attention)
            .ok_or_else(|| GlintError::GpuShaderError("Attention pipeline not found".into()))?;

        // Params: [n_heads, n_kv_heads, head_dim, seq_len, scale_bits]
        let params_data: [u32; 5] = [n_heads, n_kv_heads, head_dim, seq_len, scale.to_bits()];
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let q_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-q"),
            contents: bytemuck::cast_slice(q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-k"),
            contents: bytemuck::cast_slice(k),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-v"),
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (n_heads as usize * head_dim as usize) * std::mem::size_of::<f32>();
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn-out"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn-bg"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
            ],
        });

        // One workgroup per query head
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn-dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_heads, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn-staging"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, output_size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GlintError::GpuBufferError(format!("channel: {e}")))?
            .map_err(|e| GlintError::GpuBufferError(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Dispatch an f32 matvec.
    pub fn matvec_f32(
        &mut self,
        weight_buf: &str,
        input: &[f32],
        rows: u32,
        cols: u32,
    ) -> Result<Vec<f32>, GlintError> {
        self.dispatch_matvec(PipelineKind::MatvecF32, weight_buf, input, rows, cols)
    }

    fn dispatch_matvec(
        &mut self,
        kind: PipelineKind,
        weight_buf: &str,
        input: &[f32],
        rows: u32,
        cols: u32,
    ) -> Result<Vec<f32>, GlintError> {
        let pipeline = self
            .pipelines
            .get(&kind)
            .ok_or_else(|| GlintError::GpuShaderError(format!("pipeline {kind:?} not found")))?;

        let weights = self
            .buffers
            .get(weight_buf)
            .ok_or_else(|| {
                GlintError::GpuBufferError(format!("weight buffer '{weight_buf}' not found"))
            })?;

        // Uniform params
        let params = [rows, cols];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matvec-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Input vector buffer
        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matvec-input"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output buffer
        let output_size = (rows as usize) * std::mem::size_of::<f32>();
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matvec-output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec-bg"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch: one workgroup per row
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matvec-dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows, 1, 1);
        }

        // Readback via staging buffer
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GlintError::GpuBufferError(format!("channel: {e}")))?
            .map_err(|e| GlintError::GpuBufferError(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// RMS normalization on GPU.
    pub fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, GlintError> {
        let n = x.len() as u32;
        let pipeline = self
            .pipelines
            .get(&PipelineKind::RmsNorm)
            .ok_or_else(|| GlintError::GpuShaderError("RmsNorm pipeline not found".into()))?;

        // Params: n (u32) + eps (f32)
        let params_data: [u32; 2] = [n, eps.to_bits()];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rms-params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let x_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rms-x"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let w_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rms-w"),
                contents: bytemuck::cast_slice(weight),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_size = (x.len() * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms-bg"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rms-norm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rms-norm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // single workgroup
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GlintError::GpuBufferError(format!("channel: {e}")))?
            .map_err(|e| GlintError::GpuBufferError(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Fused SiLU + element-wise multiply on GPU (SwiGLU FFN).
    pub fn silu_mul(
        &self,
        gate: &[f32],
        up: &[f32],
    ) -> Result<Vec<f32>, GlintError> {
        let n = gate.len() as u32;
        let pipeline = self
            .pipelines
            .get(&PipelineKind::SiluMul)
            .ok_or_else(|| GlintError::GpuShaderError("SiluMul pipeline not found".into()))?;

        let params: [u32; 1] = [n];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("silu-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let gate_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("silu-gate"),
                contents: bytemuck::cast_slice(gate),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let up_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("silu-up"),
                contents: bytemuck::cast_slice(up),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_size = (gate.len() * 4) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("silu-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu-bg"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: up_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = (n + 255) / 256;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("silu-mul"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("silu-mul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GlintError::GpuBufferError(format!("channel: {e}")))?
            .map_err(|e| GlintError::GpuBufferError(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Element-wise addition on GPU.
    pub fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, GlintError> {
        let n = a.len() as u32;
        let pipeline = self
            .pipelines
            .get(&PipelineKind::Add)
            .ok_or_else(|| GlintError::GpuShaderError("Add pipeline not found".into()))?;

        let params: [u32; 1] = [n];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("add-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("add-a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("add-b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_size = (a.len() * 4) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("add-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add-bg"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = (n + 255) / 256;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("add"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("add"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GlintError::GpuBufferError(format!("channel: {e}")))?
            .map_err(|e| GlintError::GpuBufferError(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    // ── GPU-resident attention (GpuKvCache path) ──────────────────────────────

    /// Attention using K/V vectors that are already GPU-resident.
    ///
    /// Unlike [`Self::attention`], this function does **not** upload K/V from
    /// CPU.  Instead it binds the caller-supplied `kv_buf` storage buffers
    /// directly.  The decode step only uploads the new token's K/V slice
    /// (O(`head_dim`)) via [`GpuKvCache::write`], so the total per-step
    /// CPU→GPU bandwidth is `2 * n_kv_heads * head_dim * 4` bytes regardless
    /// of sequence length.
    ///
    /// # Parameters
    ///
    /// * `layer_idx` — which transformer layer this call is for.  Used to
    ///   compute the byte offset into `kv_buf`.
    /// * `window_start` — first position index in the attention window
    ///   (non-zero when sliding-window attention is active).
    /// * `attend_len` — number of K/V positions to attend to.
    pub fn attention_resident(
        &mut self,
        q: &[f32],
        kv_buf: &GpuKvBuffer,
        layer_idx: usize,
        window_start: usize,
        attend_len: usize,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<Vec<f32>, GlintError> {
        const MAX_SEQ: u32 = 4096;
        if attend_len as u32 > MAX_SEQ {
            return Err(GlintError::GpuShaderError(format!(
                "attention_resident: attend_len {attend_len} > MAX_SEQ {MAX_SEQ}"
            )));
        }

        let pipeline = self
            .pipelines
            .get(&PipelineKind::AttentionResident)
            .ok_or_else(|| GlintError::GpuShaderError("AttentionResident pipeline not found".into()))?;

        // kv_layer_off: offset in floats to this layer's start in the K/V buffer.
        let kv_layer_off = (layer_idx * kv_buf.max_seq_len * kv_buf.n_kv_heads * kv_buf.head_dim) as u32;

        // Params: [n_heads, n_kv_heads, head_dim, seq_len (attend_len), scale_bits, kv_layer_off, window_start]
        let params_data: [u32; 7] = [
            n_heads,
            n_kv_heads,
            head_dim,
            attend_len as u32,
            scale.to_bits(),
            kv_layer_off,
            window_start as u32,
        ];
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-res-params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let q_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-res-q"),
            contents: bytemuck::cast_slice(q),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let embed_dim = n_heads as usize * head_dim as usize;
        let output_size = (embed_dim * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn-res-out"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn-res-bg"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kv_buf.k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: kv_buf.v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn-res-dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention-resident"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_heads, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn-res-staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GlintError::GpuBufferError(format!("channel: {e}")))?
            .map_err(|e| GlintError::GpuBufferError(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

// ── GpuKvBuffer ───────────────────────────────────────────────────────────────

/// GPU-resident K/V storage for one sequence's KV cache.
///
/// Layout of `k_buf` (and `v_buf` identically):
/// ```text
/// [ layer 0: max_seq_len * n_kv_heads * head_dim  f32s ]
/// [ layer 1: ...                                        ]
/// ...
/// [ layer n_layers-1: ...                               ]
/// ```
///
/// On each decode step, only the new token's K/V slice (size
/// `n_kv_heads * head_dim`) is written to the buffer via
/// `wgpu::Queue::write_buffer`, making the upload O(`head_dim`) instead of
/// O(`seq_len * head_dim`).
pub struct GpuKvBuffer {
    pub(crate) k_buf: wgpu::Buffer,
    pub(crate) v_buf: wgpu::Buffer,
    pub(crate) n_layers: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) max_seq_len: usize,
}

// ── GpuKvCache ────────────────────────────────────────────────────────────────

/// KV cache that keeps K/V vectors GPU-resident.
///
/// At each decode step `write()` uploads only the **new** K/V slice
/// (O(`n_kv_heads * head_dim`) bytes) to the GPU buffer.  The attention
/// shader then reads K/V entirely from GPU memory without re-uploading the
/// full context.
///
/// A CPU mirror is maintained for:
/// - `read_k_head` / `read_v_head` (used when GPU attention is unavailable)
/// - `export_raw` / `import_raw` (snapshot support)
pub struct GpuKvCache {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue:  Arc<wgpu::Queue>,
    pub(crate) gpu_buf: GpuKvBuffer,
    // CPU mirror — kept in sync with GPU on every write.
    pub(crate) k: Vec<Vec<f32>>,
    pub(crate) v: Vec<Vec<f32>>,
    pub(crate) kv_dim: usize,     // n_kv_heads * head_dim
    pub(crate) max_seq_len: usize,
    pub(crate) len: usize,
}

impl GpuBackend {
    /// Allocate a new [`GpuKvCache`] whose K/V buffers live on this GPU.
    pub fn create_kv_cache(
        &self,
        n_layers: usize,
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> GpuKvCache {
        let kv_dim = n_kv_heads * head_dim;
        let layer_floats = max_seq_len * kv_dim;
        let total_bytes = (n_layers * layer_floats * std::mem::size_of::<f32>()) as u64;

        let make_buf = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: total_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let gpu_buf = GpuKvBuffer {
            k_buf: make_buf("gpu-kv-k"),
            v_buf: make_buf("gpu-kv-v"),
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
        };

        let k = (0..n_layers).map(|_| vec![0.0f32; layer_floats]).collect();
        let v = (0..n_layers).map(|_| vec![0.0f32; layer_floats]).collect();

        GpuKvCache {
            device: Arc::clone(&self.device),
            queue:  Arc::clone(&self.queue),
            gpu_buf,
            k, v, kv_dim, max_seq_len, len: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        // This test will be skipped on machines without a GPU
        match GpuBackend::new() {
            Ok(gpu) => {
                assert!(!gpu.pipelines.is_empty());
                eprintln!("{}", gpu.info_string());
            }
            Err(GlintError::GpuAdapterNotFound) => {
                eprintln!("no GPU adapter found — skipping test");
            }
            Err(e) => panic!("unexpected GPU error: {e}"),
        }
    }

    #[test]
    fn test_gpu_add() {
        let gpu = match GpuBackend::new() {
            Ok(g) => g,
            Err(GlintError::GpuAdapterNotFound) => {
                eprintln!("no GPU — skipping");
                return;
            }
            Err(e) => panic!("{e}"),
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let result = gpu.add(&a, &b).unwrap();
        assert_eq!(result.len(), 4);
        for (i, &v) in result.iter().enumerate() {
            let expected = a[i] + b[i];
            assert!(
                (v - expected).abs() < 1e-5,
                "add[{i}]: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gpu_silu_mul() {
        let gpu = match GpuBackend::new() {
            Ok(g) => g,
            Err(GlintError::GpuAdapterNotFound) => {
                eprintln!("no GPU — skipping");
                return;
            }
            Err(e) => panic!("{e}"),
        };

        let gate = vec![0.0f32, 1.0, -1.0, 2.0];
        let up = vec![1.0f32, 1.0, 1.0, 1.0];
        let result = gpu.silu_mul(&gate, &up).unwrap();

        // SiLU(x) = x / (1 + exp(-x))
        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "silu_mul[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_gpu_matvec_f32() {
        let mut gpu = match GpuBackend::new() {
            Ok(g) => g,
            Err(GlintError::GpuAdapterNotFound) => {
                eprintln!("no GPU — skipping");
                return;
            }
            Err(e) => panic!("{e}"),
        };

        // Simple 2×3 matrix × 3-vec
        let mat: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vec_in: Vec<f32> = vec![1.0, 1.0, 1.0];

        gpu.upload_f32("test-mat", &mat);
        let result = gpu.matvec_f32("test-mat", &vec_in, 2, 3).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-5, "row0: {}", result[0]);
        assert!((result[1] - 15.0).abs() < 1e-5, "row1: {}", result[1]);
    }

    #[test]
    fn test_gpu_rms_norm() {
        let gpu = match GpuBackend::new() {
            Ok(g) => g,
            Err(GlintError::GpuAdapterNotFound) => {
                eprintln!("no GPU — skipping");
                return;
            }
            Err(e) => panic!("{e}"),
        };

        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let eps = 1e-5f32;

        let result = gpu.rms_norm(&x, &w, eps).unwrap();

        // CPU reference
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ss / x.len() as f32 + eps).sqrt();
        let expected: Vec<f32> = x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * inv_rms * wi).collect();

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "rms_norm[{i}]: got {got}, expected {exp}"
            );
        }
    }
}
