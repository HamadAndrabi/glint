# GPU Backend (Vulkan)

Glint includes an optional GPU compute backend using `wgpu`, which targets Vulkan, Metal, and DX12. The same WGSL shaders run on AMD, NVIDIA, Intel, and Apple Silicon GPUs.

Source: `src/backend/gpu.rs`, `src/backend/pipeline.rs`

Build with: `cargo build --release --features vulkan`

---

## Design

The GPU backend follows a simple lifecycle:

1. **Initialize** — `GpuBackend::new()` requests a high-performance adapter, creates the wgpu device and queue, and compiles all WGSL compute pipelines.
2. **Upload** — `weights.upload_all_to_gpu(&mut gpu)` copies quantized weight bytes from the mmap into GPU storage buffers. Done once at model load time.
3. **Dispatch** — Per forward pass, the backend enqueues compute shader dispatches and reads results back to CPU.

The CPU path remains the fallback for any operation not dispatched to GPU.

---

## Initialization

```rust
let gpu = GpuBackend::new()?;
```

The backend requests the highest-performance available adapter across Vulkan, Metal, and DX12:

```rust
wgpu::Instance::new(&wgpu::InstanceDescriptor {
    backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
    ..Default::default()
})
```

Adapter info is logged at startup:
```
[gpu] adapter: AMD Radeon RX 6700 XT (Vulkan)
```

If no compatible adapter is found, `GpuBackend::new()` returns `GlintError::GpuAdapterNotFound`, and the forward pass falls back to CPU silently.

---

## WGSL Compute Shaders

Glint ships 12 WGSL compute shaders covering the hot-path operations:

| Shader | Operation |
|--------|-----------|
| `matvec_f32.wgsl` | f32 matrix-vector multiply |
| `matvec_q4_0.wgsl` | Q4_0 quantized matvec |
| `matvec_q4_k.wgsl` | Q4_K quantized matvec |
| `matvec_q5_k.wgsl` | Q5_K quantized matvec |
| `matvec_q6_k.wgsl` | Q6_K quantized matvec |
| `matvec_q8_0.wgsl` | Q8_0 quantized matvec |
| `rmsnorm.wgsl` | RMS normalization |
| `rope.wgsl` | Rotary positional embeddings |
| `softmax.wgsl` | Numerically stable softmax |
| `attention.wgsl` | Scaled dot-product attention |
| `silu_mul.wgsl` | SiLU activation + element-wise multiply |
| `add.wgsl` | Element-wise addition |

Shaders are compiled to SPIR-V by `wgpu` at pipeline creation time (once per model load). The compiled pipelines are stored in `GpuBackend::pipelines`.

---

## Buffer Management

Weights are stored as named GPU storage buffers:

```rust
pub struct GpuBackend {
    buffers: HashMap<String, wgpu::Buffer>,
    ...
}
```

Weight upload:
```rust
let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some(&name),
    contents: weight_bytes,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
});
buffers.insert(name, buffer);
```

Activation tensors (Q, K, V, intermediate) are uploaded per-call since they change each forward step.

---

## Memory Limits

The backend requests these minimum limits:

```rust
max_storage_buffer_binding_size: 1 GiB
max_buffer_size: 2 GiB
max_compute_workgroup_storage_size: 32 KiB
```

These are satisfied by all Vulkan/DX12/Metal GPUs. WebGPU's baseline 16 KiB workgroup memory limit is intentionally exceeded (the attention shader requires ~16.5 KiB); this is fine for native Vulkan but would need adjustment for WebGPU.

---

## CLI Usage

```bash
# Build with GPU support
cargo build --release --features vulkan

# Run on GPU
glint run -f model.gguf -p "Hello" --gpu
glint chat -f model.gguf --gpu
glint serve -f model.gguf --gpu
```

If the GPU cannot be initialized (no Vulkan driver, unsatisfied limits), Glint falls back to CPU with a warning:

```
Warning: GPU initialization failed (GpuAdapterNotFound), falling back to CPU.
```

If `--gpu` is passed without the `vulkan` feature compiled in:
```
Warning: --gpu requires the `vulkan` feature. Build with: cargo build --features vulkan
Continuing on CPU.
```

---

## Performance Considerations

The GPU backend dispatches individual operations (one compute pass per matvec), which has non-trivial overhead per dispatch. This means:

- For small models (135M, 1B parameters), CPU with AVX2 may be faster due to lower dispatch overhead
- For larger models (7B+), GPU throughput wins — the compute advantage outweighs dispatch latency
- Memory transfer (CPU → GPU for activations, GPU → CPU for outputs) is a bottleneck for batch size 1

To measure, benchmark with and without `--gpu` for your specific model and hardware.
