# Browser (WASM)

Glint can be compiled to WebAssembly and run entirely in the browser — no server required. This makes it possible to build fully client-side LLM applications.

Source: `src/wasm.rs`, `demo/index.html`, `demo/worker.js`

Build with: `wasm-pack build --target web --features wasm`

---

## Building

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the WASM module
wasm-pack build --target web --features wasm

# Output: pkg/
#   glint.js          — JS bindings
#   glint_bg.wasm     — compiled Rust binary
#   glint.d.ts        — TypeScript type definitions
```

---

## JavaScript API

### `GlintModel`

```js
import init, { GlintModel, init_panic_hook } from './pkg/glint.js';

// Initialize the WASM module once
await init();
init_panic_hook();  // Optional: routes Rust panics to console.error

// Load a model from a Uint8Array
const resp  = await fetch('model.gguf');
const bytes = new Uint8Array(await resp.arrayBuffer());
const model = new GlintModel(bytes);
```

#### `model.generate(prompt, maxTokens, temperature) → string`

Synchronous generation. Blocks until all tokens are generated.

```js
const output = model.generate("The meaning of life is", 64, 0.8);
console.log(output);  // " to find purpose and connection..."
```

Parameters:
- `prompt: string` — input text
- `maxTokens: number` — maximum new tokens
- `temperature: number` — sampling temperature; 0.0 = greedy

Returns only the newly generated text (not the prompt).

#### `model.generate_streaming(prompt, maxTokens, temperature, onToken)`

Streaming generation with a per-token callback. Designed for use in a Web Worker — each call to `onToken` can `postMessage` to the main thread for incremental display.

```js
model.generate_streaming(prompt, 256, 0.8, (text) => {
    self.postMessage({ type: 'token', text });
});
```

#### `model.context_length() → number`

Maximum tokens the model can process.

#### `model.vocab_size() → number`

Vocabulary size.

#### `model.architecture() → string`

Architecture string from GGUF metadata (e.g. `"llama"`).

---

## Demo Application

The `demo/` directory contains a complete drag-and-drop browser demo:

- **`demo/index.html`** — UI with model file drop zone, prompt input, and streaming output display
- **`demo/worker.js`** — Web Worker that runs inference off the main thread

The Web Worker pattern is essential: inference blocks the thread it runs on. Running it in a worker keeps the UI responsive during generation.

### Worker message protocol

```js
// Main thread → Worker
{ type: 'load', bytes: Uint8Array }   // load model
{ type: 'generate', prompt: string, maxTokens: number, temperature: number }

// Worker → Main thread
{ type: 'loaded' }                    // model ready
{ type: 'token', text: string }       // one generated token
{ type: 'done' }                      // generation complete
{ type: 'error', message: string }    // error
```

---

## WASM-Specific Considerations

### No filesystem access

WASM runs in a sandbox without filesystem access. `GgufModel::from_bytes(bytes)` is used instead of `load(path)`. The GGUF file must be fetched as an `ArrayBuffer` from a URL or selected via `<input type="file">`.

### No rayon (single-threaded)

Web Workers run in a single thread — rayon's thread pool is disabled on `wasm32`. The SIMD kernels in `simd.rs` also don't compile for `wasm32`. Inference uses the scalar fallback path.

For multi-threaded WASM, [SharedArrayBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer) + WASM threads would be needed (requires cross-origin isolation headers).

### Memory

Small models (≤135M parameters) work well in the browser. Larger models require significant heap memory. A Q8_0 SmolLM-135M uses ~140 MB; a Q4_K TinyLlama-1.1B uses ~700 MB. The browser enforces memory limits that vary by device.

---

## Hosting

WASM files can be served as static assets:

```html
<!-- In your HTML -->
<script type="module">
import init, { GlintModel } from './pkg/glint.js';

await init();
const model = new GlintModel(ggufBytes);
const output = model.generate("Hello", 50, 0.7);
document.getElementById('output').textContent = output;
</script>
```

No server-side processing — all inference runs in the browser.
