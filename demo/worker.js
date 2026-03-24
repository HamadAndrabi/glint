/**
 * Glint Web Worker
 *
 * Runs GGUF model loading and inference off the main thread.
 * Communicates via postMessage:
 *
 *   Main → Worker:
 *     { type: 'load',     bytes: Uint8Array }
 *     { type: 'generate', prompt: string, maxTokens: number, temperature: number }
 *
 *   Worker → Main:
 *     { type: 'loaded',   architecture: string, contextLength: number, vocabSize: number }
 *     { type: 'token',    text: string }
 *     { type: 'done' }
 *     { type: 'error',    message: string }
 *     { type: 'status',   message: string }
 */

import init, { GlintModel, init_panic_hook } from '../pkg/glint.js';

let model = null;

// Initialise the WASM module once on worker startup.
const ready = init().then(() => {
    init_panic_hook();
});

self.onmessage = async function (e) {
    const { type, ...data } = e.data;

    if (type === 'load') {
        try {
            await ready;
            postMessage({ type: 'status', message: 'Parsing model…' });
            model = new GlintModel(data.bytes);
            postMessage({
                type: 'loaded',
                architecture:  model.architecture(),
                contextLength: model.context_length(),
                vocabSize:     model.vocab_size(),
            });
        } catch (err) {
            postMessage({ type: 'error', message: String(err) });
        }
        return;
    }

    if (type === 'generate') {
        if (!model) {
            postMessage({ type: 'error', message: 'No model loaded.' });
            return;
        }
        try {
            model.generate_streaming(
                data.prompt,
                data.maxTokens,
                data.temperature,
                (text) => postMessage({ type: 'token', text }),
            );
            postMessage({ type: 'done' });
        } catch (err) {
            postMessage({ type: 'error', message: String(err) });
        }
        return;
    }
};
