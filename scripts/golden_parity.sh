#!/usr/bin/env bash
# Golden-output parity check: Glint vs llama.cpp, greedy decode.
#
# Runs the same prompt through both engines with temperature 0 and compares
# the generated text exactly. Greedy decode is deterministic, so any
# divergence means the two engines disagree somewhere in the stack —
# tokenizer, BOS handling, quantized kernels, RoPE/GQA attention, or
# sampling. (This harness caught a real bug on its first run: Glint used to
# prepend BOS unconditionally instead of honouring
# `tokenizer.ggml.add_bos_token`.)
#
# Usage:
#   scripts/golden_parity.sh MODEL.gguf [N_TOKENS] [PROMPT]
#
# Environment:
#   GLINT_BIN  path to the glint binary       (default: target/release/glint)
#   LLAMA_BIN  path to llama.cpp's completion binary
#              (default: llama-completion; older releases: llama-cli)
#
# Exit codes: 0 = parity, 1 = divergence, 2 = usage/setup error.
set -euo pipefail

MODEL=${1:?usage: golden_parity.sh MODEL.gguf [N_TOKENS] [PROMPT]}
N_TOKENS=${2:-48}
PROMPT=${3:-"The old lighthouse keeper counted the ships as they passed:"}

GLINT_BIN=${GLINT_BIN:-target/release/glint}
LLAMA_BIN=${LLAMA_BIN:-llama-completion}
command -v "$LLAMA_BIN" >/dev/null 2>&1 || LLAMA_BIN=llama-cli

[ -f "$MODEL" ] || { echo "error: model not found: $MODEL" >&2; exit 2; }
[ -x "$GLINT_BIN" ] || { echo "error: glint binary not found: $GLINT_BIN (cargo build --release?)" >&2; exit 2; }
command -v "$LLAMA_BIN" >/dev/null || { echo "error: llama-completion / llama-cli not on PATH" >&2; exit 2; }

# llama.cpp: raw completion — conversation mode OFF (-no-cnv; instruct models
# otherwise get the chat template applied), greedy, generation only.
# stdin is closed so it can never drop into interactive mode.
llama_out=$("$LLAMA_BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" \
    --temp 0 -no-cnv --no-display-prompt --simple-io --seed 0 \
    </dev/null 2>/dev/null)

# Glint: `run` is greedy by default (temperature 0). stdout carries
# "Prompt: ...", "Output: <generation>", then a "(N tokens in ...)" stats
# line; extract just the generation (which may span multiple lines).
glint_out=$("$GLINT_BIN" run -f "$MODEL" -p "$PROMPT" -m "$N_TOKENS" 2>/dev/null | awk '
    /^\(.* tokens in .*\)$/ { exit }
    found { print }
    !found && sub(/^Output: /, "") { found = 1; print }
')

# Trim leading/trailing whitespace only — interior divergence must fail.
trim() { local s=$1; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }
llama_trimmed=$(trim "$llama_out")
glint_trimmed=$(trim "$glint_out")

model_name=$(basename "$MODEL")
if [ "$llama_trimmed" = "$glint_trimmed" ] && [ -n "$glint_trimmed" ]; then
    echo "PARITY  $model_name  (${N_TOKENS} tokens, greedy)"
    echo "  output: $(printf '%s' "$glint_trimmed" | tr '\n' ' ' | head -c 100)..."
    exit 0
fi

echo "DIVERGENCE  $model_name" >&2
echo "--- llama.cpp:" >&2
printf '%s\n' "$llama_trimmed" >&2
echo "--- glint:" >&2
printf '%s\n' "$glint_trimmed" >&2
exit 1
