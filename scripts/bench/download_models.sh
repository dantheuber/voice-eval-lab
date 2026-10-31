#!/usr/bin/env bash
# Downloads benchmark models into .runtime/models. Safe to re-run; hf skips complete files.
# Usage: scripts/bench/download_models.sh [small|asr|tts|judge|all]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
M="$ROOT/.runtime/models"
mkdir -p "$M"
HF="${HF:-$HOME/.local/bin/hf}"

get() { # repo file...
  local repo="$1"; shift
  local dir="$M/${repo//\//__}"
  mkdir -p "$dir"
  "$HF" download "$repo" "$@" --local-dir "$dir" 2>&1 | tail -1
}

small() {
  get LiquidAI/LFM3-Audio-2B-GGUF LFM3-Audio-2B-Q4_K_M.gguf mmproj-LFM3-Audio-2B-Q4_K_M.gguf
  get LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q4_K_M.gguf
  get unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
  get ggml-org/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_0.gguf
  get nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf
}

# Not in the default set: over the 35 GB budget.
extra() {
  get unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf
  get ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_0.gguf
  get ggml-org/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf
}

asr() {
  get ggml-org/Qwen3-ASR-0.6B-GGUF Qwen3-ASR-0.6B-Q8_0.gguf mmproj-Qwen3-ASR-0.6B-Q8_0.gguf
  get nvidia/parakeet-tdt-0.6b-v3 parakeet-tdt-0.6b-v3.q8_0.gguf
  get nvidia/nemotron-speech-streaming-en-0.6b nemotron-speech-streaming-en-0.6b.q8_0.gguf
  mkdir -p "$M/whisper"
  [ -s "$M/whisper/ggml-large-v3-turbo-q5_0.bin" ] || curl -fsSL -o "$M/whisper/ggml-large-v3-turbo-q5_0.bin" \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
}

tts() {
  mkdir -p "$M/kokoro"
  [ -s "$M/kokoro/kokoro-v1.0.onnx" ] || curl -fsSL -o "$M/kokoro/kokoro-v1.0.onnx" \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
  [ -s "$M/kokoro/voices-v1.0.bin" ] || curl -fsSL -o "$M/kokoro/voices-v1.0.bin" \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
  get ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
}

judge() {
  get unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf
}

case "${1:-all}" in
  small) small ;;
  asr) asr ;;
  tts) tts ;;
  judge) judge ;;
  extra) extra ;;
  all) small; asr; tts; judge ;;
  *) echo "unknown set: $1" >&2; exit 2 ;;
esac
du -sh "$M"
