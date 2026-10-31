#!/usr/bin/env bash
# Starts the model servers for a profile and waits for health. Usage: scripts/bench/launch_stack.sh local-fast|local-quality|judge|stop
# Ports: 18101 STT, 18102 LLM, 18103 LFM3-Audio, 18104 whisper eval, 18105 judge. Phonon and Pocket run in-process in the app.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RT="$ROOT/.runtime"
M="$RT/models"
L="$RT/llama-vulkan/llama-b12017"
N="$RT/nemo/nemo-speech-0.3.0-linux-x86_64-vulkan"
W="$RT/whisper.cpp/build/bin/whisper-server"
PIDS="$RT/stack.pids"
export LD_LIBRARY_PATH="$L:${LD_LIBRARY_PATH:-}"

llama() { # port log args...
  local port="$1" log="$2"; shift 2
  "$L/llama-server" --host 127.0.0.1 --port "$port" --no-webui "$@" > "$RT/logs/stack-$log.log" 2>&1 &
  echo $! >> "$PIDS"
}

parakeet() {
  LD_LIBRARY_PATH="$N/lib" "$N/bin/nemo-speech" serve --no-ui --port 18101 \
    --asr-model "$M/nvidia__parakeet/parakeet-tdt-0.6b-v3.q8_0.gguf" --device vulkan:0 > "$RT/logs/stack-stt.log" 2>&1 &
  echo $! >> "$PIDS"
}

qwen_asr() {
  llama 18101 stt -m "$M/ggml-org__Qwen3-ASR-0.6B-GGUF/Qwen3-ASR-0.6B-Q8_0.gguf" \
    -mm "$M/ggml-org__Qwen3-ASR-0.6B-GGUF/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf" --temp 0 -c 8192 -np 2 -ngl 999
}

support_llm_quality() {
  llama 18102 llm -m "$M/unsloth__Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf" --jinja --reasoning off -c 32768 -np 2 -ngl 999
}

support_llm_fast() {
  llama 18102 llm -m "$M/LiquidAI__LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf" --jinja --reasoning off -c 32768 -np 2 -ngl 999 \
    --temp 0.1 --top-k 50 --repeat-penalty 1.1
}

lfm_audio() {
  llama 18103 lfm -m "$M/LiquidAI__LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q4_K_M.gguf" \
    -mm "$M/LiquidAI__LFM3-Audio-2B-GGUF/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" --temp 0 -c 16384 -np 2 -ngl 999
}

whisper_eval() {
  LD_LIBRARY_PATH="$RT/toolchain/lib" "$W" -m "$M/whisper/ggml-large-v3-q5_0.bin" --host 127.0.0.1 --port 18104 -l en -t 8 > "$RT/logs/stack-whisper.log" 2>&1 &
  echo $! >> "$PIDS"
}

judge() {
  llama 18105 judge -m "$M/unsloth__Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" --jinja --reasoning off --spec-type draft-mtp -c 32768 -np 1 -ngl 999 --temp 0
}

wait_health() {
  for port in "$@"; do
    for _ in $(seq 1 600); do
      curl -fs -m 1 "http://127.0.0.1:$port/health" > /dev/null && break
      sleep 0.2
    done
    curl -fs -m 1 "http://127.0.0.1:$port/health" > /dev/null && echo "port $port healthy" || echo "port $port NOT healthy"
  done
}

case "${1:-}" in
  local-fast) parakeet; support_llm_fast; wait_health 18101 18102 ;;
  local-quality) qwen_asr; support_llm_quality; lfm_audio; wait_health 18101 18102 18103 ;;
  judge) whisper_eval; judge; wait_health 18104 18105 ;;
  stop) [ -f "$PIDS" ] && xargs -r kill < "$PIDS"; rm -f "$PIDS" ;;
  *) echo "usage: $0 local-fast|local-quality|judge|stop" >&2; exit 2 ;;
esac
