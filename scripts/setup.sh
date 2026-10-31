#!/usr/bin/env bash
# Recreates what a local model profile needs under .runtime/, from a clean checkout, with no sudo.
# Run it again at any time: anything already present is skipped. It never deletes anything.
#
#   scripts/setup.sh                     set up local-default
#   scripts/setup.sh --profile all       set up local-default, local-fast, and local-realtime
#   scripts/setup.sh --check             print what is present and missing, change nothing
#   scripts/setup.sh --yes               do not ask before a large download
#
# This covers the model servers. ./run.sh handles the Python app (uv sync) and the dashboard build.
#
# Linux x86-64 only:
#   - the llama.cpp and NeMo-Speech.cpp archives, the conda toolchain, and every Python wheel.
# Vulkan only:
#   - the llama.cpp, NeMo-Speech.cpp, and whisper.cpp builds. There is no CUDA or ROCm path here.
#     Tested on one machine: Radeon 8060S, RADV from Mesa 25.2.8, Ubuntu 25.10, glibc 2.42.
#     Vulkan itself must come from the system. Check it with scripts/doctor.py.
# Built on this machine, not downloaded:
#   - whisper-server, because whisper.cpp has no Linux prebuilt. GGML_NATIVE=ON tunes it to this
#     CPU, so do not copy the binary to a different CPU. Its RPATH holds this checkout's path.
#   - No system compiler is used. The toolchain is conda-forge, pinned package by package.
#
# Pins live in scripts/setup/: manifest.tsv for models, toolchain-explicit.txt for the compiler, and
# bench-venv-freeze.txt for Python. The venv keeps the name "bench" because the profiles point at
# it, and it holds only what Phonon and Pocket TTS need unless you install the whole freeze file.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$ROOT/.runtime"
MANIFEST="$ROOT/scripts/setup/manifest.tsv"
CONDA_LOCK="$ROOT/scripts/setup/toolchain-explicit.txt"
FREEZE="$ROOT/scripts/setup/bench-venv-freeze.txt"
VENV="$RT/venvs/bench"
ALL_PROFILES="local-default local-fast local-realtime"
ASK_ABOVE_BYTES=$((1024 * 1024 * 1024))

LLAMA_DIR="$RT/llama-vulkan/llama-b12017"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/b12017/llama-b12017-bin-ubuntu-vulkan-x64.tar.gz"
LLAMA_SHA="e940a901831c2e157b6329b8738f6edd6b036889c07e4e39d4fc818831220f61"
LLAMA_BYTES=35665113

NEMO_DIR="$RT/nemo/nemo-speech-0.3.0-linux-x86_64-vulkan"
NEMO_URL="https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.3.0/nemo-speech-0.3.0-linux-x86_64-vulkan.tar.gz"
NEMO_SHA="2a8b2ca4610b6ea5d659ba10794526f6e5a77fb1623d52c35272a9fd1f74abac"
NEMO_BYTES=19482397

MAMBA_URL="https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2"
MAMBA_SHA="9a97976244929bbae28fb1fae504e74a3e7e6d1932870e3a03321f01b7fbd07b"
MAMBA_BYTES=7317886
TOOLCHAIN_BYTES=239250312 # sum of the conda package sizes in toolchain-explicit.txt
WHISPER_SOURCE_BYTES=9000000 # shallow clone, approximate
WHISPER_BIN="$RT/whisper.cpp/build/bin/whisper-server"
WHISPER_DISK_BYTES=3900000000 # toolchain, package cache, source, and build, measured here

VENV_BYTES=3074672747 # sum of the 69 wheels on PyPI
VENV_DISK_BYTES=5400000000 # measured here

PHONON_CACHE="$RT/hf/hub/models--gradium--phonon-31m-en"
PHONON_REV="92741c5b07d085ff840e26fea8ce43110b87852f"

die() {
  echo "setup: $*" >&2
  exit 1
}

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo "Options: --profile local-default|local-fast|local-realtime|all, --check, --yes, --quiet"
}

PROFILE=local-default
CHECK_ONLY=0
ASSUME_YES=0
QUIET=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile)
      [ $# -ge 2 ] || die "--profile needs a value"
      PROFILE="$2"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --check) CHECK_ONLY=1 && shift ;;
    --yes | -y) ASSUME_YES=1 && shift ;;
    --quiet) QUIET=1 && shift ;; # with --check: print only what is missing
    -h | --help) usage && exit 0 ;;
    *) usage >&2 && die "unknown option: $1" ;;
  esac
done

case " $ALL_PROFILES all " in
  *" $PROFILE "*) ;;
  *) die "unknown profile '$PROFILE', use local-default, local-fast, local-realtime, or all" ;;
esac

wanted() { # comma list of profiles
  [ "$PROFILE" = all ] || [[ ",$1," == *",$PROFILE,"* ]]
}

human() {
  awk -v b="$1" 'BEGIN {
    if (b >= 1e9) printf "%.1f GB", b / 1e9
    else if (b >= 1e6) printf "%.1f MB", b / 1e6
    else printf "%.0f KB", b / 1e3
  }'
}

line() {
  [ "$QUIET" = 1 ] && [ "$1" = present ] && return 0
  printf '  %-8s %-46s %s\n' "$1" "$2" "${3:-}"
}

# Status ------------------------------------------------------------------------------------------

file_ok() { # path-from-repo-root expected-bytes
  [ -e "$ROOT/$1" ] && [ "$(stat -Lc %s "$ROOT/$1")" = "$2" ]
}

venv_ok() {
  [ -x "$VENV/bin/python" ] && [ -x "$VENV/bin/pocket-tts" ] && [ -x "$VENV/bin/hf" ] &&
    compgen -G "$VENV/lib/python3.12/site-packages/ptts-0.6.0.dist-info" > /dev/null &&
    compgen -G "$VENV/lib/python3.12/site-packages/pocket_tts-4.0.1.dist-info" > /dev/null
}

llama_ok() { [ -x "$LLAMA_DIR/llama-server" ]; }
nemo_ok() { [ -x "$NEMO_DIR/bin/nemo-speech" ]; }
whisper_ok() { [ -x "$WHISPER_BIN" ]; }
phonon_ref_ok() { [ "$(cat "$PHONON_CACHE/refs/main" 2> /dev/null)" = "$PHONON_REV" ]; }
nemo_wanted() { wanted local-fast; }

MISSING_REQUIRED=0
MISSING_ON_DEMAND=0
DOWNLOAD_BYTES=0
DISK_BYTES=0

piece() { # ok-function label path download-bytes disk-bytes
  if "$1"; then
    line present "$2" "${3#"$ROOT"/}"
  else
    line missing "$2" "$(human "$4") to download"
    MISSING_REQUIRED=$((MISSING_REQUIRED + 1))
    DOWNLOAD_BYTES=$((DOWNLOAD_BYTES + $4))
    DISK_BYTES=$((DISK_BYTES + $5))
  fi
}

group_line() { # label have total missing-bytes required? names...
  local label="$1" have="$2" total="$3" bytes="$4" required="$5"
  shift 5
  if [ "$have" = "$total" ]; then
    line present "$label, $total of $total"
    return
  fi
  line missing "$label, $have of $total" "$(human "$bytes") to download: $*"
  if [ "$required" = 1 ]; then
    MISSING_REQUIRED=$((MISSING_REQUIRED + total - have))
  else
    MISSING_ON_DEMAND=$((MISSING_ON_DEMAND + total - have))
  fi
  DOWNLOAD_BYTES=$((DOWNLOAD_BYTES + bytes))
  DISK_BYTES=$((DISK_BYTES + bytes))
}

models_status() {
  local phonon_have=0 phonon_total=0 phonon_bytes=0 pocket_have=0 pocket_total=0 pocket_bytes=0
  local -a pocket_names=()
  local profiles need path bytes repo _rev file _hash name
  while IFS=$'\t' read -r profiles need path bytes repo _rev file _hash; do
    [[ "$profiles" == \#* ]] && continue
    wanted "$profiles" || continue
    name="$(basename "$path")"
    if [[ "$path" == */voices/*.safetensors ]]; then
      phonon_total=$((phonon_total + 1))
      if file_ok "$path" "$bytes"; then
        phonon_have=$((phonon_have + 1))
      else
        phonon_bytes=$((phonon_bytes + bytes))
      fi
    elif [[ "$path" == */embeddings/*.safetensors ]] && [ "$need" = on-demand ]; then
      pocket_total=$((pocket_total + 1))
      if file_ok "$path" "$bytes"; then
        pocket_have=$((pocket_have + 1))
      else
        pocket_bytes=$((pocket_bytes + bytes))
        pocket_names+=("${name%.safetensors}")
      fi
    elif file_ok "$path" "$bytes"; then
      line present "$repo" "$file"
    else
      line missing "$repo" "$file, $(human "$bytes") to download"
      MISSING_REQUIRED=$((MISSING_REQUIRED + 1))
      DOWNLOAD_BYTES=$((DOWNLOAD_BYTES + bytes))
      DISK_BYTES=$((DISK_BYTES + bytes))
    fi
  done < "$MANIFEST"
  [ "$phonon_total" -eq 0 ] || group_line "Phonon voices" "$phonon_have" "$phonon_total" \
    "$phonon_bytes" 1 "the voices are named in local_speech.py"
  [ "$pocket_total" -eq 0 ] || group_line "Pocket voice presets" "$pocket_have" "$pocket_total" \
    "$pocket_bytes" 0 "${pocket_names[@]+"${pocket_names[@]}"}"
}

status() {
  piece llama_ok "llama.cpp b12017, Vulkan build" "$LLAMA_DIR" "$LLAMA_BYTES" 100000000
  nemo_wanted && piece nemo_ok "NeMo-Speech.cpp 0.3.0, Vulkan build" "$NEMO_DIR" "$NEMO_BYTES" 60000000
  piece whisper_ok "whisper-server v1.9.5, built here" "$WHISPER_BIN" \
    $((MAMBA_BYTES + TOOLCHAIN_BYTES + WHISPER_SOURCE_BYTES)) "$WHISPER_DISK_BYTES"
  piece venv_ok "Python venv for Phonon and Pocket TTS" "$VENV" "$VENV_BYTES" "$VENV_DISK_BYTES"
  models_status
  if ! phonon_ref_ok; then
    line missing "Phonon offline ref" "refs/main, written after the download"
    MISSING_REQUIRED=$((MISSING_REQUIRED + 1))
  fi
}

# Steps -------------------------------------------------------------------------------------------

need_tools() {
  local tool
  for tool in curl git tar sha256sum df; do
    command -v "$tool" > /dev/null || die "$tool is missing, and setup can't install it without sudo"
  done
  command -v uv > /dev/null ||
    die "uv is missing. It installs without sudo: curl -LsSf https://astral.sh/uv/install.sh | sh"
}

fetch() { # url destination sha256
  local url="$1" dest="$2" sha="$3"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "Using the copy in ${dest#"$ROOT"/}"
    return
  fi
  echo "Downloading ${url##*/}"
  curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$dest.part" "$url"
  if [ "$(sha256sum "$dest.part" | cut -d' ' -f1)" != "$sha" ]; then
    rm -f "$dest.part"
    die "checksum mismatch for $url"
  fi
  mv "$dest.part" "$dest"
}

unpack() { # archive parent-directory
  local work="$2.unpacking"
  mkdir -p "$work"
  tar xzf "$1" -C "$work"
  mv "$work"/* "$2"/
  rmdir "$work"
}

install_venv() {
  echo "Creating the Python venv for Phonon and Pocket TTS"
  uv venv --allow-existing --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" -c "$FREEZE" ptts pocket-tts
}

install_llama() {
  fetch "$LLAMA_URL" "$RT/dl/llama-vulkan.tgz" "$LLAMA_SHA"
  mkdir -p "$RT/llama-vulkan"
  unpack "$RT/dl/llama-vulkan.tgz" "$RT/llama-vulkan"
}

install_nemo() {
  fetch "$NEMO_URL" "$RT/dl/nemo/nemo.tgz" "$NEMO_SHA"
  mkdir -p "$RT/nemo"
  unpack "$RT/dl/nemo/nemo.tgz" "$RT/nemo"
}

install_toolchain() {
  [ -x "$RT/toolchain/bin/x86_64-conda-linux-gnu-gcc" ] && return
  local mamba="$RT/dl/micromamba/bin/micromamba"
  if [ ! -x "$mamba" ]; then
    fetch "$MAMBA_URL" "$RT/dl/micromamba/mm.tar.bz2" "$MAMBA_SHA"
    # Python, not tar: this machine has no bzip2.
    "$VENV/bin/python" -c '
import sys, tarfile
with tarfile.open(sys.argv[1], "r:bz2") as archive:
    archive.extract("bin/micromamba", sys.argv[2], filter="data")
' "$RT/dl/micromamba/mm.tar.bz2" "$RT/dl/micromamba"
    chmod +x "$mamba"
  fi
  echo "Installing the conda-forge toolchain into .runtime/toolchain"
  MAMBA_ROOT_PREFIX="$RT/mamba" "$mamba" create -y -q -p "$RT/toolchain" -f "$CONDA_LOCK"
}

install_whisper() {
  install_toolchain
  echo "Building whisper-server. This takes a few minutes on all cores."
  bash "$ROOT/scripts/bench/build_whisper_cpp.sh"
}

download_models() {
  local -A groups=()
  local profiles _need path bytes repo rev file _hash dir key
  while IFS=$'\t' read -r profiles _need path bytes repo rev file _hash; do
    [[ "$profiles" == \#* ]] && continue
    wanted "$profiles" || continue
    file_ok "$path" "$bytes" && continue
    dir=""
    [[ "$path" == .runtime/models/* ]] && dir="${path%/*}"
    groups["$repo|$rev|$dir"]+="$file"$'\n'
  done < "$MANIFEST"

  local -a files arguments
  for key in "${!groups[@]}"; do
    IFS='|' read -r repo rev dir <<< "$key"
    mapfile -t files <<< "${groups[$key]%$'\n'}"
    echo "Downloading ${#files[@]} file(s) from $repo"
    arguments=(download "$repo" "${files[@]}" --revision "$rev")
    [ -z "$dir" ] || arguments+=(--local-dir "$ROOT/$dir")
    HF_HOME="$RT/hf" HF_HUB_DISABLE_TELEMETRY=1 "$VENV/bin/hf" "${arguments[@]}" > /dev/null
  done

  # phonon_server.py runs offline and names no revision, so the cache needs this pointer.
  if [ -d "$PHONON_CACHE/snapshots/$PHONON_REV" ]; then
    mkdir -p "$PHONON_CACHE/refs"
    printf %s "$PHONON_REV" > "$PHONON_CACHE/refs/main"
  fi
}

confirm() {
  local available
  available="$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d ' ')"
  if [ "$DISK_BYTES" -gt "$available" ]; then
    die "this needs about $(human "$DISK_BYTES") of disk and $(human "$available") is free"
  fi
  [ "$DOWNLOAD_BYTES" -ge "$ASK_ABOVE_BYTES" ] || return 0
  [ "$ASSUME_YES" = 0 ] || return 0
  if [ ! -t 0 ]; then
    die "this downloads $(human "$DOWNLOAD_BYTES"). Run again with --yes to allow it"
  fi
  local answer
  read -r -p "Download $(human "$DOWNLOAD_BYTES") and use about $(human "$DISK_BYTES") of disk? [y/N] " answer
  [[ "$answer" == [yY]* ]] || die "stopped, nothing downloaded"
}

# Main --------------------------------------------------------------------------------------------

[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] ||
  die "this needs Linux x86-64, because the archives and wheels it installs are built for that"

[ "$QUIET" = 1 ] || echo "Profile $PROFILE, Linux x86-64 and Vulkan builds only."
status

if [ "$CHECK_ONLY" = 1 ]; then
  if [ "$QUIET" = 0 ]; then
    if [ "$MISSING_REQUIRED" -eq 0 ] && [ "$MISSING_ON_DEMAND" -eq 0 ]; then
      echo "Everything is present."
    else
      echo "Missing: $(human "$DOWNLOAD_BYTES") to download, about $(human "$DISK_BYTES") of disk."
      echo "Run: scripts/setup.sh --profile $PROFILE"
    fi
  fi
  [ "$MISSING_REQUIRED" -eq 0 ]
  exit
fi

if [ "$MISSING_REQUIRED" -eq 0 ] && [ "$MISSING_ON_DEMAND" -eq 0 ]; then
  echo "Nothing to do: everything for $PROFILE is already here."
  exit 0
fi

need_tools
echo "To fetch: $(human "$DOWNLOAD_BYTES"), about $(human "$DISK_BYTES") of disk."
confirm
mkdir -p "$RT"

venv_ok || install_venv
llama_ok || install_llama
if nemo_wanted && ! nemo_ok; then install_nemo; fi
whisper_ok || install_whisper
download_models

echo
MISSING_REQUIRED=0
MISSING_ON_DEMAND=0
DOWNLOAD_BYTES=0
DISK_BYTES=0
QUIET=1
status
if [ "$MISSING_REQUIRED" -ne 0 ] || [ "$MISSING_ON_DEMAND" -ne 0 ]; then
  die "something above is still missing"
fi
echo "Done. Check the machine with scripts/doctor.py, then start with ./run.sh."
