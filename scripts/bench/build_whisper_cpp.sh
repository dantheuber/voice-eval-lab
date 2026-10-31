#!/usr/bin/env bash
# Builds whisper.cpp with Vulkan into .runtime/whisper.cpp using a conda-forge toolchain in .runtime/toolchain.
# No system packages. Toolchain install:
#   micromamba create -p .runtime/toolchain -c conda-forge gcc=14 gxx=14 cmake ninja shaderc glslang \
#     libvulkan-headers libvulkan-loader spirv-headers sysroot_linux-64
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RT="$ROOT/.runtime"
TC="$RT/toolchain"
SRC="$RT/whisper.cpp"
TAG="${WHISPER_TAG:-v1.9.5}"
export PATH="$TC/bin:$PATH"
export CC="$TC/bin/x86_64-conda-linux-gnu-gcc" CXX="$TC/bin/x86_64-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$TC/lib:${LD_LIBRARY_PATH:-}"
[ -d "$SRC/.git" ] || git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/whisper.cpp.git "$SRC"
cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON \
  -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_PREFIX_PATH="$TC" -DCMAKE_BUILD_RPATH="$TC/lib" -DCMAKE_INSTALL_RPATH="$TC/lib"
cmake --build "$SRC/build" -j 16
ls -la "$SRC/build/bin"
