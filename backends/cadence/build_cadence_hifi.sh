#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

BUNDLE_IO=false
BUILD_TESTS=false
SKIP_PIP=false

for arg in "$@"; do
  case "$arg" in
    -b|--bundle-io)  BUNDLE_IO=true ;;
    -t|--tests)      BUILD_TESTS=true ;;
    --skip-pip)      SKIP_PIP=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

if [ -z "${XTENSA_CORE:-}" ]; then
    echo "Error: XTENSA_CORE environment variable is not set."
    exit 1
fi

HIFI_CORE=$(echo "$XTENSA_CORE" | tr '[:upper:]' '[:lower:]' | grep -Eo 'hifi[0-9]+' || true)
if [ -z "$HIFI_CORE" ]; then
    echo "Error: Could not extract a valid HiFi version from XTENSA_CORE ($XTENSA_CORE)"
    exit 1
fi
echo "Detected HiFi core: $HIFI_CORE"

unset CMAKE_PREFIX_PATH
git submodule sync
git submodule update --init --recursive
./backends/cadence/cadence_apply_patch.sh

rm -rf cmake-out

if ! $SKIP_PIP; then
    export CMAKE_ARGS="\
-DEXECUTORCH_BUILD_XNNPACK=OFF \
-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF \
-DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
-DEXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF \
-DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF \
-DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF \
-DEXECUTORCH_BUILD_COREML=OFF \
-DEXECUTORCH_BUILD_VULKAN=OFF \
-DEXECUTORCH_BUILD_OPENVINO=OFF \
-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
-DEXECUTORCH_BUILD_PYBIND=OFF \
-DEXECUTORCH_BUILD_CMSIS_NN_PYBINDS=OFF"

    ./install_executorch.sh
    unset CMAKE_ARGS
fi
cmake_prefix_path="${PWD}/cmake-out/lib/cmake/ExecuTorch"

extra_flags=()

if $BUNDLE_IO; then
    extra_flags+=(
        -DEXECUTORCH_BUILD_CADENCE_BUNDLE_IO=ON
        -DEXECUTORCH_BUILD_DEVTOOLS=ON
    )
fi

if $BUILD_TESTS; then
    extra_flags+=(-DEXECUTORCH_BUILD_CADENCE_OP_TESTS=ON)
fi

./backends/cadence/install_requirements.sh
CXXFLAGS="-fno-exceptions -fno-rtti" cmake \
    -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
    -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
    -DEXECUTORCH_BUILD_CPUINFO=OFF \
    -DEXECUTORCH_BUILD_CADENCE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_HIFI_CORE="$HIFI_CORE" \
    -DEXECUTORCH_USE_DL=OFF \
    -DHAVE_FNMATCH_H=OFF \
    -DFLATCC_ALLOW_WERROR=OFF \
    "${extra_flags[@]}" \
    -Bcmake-out

cmake --build cmake-out --target install --config Release -j8

python3 -m examples.portable.scripts.export --model_name="add"
xt-run --turbo cmake-out/backends/cadence/cadence_executor_runner_sim --model_path=add.pte
