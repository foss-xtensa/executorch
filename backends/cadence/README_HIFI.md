# Building ExecuTorch for the Cadence HiFi DSP

This README helps you build the Cadence HiFi backend with one stage at a time. It explains each command, the ExecuTorch installation options, and Cadence-specific CMake flags. Use these stages to customize the build flow, troubleshoot issues, or integrate the build.
## Table of Contents

*   [1. Prerequisites](#1-prerequisites)
*   [2. Python Virtual Environment Setup](#2-python-virtual-environment-setup)
*   [3. Environment Configuration](#3-environment-configuration)
*   [4. Build Stages](#4-build-stages)
    *   [Stage 1 - Select the HiFi Core](#stage-1---select-the-hifi-core)
    *   [Stage 2 - Prepare the Source Tree](#stage-2---prepare-the-source-tree)
    *   [Stage 3 - Install the Host ExecuTorch Package](#stage-3---install-the-host-executorch-package)
    *   [Stage 4 - Fetch Cadence NN Libraries and FACTO](#stage-4---fetch-cadence-nn-libraries-and-facto)
    *   [Stage 5 - Configure the Cross-Compile for the Xtensa Target](#stage-5---configure-the-cross-compile-for-the-xtensa-target)
    *   [Stage 6 - Build and Install Runtime Components](#stage-6---build-and-install-runtime-components)
    *   [Stage 7 - Run Model On the Xtensa ISS](#stage-7---run-model-on-the-xtensa-iss)
*   [5. Exporting Models to `.pte`](#5-exporting-models-to-pte)
    *   [5.1 Example Models](#51-example-models)
    *   [5.2 `export_and_run_model` parameters](#52-export_and_run_model-parameters)
    *   [5.3 Locating the Exported File](#53-locating-the-exported-file)
*   [6. Verification of Models](#6-verification-of-models)
*   [7. Additional Information](#7-additional-information)
    *   [7.1 Runner and Xtensa ISS Details](#71-runner-and-xtensa-iss-details)
    *   [7.2 The ExecuTorch Install Subset](#72-the-executorch-install-subset)
    *   [7.3 Cadence-internal CMake Flags](#73-cadence-internal-cmake-flags)
    *   [7.4 Optional Build Features](#74-optional-build-features)
*   [8. Troubleshooting](#8-troubleshooting)
*   [9. Reference Script](#9-reference-script)

---

## 1. Prerequisites

Ensure that you install the following tools on the host system before you begin the build process.

*   **Python:** **3.12-3.13**, in an active virtual environment
*   **Host Compiler (C++17):** `GCC` **12** or `Clang` **5.0** and later.
*   **System Assembler (`binutils`):** **2.40** and later (for AVX-512/BF16 host kernels)
*   **Cadence Xtensa toolchain** binaries accessible on your system `PATH`
    (`xt-clang`, `xt-ld`, `xt-run`, and so on)

---

## 2. Python Virtual Environment Setup

Use a clean environment to isolate dependencies and avoid library conflicts. Run the following commands from the `executorch` root directory.

**Bash / Zsh:**

```bash
# 1. Create the virtual environment (use an explicit 3.12+ interpreter)
python3.12 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Verify it is active (should point to executorch/.venv/bin/python)
which python

# 4. Confirm the interpreter is >= 3.12
python --version
```

**Csh / Tcsh:**

```csh
# 1. Create the virtual environment
python3.12 -m venv .venv

# 2. Activate it
source .venv/bin/activate.csh

# 3. Verify it is active
which python

# 4. Confirm the interpreter is >= 3.12
python --version
```

---

## 3. Environment Configuration

Export the variables that point to your local Xtensa installation and target core, and add the toolchain binaries (`xt-clang`, `xt-ld`, `xt-run`, and so on) on your system `PATH`. The Xtensa toolchain **requires** `XTENSA_CORE` environment variable. Select the HiFi core family separately in [Stage 1](#stage-1---select-the-hifi-core) (For more information, refer to
[Cadence flags](#73-cadence-internal-cmake-flags)).

**Bash / Zsh:**

```bash
# Path to the Xtensa tools install dir (For example, /opt/Xtensa/XtDevTools/install/tools)
export XTENSA_TOOLCHAIN=/path/to/XtDevTools/install/tools

# Toolchain version release string (For example, RI-2023.11-linux)
export TOOLCHAIN_VER=your_release_version

# Xtensa configuration registry directory
export XTENSA_SYSTEM=${XTENSA_TOOLCHAIN}/${TOOLCHAIN_VER}/XtensaTools/config/

# Target core configuration profile (For example,  AE_HiFi5s_LE5_AO_FP_XC)
export XTENSA_CORE=your_hifi_core_name

# Put the toolchain binaries (xt-clang, xt-ld, xt-run, ...) on PATH
export PATH=${XTENSA_TOOLCHAIN}/${TOOLCHAIN_VER}/XtensaTools/bin:$PATH
```

**Csh / Tcsh:**

```csh
setenv XTENSA_TOOLCHAIN /path/to/XtDevTools/install/tools
setenv TOOLCHAIN_VER your_release_version
setenv XTENSA_SYSTEM ${XTENSA_TOOLCHAIN}/${TOOLCHAIN_VER}/XtensaTools/config/
setenv XTENSA_CORE your_hifi_core_name
setenv PATH ${XTENSA_TOOLCHAIN}/${TOOLCHAIN_VER}/XtensaTools/bin:${PATH}
```

---

## 4. Build Stages

Run the following commands from the `executorch` root with your virtual environment activated (see [Section 2](#2-python-virtual-environment-setup)) and the `XTENSA_*` environment variables set (see [Section 3](#3-environment-configuration)). The examples use Bash/Zsh syntax. If you use Csh/Tcsh, adjust the variable assignments as needed.

### Stage 1 - Select the HiFi Core

Specify the target HiFi version. Valid values are `hifi1`, `hifi4`, or `hifi5`. This value is passed to CMake as `-DEXECUTORCH_HIFI_CORE`. If you leave the value unset or specify an unsupported value, the build uses the portable 'generic' kernels that run on any Xtensa core without the HiFi NN libraries.

```bash
export HIFI_CORE=hifi5   # or hifi1 / hifi4; empty for the generic (portable) kernels
```

### Stage 2 - Prepare the Source Tree

Clear any existing prefix path, synchronize and update the submodules, and apply the Cadence gflags patch. The patch operation detects whether the patch has already been applied.
> NOTE: Make sure you are in `rel_hifi` branch before you start.

```bash
unset CMAKE_PREFIX_PATH

git submodule sync
git submodule update --init --recursive

# Applies backends/cadence/cadence_patch.patch to third-party/gflags
./backends/cadence/cadence_apply_patch.sh
```

### Stage 3 - Install the Host ExecuTorch Package

This step builds the Python package used to export models and generates the host-side ExecuTorch CMake configuration files (`cmake-out/lib/cmake/ExecuTorch`) required for cross-compilation. Instead of performing a full installation, use the curated `CMAKE_ARGS` configuration shown below. This configuration clears the host backends and kernels that are not used in the DSP workflow. For an explanation of each option and a comparison with the `--minimal` configuration, see [Section 7.2](#72-the-executorch-install-subset).

```bash
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
```

Run this stage only when Python dependencies or the host build change. Skip this stage for executorch runner recompiles. See Rebuilds note at the end of this section.

### Stage 4 - Fetch Cadence NN Libraries and FACTO

Clone the Cadence HiFi kernel libraries and install the FACTO operator-testing package in editable mode. The process skips repositories that have already been cloned, so it is safe to rerun this step.

```bash
./backends/cadence/install_requirements.sh
```

This populates:

*   `nnlib-hifi4`  -> `hifi/third-party/nnlib/nnlib-hifi4`
*   `nnlib-hifi5`  -> `hifi/third-party/nnlib/nnlib-hifi5`
*   `FACTO`        -> `pip install -e backends/cadence/utils/FACTO`

### Stage 5 - Configure the Cross-Compile for the Xtensa Target

> **NOTE - Host compiler.** Although the target is Xtensa, this stage still needs
> a working **host** GCC / binutils. The build compiles host-side tooling such as
> `flatcc`, which generates the FlatBuffers headers that `libexecutorch.a` is
> built against. **On an old GCC / binutils the build will fail** (missing
> AVX-512/BF16 assembler support, stale FlatBuffers header handling, etc.). We
> recommend **GCC 12** with **binutils 2.40+**; the versions verified for this
> guide are **GCC 12.4.0** and **binutils 2.40**.
>
> If the required GCC or binutils are not the system default (common on
> shared/managed hosts), set the installation path *before* you configure the build. Update the
> following root directory paths to match your environment.
>
> ```bash
> GCC_ROOT=/path/to/gcc/v12.4.0
> BINUTILS_ROOT=/path/to/binutils/v2.40
>
> export LD_LIBRARY_PATH="${GCC_ROOT}/lib64:${LD_LIBRARY_PATH}"   # GCC 12 runtime libs
> export PATH="${BINUTILS_ROOT}/bin:${GCC_ROOT}/bin:${PATH}"      # new binaries first
> export CC="${GCC_ROOT}/bin/gcc"
> export CXX="${GCC_ROOT}/bin/g++"
> export CFLAGS="-B${BINUTILS_ROOT}/bin/"                         # use new assembler/linker
> export CXXFLAGS="-B${BINUTILS_ROOT}/bin/"
> ```
>
> Before configuring the build, verify that you use the correct GCC and binutils versions. Run `which gcc g++ ld`, and then verify the versions by running `gcc --version` and `ld --version`. Use GCC 12.x and binutils 2.40 or later. The `CXXFLAGS` setting shown below extends this configuration and preserves the `-B` path redirection.

Configure CMake with Cadence cross-compilation tool chain. Options include:
*   `CXXFLAGS="-fno-exceptions -fno-rtti"` - The Xtensa runtime is built without C++ exceptions or RTTI.
*   `-DCMAKE_TOOLCHAIN_FILE=backends/cadence/cadence.cmake` - Selects the Xtensa cross toolchain (`xt-clang` / `xt-ld`) using your `XTENSA_*` environment variables.
*   `-DEXECUTORCH_BUILD_CADENCE=ON` / `-DEXECUTORCH_HIFI_CORE="$HIFI_CORE"` - The Cadence-internal switches described in [Section 7.3](#73-cadence-internal-cmake-flags).
*   `-DCMAKE_PREFIX_PATH=...` - Points at the host package built in [Stage 3](#stage-3---install-the-host-executorch-package).

```bash
CXXFLAGS="-fno-exceptions -fno-rtti ${CXXFLAGS:-}" cmake \
    -DCMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch" \
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
    -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=OFF \
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=OFF \
    -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_HIFI_CORE="$HIFI_CORE" \
    -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=OFF \
    -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=OFF \
    -DEXECUTORCH_USE_DL=OFF \
    -DHAVE_FNMATCH_H=OFF \
    -DFLATCC_ALLOW_WERROR=OFF \
    -Bcmake-out
```

To enable BundledIO or the output validation tests, add the extra flags described in [Section 7.4](#74-optional-build-features) to this command.

### Stage 6 - Build and Install Runtime Components

```bash
cmake --build cmake-out --target install --config Release -j8
```

### Stage 7 - Run a Model on the Xtensa ISS

Run a `.pte` on the instruction-set simulator with the runner built above:

```bash
xt-run --turbo cmake-out/backends/cadence/cadence_executor_runner_sim --model_path=add.pte
```

Generate `add.pte` simple example model using the portable AOT export workflow.

```bash
python3 -m examples.portable.scripts.export --model_name="add"
```

See [Additional Information](#7-additional-information) for the full runner CLI,
`xt-run` options, and the ExecuTorch install variants.

> **NOTE - Rebuilds.** After the first successful build, recompile C++ changes directly without repeating Stages 3-4:
>
> cmake --build cmake-out --target install --config Release -j8
>
> Repeat Stage 3 and 4 only when Python dependencies or the host ExecuTorch build changes.

---

## 5. Exporting Models to `.pte`

The Cadence AOT (Ahead of Time) flow quantizes a PyTorch `nn.Module`, lowers it through the Cadence-specific passes, and serializes it to a `.pte` (and a `.bpte` bundled program with reference IO). The entry points are located in [`backends/cadence/aot/export_example.py`](aot/export_example.py).

*   **`export_model(model, example_inputs, ...)`** - Runs the full export
    pipeline (prepare -> calibrate -> convert -> quantize -> lower) and writes the
    `.pte` / `.bpte`. Returns the `ExecutorchProgramManager`.
*   **`export_and_run_model(model, example_inputs, ...)`** - Calls `export_model`,
    and (only when `verify=True`) runs the exported `.pte` through a host GCC
    build of the runner to verify the program is not corrupted. This does **not**
    use `xt-run` or the Xtensa ISS as it is a host-side sanity check only.
    Setting `verify` also overwrites the default `cmake-out` build directory.
    Keep `verify=False` (the default) unless you explicitly need this check.

### 5.1 Example Models

Ready-to-run examples are located under [`examples/cadence/models/`](../../examples/cadence/models/) directory. Refer to these examples as templates for exporting your own model.

*   `babyllama.py` - A small Llama transformer
*   `mobilenet_v2.py`, `resnet50.py` - Torchvision CNNs
*   `wav2vec2.py` - Speech model
*   `rnnt_encoder.py`, `rnnt_predictor.py`, `rnnt_joiner.py` - RNN-T components

Each script builds a model with `example_inputs` and calls one of the export
entry points. Run one model with:

```bash
python3 -m examples.cadence.models.mobilenet_v2
```

### 5.2 `export_and_run_model` parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `model` | *(required)* | The `nn.Module` to export (set to `eval()` for inference models). |
| `example_inputs` | *(required)* | Tuple of example input tensors defining the input shapes/dtypes. |
| `file_name` | `"CadenceDemoModel"` | Base name for the output `.pte` / `.bpte`. |
| `verify` | `False` | When `True`, run the exported `.pte` through a host GCC build of the runner to verify the program is not corrupted (host-side check only, it does **not** use `xt-run` or the Xtensa ISS). Also overwrites the default `cmake-out` directory. **Recommended to keep off.** |
| `eps_error` / `eps_warn` | `1e-1` / `1e-5` | Error/warning thresholds used only when `verify=True`. |
| `force_rebuild` | `False` | Force the runtime rebuild during verification. |
| `working_dir` | `None` | Directory for output `.pte` / `.bpte` files. If `None`, a fresh temporary directory under `/tmp` is created automatically. |

`export_model` accepts the same export-related parameters (`file_name`,
`working_dir`) but does not run the model.

### 5.3 Locating the Exported File

Both export entry points log the output path when the program is written, for
example:

```
[INFO ...] Saved exported program to /tmp/tmpXXXXXX/CadenceDemoModel.pte
[INFO ...] Saved exported program to /tmp/tmpXXXXXX/CadenceDemoModel.bpte
```

If `working_dir` is not passed, the files are written to a fresh temporary
directory under `/tmp`. Copy the `.pte` (or `.bpte`) from the logged path to run
it on the ISS with the runner from Stage 7. Pass `working_dir=<path>` (or a
`file_name` ending in `.pte`/`.bpte`) to control where the output lands.

---

## 6. Verification of Models

The primary validation metric is on-device accuracy. The quantized model is exported as a BundledIO program (`.bpte`) that contains the x86 quantized PyTorch reference inputs and outputs. When you run the `.bpte` file on the DSP simulator, the simulator reports two error metrics:

*   Absolute error (mean and maximum)
*   Relative error (mean and maximum)

These metrics compare DSP execution results directly with the embedded reference outputs.

To use this validation method, set BundledIO during the build (see [Section 7.4](#74-optional-build-features)). This setting generates a `.bpte` file in addition to the `.pte` file. Run the model by using the following command:

```bash
xt-run --model_path=model.bpte
```

By default, the primary metric's validation thresholds for both absolute and relative error are set to `0.01`. These values were determined empirically to detect kernel implementation issues, such as fixed-point mismatches, while allowing natural INT8 rounding differences across platforms. Review these thresholds for your use cases and adjust them as needed by using the following options:

*   `--bundle_atol` for the absolute error threshold
*   `--bundle_rtol` for the relative error threshold

You can also validate DSP output against the output of the original unquantized PyTorch model. In this comparison, both quantization error and DSP optimization error contribute to the overall difference. Thus, the threshold is higher than the primary metric.

To perform this validation:

1.  Export the tensors from the PyTorch model inputs and outputs as float32 `.bin` files.
2.  Run the simulator with the same input data by using the `--inputs` and `--dump-output` options.
3.  Compare the simulator output with the PyTorch reference output on an element-by-element basis.

---

## 7. Additional Information

This section provides reference information for the build process, including runner and Xtensa ISS details, ExecuTorch installation options, Cadence internal CMake flags, and optional build features.

### 7.1 Runner and Xtensa ISS Details

The build produces `cadence_executor_runner_sim`
(`cmake-out/backends/cadence/cadence_executor_runner_sim`), a self-contained
runner compiled from
[`executor_runner_sim.cpp`](executor_runner_sim.cpp). It loads a
`.pte` (or `.bpte`), runs the first method, and prints/dumps IO. It has **no
gflags and no threadpool dependency** and uses a plain `argv` parser.

**Runner CLI flags** - The runner parses these flags independent of the launch method. Use the `--flag=value` for value options. Specify Boolean options without a value.

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model_path=<file>` | *(required)* | Path to the `.pte` (or `.bpte`) to run. |
| `--inputs=<a.bin,b.bin,...>` | *(none)* | Comma-separated raw `.bin` files, one per method input, in order. Each file's byte count must exactly match the tensor's `nbytes`. If omitted, inputs are filled with deterministic pseudo-random values (seeded with `0`, so runs are reproducible). Ignored for BundledIO. |
| `--print_input=<none\|summary>` | `summary` | Input print verbosity. `summary` shows the first/last few elements. |
| `--print_output=<none\|summary\|all>` | `summary` | Output print verbosity. `all` prints every element (matches the portable runner). |
| `--dump-input` | off | Write each input tensor's raw bytes to `<model>-in-<i>.bin`. |
| `--dump-output` | off | Write each output tensor's raw bytes to `<model>-out-<i>.bin`. |
| `--bundle_rtol=<float>` | `0.01` | *(BundledIO builds only)* Relative tolerance for output verification. |
| `--bundle_atol=<float>` | `0.01` | *(BundledIO builds only)* Absolute tolerance for output verification. |

Note:

*   **Random input fill** covers `float` (uniform in `[-1, 1]`), `int32`/`int64`
    (small non-negative values, safe for token IDs/indices), `int8`, and `bool`.
    Other dtypes are left unmodified - supply them with `--inputs` instead.
*   **BundledIO (`.bpte`)** is auto-detected at runtime if the runner was built
    with `-DEXECUTORCH_BUILD_CADENCE_BUNDLE_IO=ON` ([Section 7.4](#74-optional-build-features)). In that mode the
    runner ignores `--inputs`, loads the embedded reference inputs, runs, prints
    error stats (mean/max abs & relative error), and verifies outputs against the
    embedded references using `--bundle_rtol` / `--bundle_atol`. On mismatch, it
    logs `Test_result: FAIL` and exits non-zero (usable as a CI gate).
*   The runner reports execution cost as `Execute cycles = <n>` (from `times()`),
    and its memory pools are fixed at compile time (two 4 MB arenas and
    dynamically sized planned buffers).

#### Launching with `xt-run`
`xt-run` performs the cross-compiled ELF on the
instruction-set simulator and provides semi-hosting, so the runner's file I/O
(`--model_path`, `--inputs`, `--dump-*`) transparently reaches the host
filesystem. Everything after the ELF path is passed straight through to the
runner.

```bash
xt-run cmake-out/backends/cadence/cadence_executor_runner_sim --model_path=add.pte
```

Common `xt-run` options (these belong to `xt-run`, before the ELF path):

| `xt-run` option | Purpose |
| --------------- | ------- |
| `--turbo` | Fast functional (JIT) simulation. Much faster, but does **not** model cycles - use it for correctness/bring-up, not performance. |
| *(omit `--turbo`)* | Cycle-accurate simulation. Slower, but the reported cycle counts are meaningful. Use this to read `Execute cycles`. |
| `--mem_model` | Enable the memory hierarchy model (caches/local memories/latencies) for cycle-accurate runs, so timing reflects real memory behavior instead of assuming ideal single-cycle memory. |
| `--mem_model --nosummary` | Same, without the end-of-run ISS summary. |

Examples:

```bash
# Fast correctness check (functional, no timing)
xt-run --turbo cmake-out/backends/cadence/cadence_executor_runner_sim \
    --model_path=add.pte --print_output=all

# Cycle-accurate run with realistic memory timing
xt-run --mem_model cmake-out/backends/cadence/cadence_executor_runner_sim \
    --model_path=add.pte

# Feed real inputs and capture outputs to disk
xt-run --turbo cmake-out/backends/cadence/cadence_executor_runner_sim \
    --model_path=model.pte --inputs=in0.bin,in1.bin --dump-output

# Verify a BundledIO program (runner built with --bundle-io)
xt-run --turbo cmake-out/backends/cadence/cadence_executor_runner_sim \
    --model_path=model.bpte --bundle_rtol=1e-3 --bundle_atol=1e-3
```

> **Important:** Use `--turbo` while iterating on correctness; drop `--turbo`
> and add `--mem_model` when you need trustworthy cycle counts.

### 7.2 The ExecuTorch Install Subset

Stage 3 installs the **host / AOT** ExecuTorch package. The Cadence build does
not use a full install, but it passes a trimmed-down subset through `CMAKE_ARGS`. The
three variants below explain the trade-offs.

#### Normal (full) Install

```bash
./install_executorch.sh
```

Installs the full wheel, including all examples dependencies, default backends and kernels, and AOT tooling. This option is intended for general ExecuTorch development and install **more components than the Cadence backend requires.**

#### Minimal Install (`--minimal` / `-m`)

```bash
./install_executorch.sh --minimal
```

`--minimal` skips the extra packages that are only needed to run the example
scripts (`install_executorch.py` only calls
`install_optional_example_requirements()` when `--minimal` is not set). The
resulting wheel ships a slimmer runtime dependency set (the AOT-export subset:
`flatbuffers`, `numpy`, `packaging`, `pyyaml`, `ruamel.yaml`, `sympy`,
`tabulate`, `typing-extensions`).

A separate build-time knob, `EXECUTORCH_BUILD_MINIMAL=ON`, clears every
optional CMake target (`_minimal_cmake_flags()` in `setup.py`). It targets the
AOT-export-only wheel and is distinct from the Cadence subset below.

#### Cadence Custom Subset (the default for this backend)

The Cadence build does **not** use `--minimal` option. Instead, it passes the curated `CMAKE_ARGS` configuration from [Stage 3](#stage-3---install-the-host-executorch-package) to normal `./install_executorch.sh`.This configuration retains the example and development dependencies while clearing the host-side backends and kernels that are not used in DSP flow. The following table explains each flag and the reason for clearing it.

| Flag (`OFF`)                              | Why it is disabled for Cadence |
| ----------------------------------------- | ------------------------------ |
| `EXECUTORCH_BUILD_XNNPACK`                | Host CPU delegate; not used on the HiFi DSP. |
| `EXECUTORCH_BUILD_KERNELS_OPTIMIZED`      | Host-optimized (AVX/NEON) kernels; irrelevant to Xtensa. |
| `EXECUTORCH_BUILD_KERNELS_LLM`            | LLM custom kernels; not part of the DSP flow. |
| `EXECUTORCH_BUILD_KERNELS_LLM_AOT`        | AOT side of the LLM kernels. |
| `EXECUTORCH_BUILD_EXTENSION_LLM`          | LLM runtime extension. |
| `EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER`   | LLM runner extension. |
| `EXECUTORCH_BUILD_EXTENSION_TRAINING`     | On-device training; unused. |
| `EXECUTORCH_BUILD_COREML`                 | Apple backend; wrong platform. |
| `EXECUTORCH_BUILD_VULKAN`                 | GPU backend; wrong platform. |
| `EXECUTORCH_BUILD_OPENVINO`               | Intel backend; wrong platform. |
| `EXECUTORCH_BUILD_EXECUTOR_RUNNER`        | Default host runner; Cadence ships its own simulation runner. |
| `EXECUTORCH_BUILD_PYBIND`                 | Python bindings not needed for the AOT export used here. |
| `EXECUTORCH_BUILD_CMSIS_NN_PYBINDS`       | Cortex-M CMSIS-NN bindings; unrelated backend. |

This configuration reduces installation time and provides a faster host installation while retaining everything required to **export a `.pte`** and generate ExecuTorch CMake package used by cross-compilation build.

You can customize this configuration as needed. For example, to build the host Python bindings, remove `-DEXECUTORCH_BUILD_PYBIND=OFF` from the configuration or set it to `ON` before running the installation script.

### 7.3 Cadence-internal CMake Flags

These flags are specific to the Cadence backend and are defined in
[`backends/cadence/CMakeLists.txt`](CMakeLists.txt).

**`EXECUTORCH_HIFI_CORE`** - selects which HiFi kernel set is compiled. Accepts
`hifi1`, `hifi4`, or `hifi5` (set in Stage 1). Related target selectors, mutually
exclusive with the HiFi path and not set by the HiFi flow:

*   `EXECUTORCH_NNLIB_OPT` - force the generic `hifi` nnlib + kernels path.
*   `EXECUTORCH_FUSION_G3_OPT` - build the Fusion G3 target (`fusion_g3/...`).
*   `EXECUTORCH_VISION_OPT` - build the Vision target (`vision/kernels`).
*   If none are set, the backend falls back to the `generic` kernels.

**`EXECUTORCH_BUILD_CADENCE`** - master switch for building the Cadence backend
(operators + kernels). Set `ON` in the cross-compile step.

**`EXECUTORCH_BUILD_CADENCE_BUNDLE_IO`** - compiles BundledIO (`.bpte`) support
into `cadence_executor_runner_sim` (defines `ET_BUNDLE_IO_ENABLED` and links
`bundled_program`). **Requires** `-DEXECUTORCH_BUILD_DEVTOOLS=ON` - CMake
hard-errors otherwise. See [optional build features](#74-optional-build-features).

**`EXECUTORCH_BUILD_CADENCE_OP_TESTS`** - builds the op-level gtest suite
(cross-compiled for the Xtensa ISS) when the selected backend ships an
`operators/tests/CMakeLists.txt`.

### 7.4 Optional Build Features

Two optional features add extra CMake options to the Stage 5 cross-compile.
Append them to the `cmake` command directly, or use the convenience flags on
`build_cadence_hifi.sh`.

**BundledIO (`.bpte`)** - enables output verification against embedded reference
IO in the runner (see the BundledIO note in Section 7.1). Adds the following:

```
-DEXECUTORCH_BUILD_CADENCE_BUNDLE_IO=ON
-DEXECUTORCH_BUILD_DEVTOOLS=ON
```

Script equivalent: `./backends/cadence/build_cadence_hifi.sh --bundle-io`.

**Op tests** - builds the cross-compiled op-level gtest suite. Adds:

```
-DEXECUTORCH_BUILD_CADENCE_OP_TESTS=ON
```

Script equivalent: `./backends/cadence/build_cadence_hifi.sh --tests`.

---

## 8. Troubleshooting
A few common troubleshooting examples are

*   **Missing files or submodule corruption:** force Git to pull clean submodule
    states:

    ```bash
    git submodule update --init --recursive --force
    ```

*   **Stale Python / pip build artifacts:** purge the ExecuTorch pip build
    output:

    ```bash
    rm -rf pip-out/
    ```

*   **`Nothing found at XTENSA_TOOLCHAIN_PATH`:** `XTENSA_TOOLCHAIN` points at a
    path that does not exist; fix Section 3.

*   **Wrong or missing HiFi core:** ensure `EXECUTORCH_HIFI_CORE` (Stage 1) is
    one of `hifi1` / `hifi4` / `hifi5` and matches the core named by
    `XTENSA_CORE`.

*   **Patch does not apply cleanly:** `cadence_apply_patch.sh` reports a conflict
    in `third-party/gflags`. Reset that submodule
    (`git submodule update --init --recursive --force third-party/gflags`) and
    re-run.

*   **FlatBuffers / stdlib header errors with GCC 14/15:** switch to GCC 12/13 or
    Clang; bleeding-edge toolchains break third-party submodules.

---

## 9. Reference Script

[`build_cadence_hifi.sh`](build_cadence_hifi.sh) wires Stages 2-7 together into a
single script. It is provided as a **reference example** of one working
configuration, as it encodes specific choices (a fixed install subset, `-j8`, an
`add.pte` smoke test) that will not fit every setup. Prefer the manual stages
above and treat this script as a starting point to copy and adapt, not a
drop-in build for all environments.

---

## Copyright Information

Copyright © 2026 Cadence Design Systems, Inc. Cadence Design Systems, Inc. (Cadence), 2655 Seely Ave., San Jose, CA 95134, USA.

### Trademarks

Trademarks and service marks of Cadence Design Systems, Inc. (Cadence) contained in this document are attributed to Cadence with the appropriate symbol. For queries regarding Cadence's trademarks, contact the corporate legal department at the address shown above or call 1-800-862-4522. All other trademarks are the property of their respective holders.

### Restricted Permission

This publication is protected by copyright law and international treaties and contains trade secrets and proprietary information owned by Cadence. Unauthorized reproduction or distribution of this publication, or any portion of it, may result in civil and criminal penalties. Except as expressly permitted below, this publication may not be copied, reproduced, modified, published, uploaded, posted, transmitted, or distributed in any way, including automated processes, training or services involving a large language model, foundation model, deep machine learning, generative artificial intelligence, or any other similar process or technology, without prior written permission from Cadence. 

Unless otherwise agreed to by Cadence in writing, customers are granted permission to print one (1) hard copy of this publication, subject to the following conditions:

- The publication may be used solely for personal, informational, and noncommercial purposes.
- The publication may not be modified in any way.
- Any copy of the publication or portion thereof must include all original copyright, trademark, and other proprietary notices and this permission statement.
- The information contained in this document cannot be used in the development of like products or software, whether for internal or external use, and shall not be used for the benefit of any other party, whether or not for consideration.
- Cadence reserves the right to revoke this authorization at any time, and any such use shall be discontinued immediately upon written notice from Cadence.

### Disclaimer

Information in this publication is subject to change without notice and does not represent a commitment on the part of Cadence. The information contained herein is the proprietary and confidential information of Cadence or its licensors, and is supplied subject to, and may be used only by Cadence's customer in accordance with, a written agreement between Cadence and its customer. Except as may be explicitly set forth in such agreement, Cadence does not make, and expressly disclaims, any representations or warranties as to the completeness, accuracy or usefulness of the information contained in this document. Cadence does not warrant that use of such information will not infringe any third party rights, nor does Cadence assume any liability for damages or costs of any kind that may result from use of such information.

### Restricted Rights

Use, duplication, or disclosure by the Government is subject to restrictions as set forth in FAR52.227-14 and DFAR252.227-7013 et seq. or its successor.

### Support

For further assistance, contact Cadence ASK at [https://support.cadence.com/](https://support.cadence.com/).


Product Release: ExecuTorch HiFi Early Access
Last Updated: 08/2026
Version: 1.0
