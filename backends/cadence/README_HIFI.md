# Building ExecuTorch for the Cadence HiFi DSP

This guide builds the Cadence HiFi backend **manually**, one stage at a time,
documenting each command, the ExecuTorch install subset, and every
Cadence-internal CMake flag. Follow the manual stages when you need to customize
the flow, debug a failure, or integrate the build into a larger system.

---

## 1. Prerequisites

Ensure the following are installed and available on your host before building:

*   **Python:** **3.12-3.13**, in an active virtual environment.
*   **Host Compiler (C++17):** `GCC` **12** or `Clang` **5.0+**.
*   **System Assembler (`binutils`):** **2.40+** (for AVX-512/BF16 host kernels).
*   **Cadence Xtensa toolchain** binaries reachable on your `PATH`
    (`xt-clang`, `xt-ld`, `xt-run`, ...).

---

## 2. Python Virtual Environment Setup

Always use a clean environment to isolate dependencies and prevent library
pollution. Run these from the `executorch` root directory.

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

Export the variables that point at your local Xtensa installation and target
core, and put the toolchain binaries (`xt-clang`, `xt-ld`, `xt-run`, ...) on your
`PATH`. The Xtensa toolchain **requires** `XTENSA_CORE`; the HiFi kernel set you
build is chosen separately in [Stage 1](#stage-1--choose-the-hifi-core) (see
[Cadence flags](#63-cadence-internal-cmake-flags)).

**Bash / Zsh:**

```bash
# Path to the Xtensa tools install dir (e.g. /opt/Xtensa/XtDevTools/install/tools)
export XTENSA_TOOLCHAIN=/path/to/XtDevTools/install/tools

# Toolchain version release string (e.g. RI-2023.11-linux)
export TOOLCHAIN_VER=your_release_version

# Xtensa configuration registry directory
export XTENSA_SYSTEM=${XTENSA_TOOLCHAIN}/${TOOLCHAIN_VER}/XtensaTools/config/

# Target core configuration profile (e.g. AE_HiFi5s_LE5_AO_FP_XC)
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

## 4. Manual Build - Stage by Stage

Run every command below from the `executorch` root with your virtual environment
active (Section 2) and the `XTENSA_*` variables exported (Section 3). Bash / Zsh
syntax is shown; adapt variable assignments for Csh/Tcsh if needed.

### Stage 1 - Choose the HiFi core

Set the HiFi version you are targeting. Accepted values are `hifi1`, `hifi4`, or
`hifi5`; it is passed to CMake later as `-DEXECUTORCH_HIFI_CORE`. Leave it empty
(or set any other value) to fall back to the portable `generic` kernels, which
run on any Xtensa core without the HiFi NN libraries.

```bash
export HIFI_CORE=hifi5   # or hifi1 / hifi4; empty for the generic (portable) kernels
```

### Stage 2 - Prepare the source tree

Clear any stale prefix path, sync and update submodules, and apply the Cadence
gflags patch. The patch step is idempotent (it detects an already-applied patch).

```bash
unset CMAKE_PREFIX_PATH

git submodule sync
git submodule update --init --recursive

# Applies backends/cadence/cadence_patch.patch to third-party/gflags
./backends/cadence/cadence_apply_patch.sh
```

### Stage 3 - Install the host / AOT ExecuTorch package

This produces the Python package used to export models and the host-side
ExecuTorch CMake config (`cmake-out/lib/cmake/ExecuTorch`) that the cross-compile
links against. Rather than a full install, use the curated `CMAKE_ARGS` subset
below - it disables host backends/kernels the DSP flow never uses. See
[Section 6.2](#62-the-executorch-install-subset) for why each flag is off and how
this differs from `--minimal`.

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

You only need this stage when Python deps or the host build change. For plain
kernel/operator recompiles, skip it (see the Rebuilds note at the end of this
section).

### Stage 4 - Fetch Cadence NN libraries and FACTO

Clone the vendor kernel libraries and install the FACTO operator-testing package
(editable). Already-cloned repos are skipped, so this is safe to re-run.

```bash
./backends/cadence/install_requirements.sh
```

This populates:

*   `nnlib-hifi4`  -> `hifi/third-party/nnlib/nnlib-hifi4`
*   `nnlib-hifi5`  -> `hifi/third-party/nnlib/nnlib-hifi5`
*   `nnlib-FusionG3` (pinned commit) -> `fusion_g3/third-party/nnlib/nnlib-FusionG3`
*   `FACTO`        -> `pip install -e backends/cadence/utils/FACTO`

### Stage 5 - Configure the cross-compile for the Xtensa target

> **NOTE - Host compiler.** Although the target is Xtensa, this stage still needs
> a working **host** GCC / binutils. The build compiles host-side tooling such as
> `flatcc`, which generates the FlatBuffers headers that `libexecutorch.a` is
> built against. **On an old GCC / binutils the build will fail** (missing
> AVX-512/BF16 assembler support, stale FlatBuffers header handling, etc.). We
> recommend **GCC 12** with **binutils 2.40+**; the versions verified for this
> guide are **GCC 12.4.0** and **binutils 2.40**.
>
> If the required GCC / binutils are not the system default (common on
> shared/managed hosts), point the build at them explicitly *before* configuring -
> edit the two roots to match your environment:
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
> Verify with `which gcc g++ ld`, `gcc --version` (12.x), and `ld --version`
> (2.40+) before configuring. The `CXXFLAGS` below extends this value; the
> `-B...` redirect is preserved.

Configure CMake with the Cadence cross toolchain. Key options:

*   `CXXFLAGS="-fno-exceptions -fno-rtti"` - the Xtensa runtime is built without
    C++ exceptions or RTTI.
*   `-DCMAKE_TOOLCHAIN_FILE=backends/cadence/cadence.cmake` - selects the Xtensa
    cross toolchain (`xt-clang` / `xt-ld`) using your `XTENSA_*` env vars.
*   `-DEXECUTORCH_BUILD_CADENCE=ON` / `-DEXECUTORCH_HIFI_CORE="$HIFI_CORE"` - the
    Cadence-internal switches described in
    [Section 6.3](#63-cadence-internal-cmake-flags).
*   `-DCMAKE_PREFIX_PATH=...` - points at the host package built in Stage 3.

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

To enable BundledIO or the op tests, add the extra flags described in
[Section 6.4](#64-optional-build-features) to this command.

### Stage 6 - Build and install

```bash
cmake --build cmake-out --target install --config Release -j8
```

### Stage 7 - Run on the Xtensa ISS

Run a `.pte` on the instruction-set simulator with the runner built above:

```bash
xt-run --turbo cmake-out/backends/cadence/cadence_executor_runner_sim --model_path=add.pte
```

`add.pte` is a trivial model you can generate with the portable AOT export flow:

```bash
python3 -m examples.portable.scripts.export --model_name="add"
```

See [Additional Information](#6-additional-information) for the full runner CLI,
`xt-run` options, and the ExecuTorch install variants.

> **NOTE - Rebuilds.** After the first successful build, recompile C++ changes
> directly without repeating Stages 3-4:
>
> ```bash
> cmake --build cmake-out --target install --config Release -j8
> ```
>
> Re-run Stage 3 only when Python dependencies or the host ExecuTorch build
> change.

---

## 5. Exporting Models to `.pte`

The Cadence AOT flow quantizes a PyTorch `nn.Module`, lowers it through the
Cadence-specific passes, and serializes it to a `.pte` (and a `.bpte` bundled
program with reference IO). The entry points live in
[`backends/cadence/aot/export_example.py`](aot/export_example.py):

*   **`export_model(model, example_inputs, ...)`** - runs the full export
    pipeline (prepare -> calibrate -> convert -> quantize -> lower) and writes the
    `.pte` / `.bpte`. Returns the `ExecutorchProgramManager`.
*   **`export_and_run_model(model, example_inputs, ...)`** - calls `export_model`,
    and (only when `verify=True`) runs the result on the ISS and compares against
    the eager reference outputs.

### 5.1 Example models

Ready-to-run examples live under
[`examples/cadence/models/`](../../examples/cadence/models/) - refer to them as
templates for exporting your own model:

*   `babyllama.py` - a small Llama transformer
*   `mobilenet_v2.py`, `resnet50.py` - torchvision CNNs
*   `vision_transformer.py` - ViT
*   `wav2vec2.py` - speech model
*   `rnnt_encoder.py`, `rnnt_predictor.py`, `rnnt_joiner.py` - RNN-T components

Each script builds a model plus `example_inputs` and calls one of the export
entry points. Run one with:

```bash
python3 -m examples.cadence.models.mobilenet_v2
```

### 5.2 `export_and_run_model` parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `model` | *(required)* | The `nn.Module` to export (set to `eval()` for inference models). |
| `example_inputs` | *(required)* | Tuple of example input tensors defining the input shapes/dtypes. |
| `file_name` | `"CadenceDemoModel"` | Base name for the output `.pte` / `.bpte`. |
| `verify` | `False` | When `True`, run the exported program on the ISS and compare against eager reference outputs. When `False` (default), only export - no execution. |
| `opt_level` | `1` | Cadence optimization level for lowering. |
| `mem_algo` | `0` | Memory-planning algorithm selector. |
| `edge_passes_config` | `None` | Optional `EdgePassesConfig` to toggle edge passes (e.g. `use_im2row_transform=True`). |
| `additional_quantizers` | `None` | Extra quantizers, prepended so they take priority over the A8W8 defaults. |
| `eps_error` / `eps_warn` | `1e-1` / `1e-5` | Error/warning thresholds used only when `verify=True`. |
| `force_rebuild` | `False` | Force the runtime rebuild during verification. |

`export_model` accepts the same export-related parameters (`file_name`,
`working_dir`, `opt_level`, `mem_algo`, `edge_passes_config`,
`additional_quantizers`) but does no execution.

### 5.3 Locating the exported file

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

## 6. Additional Information

Reference material for the build above: runner and Xtensa ISS details, the
ExecuTorch install variants, the Cadence-internal CMake flags, and optional
build features.

### 6.1 Runner and Xtensa ISS details

The build produces `cadence_executor_runner_sim`
(`cmake-out/backends/cadence/cadence_executor_runner_sim`), a self-contained
runner compiled from
[`executor_runner_sim.cpp`](executor_runner_sim.cpp). It loads a
`.pte` (or `.bpte`), runs the first method, and prints/dumps IO. It has **no
gflags and no threadpool dependency** and uses a plain `argv` parser.

**Runner CLI flags** - parsed by the runner itself, independent of how you launch
it. Value flags use `--flag=value`; boolean flags are bare.

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

Notes:

*   **Random input fill** covers `float` (uniform in `[-1, 1]`), `int32`/`int64`
    (small non-negative values, safe for token IDs/indices), `int8`, and `bool`.
    Other dtypes are left unmodified - supply them with `--inputs` instead.
*   **BundledIO (`.bpte`)** is auto-detected at runtime *if* the runner was built
    with `-DEXECUTORCH_BUILD_CADENCE_BUNDLE_IO=ON` (Section 6.4). In that mode the
    runner ignores `--inputs`, loads the embedded reference inputs, runs, prints
    error stats (mean/max abs & relative error), and verifies outputs against the
    embedded references using `--bundle_rtol` / `--bundle_atol`. On mismatch it
    logs `Test_result: FAIL` and exits non-zero (usable as a CI gate).
*   The runner reports execution cost as `Execute cycles = <n>` (from `times()`),
    and its memory pools are fixed at compile time (two 4 MB arenas plus
    dynamically sized planned buffers).

**Launching with `xt-run`.** `xt-run` executes the cross-compiled ELF on the
instruction-set simulator and provides **semi-hosting**, so the runner's file I/O
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

> **Rule of thumb.** Use `--turbo` while iterating on correctness; drop `--turbo`
> and add `--mem_model` when you need trustworthy cycle counts.

### 6.2 The ExecuTorch install subset

Stage 3 installs the **host / AOT** ExecuTorch package. The Cadence build does
not use a full install - it passes a trimmed-down subset via `CMAKE_ARGS`. The
three variants below explain the trade-offs.

#### Normal (full) install

```bash
./install_executorch.sh
```

Installs the full wheel: all example dependencies, every default backend/kernel
that the top-level preset enables, plus AOT tooling. Heaviest option;
appropriate for general ExecuTorch development but **more than the Cadence
backend needs**.

#### Minimal install (`--minimal` / `-m`)

```bash
./install_executorch.sh --minimal
```

`--minimal` skips the extra packages that are only needed to run the example
scripts (`install_executorch.py` only calls
`install_optional_example_requirements()` when `--minimal` is *not* set). The
resulting wheel ships a slimmer runtime dependency set (the AOT-export subset:
`flatbuffers`, `numpy`, `packaging`, `pyyaml`, `ruamel.yaml`, `sympy`,
`tabulate`, `typing-extensions`).

A separate build-time knob, `EXECUTORCH_BUILD_MINIMAL=ON`, disables every
optional CMake target (`_minimal_cmake_flags()` in `setup.py`). It targets the
AOT-export-only wheel and is distinct from the Cadence subset below.

#### Cadence custom subset (the default for this backend)

The Cadence build does **not** pass `--minimal`. Instead it feeds the curated
`CMAKE_ARGS` list from
[Stage 3](#stage-3--install-the-host--aot-executorch-package) to a normal
`./install_executorch.sh`. This keeps the example/dev dependencies but explicitly
disables the host-side backends and kernels the DSP flow will never use. Each
flag and its rationale:

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

Net effect: a faster, lighter host install that still contains everything needed
to **export a `.pte`** and produce the ExecuTorch CMake package that the
cross-compile step links against.

You can extend this subset. To also produce the host Python bindings, for
example, drop `-DEXECUTORCH_BUILD_PYBIND=OFF` from the list (or set it to `ON`)
before running the install.

### 6.3 Cadence-internal CMake flags

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
hard-errors otherwise. See [optional build features](#64-optional-build-features).

**`EXECUTORCH_BUILD_CADENCE_OP_TESTS`** - builds the op-level gtest suite
(cross-compiled for the Xtensa ISS) when the selected backend ships an
`operators/tests/CMakeLists.txt`.

### 6.4 Optional build features

Two optional features add extra CMake options to the Stage 5 cross-compile.
Append them to the `cmake` command directly, or use the convenience flags on
`build_cadence_hifi.sh`.

**BundledIO (`.bpte`)** - enables output verification against embedded reference
IO in the runner (see the BundledIO note in Section 6.1). Adds:

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

## 7. Troubleshooting

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

## 8. Reference Script

[`build_cadence_hifi.sh`](build_cadence_hifi.sh) wires Stages 2-7 together into a
single script. It is provided as a **reference example** of one working
configuration - it encodes specific choices (a fixed install subset, `-j8`, an
`add.pte` smoke test) that will not fit every setup. Prefer the manual stages
above and treat this script as a starting point to copy and adapt, not a
drop-in build for all environments.
