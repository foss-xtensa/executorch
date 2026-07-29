/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Default ExecuTorch runner for the Cadence backend (built as the
 * `executor_runner` target). Targets the Xtensa ISS via xt-run semi-hosting,
 * executes the first method, and reports outputs. No gflags, no threadpool.
 *
 * Inputs default to deterministic pseudo-random values or raw .bin files
 * (--inputs). Prints a compact summary by default; --print_output=all matches
 * the portable runner. BundledIO (.bpte) is auto-detected when built with
 * ET_BUNDLE_IO_ENABLED and verified against the embedded reference outputs.
 */

#include <sys/times.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
// patternlint-disable executorch-cpp-nostdinc
#include <optional>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#ifdef ET_BUNDLE_IO_ENABLED
#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#endif

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB
static uint8_t temp_allocator_pool[4 * 1024U * 1024U]; // 4 MB

using executorch::aten::ScalarType;
using executorch::extension::FileDataLoader;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
#ifdef ET_BUNDLE_IO_ENABLED
using executorch::bundled_program::compute_method_output_error_stats;
using executorch::bundled_program::ErrorStats;
using executorch::bundled_program::verify_method_outputs;
using executorch::extension::BufferDataLoader;
#endif

namespace {

// Leading/trailing elements shown by summary mode for inputs and outputs.
constexpr std::size_t kInputEdgeItems = 10;
constexpr std::size_t kOutputEdgeItems = 20;

#ifdef ET_BUNDLE_IO_ENABLED
// Only the first bundled test set is exercised.
constexpr std::size_t kBundleTestsetIdx = 0;
#endif

// Controls how much of a tensor is printed.
// Inputs support None|Summary only (summary = first/last kInputEdgeItems).
// Outputs additionally support All (every element).
enum class PrintMode { None, Summary, All };

// Parsed command-line configuration.
struct Options {
  const char* model_path = nullptr;
  PrintMode print_input = PrintMode::Summary;
  PrintMode print_output = PrintMode::Summary;
  bool dump_input = false;
  bool dump_output = false;
  std::vector<std::string> input_files; // From --inputs, in tensor order.
#ifdef ET_BUNDLE_IO_ENABLED
  double bundle_rtol = 0.01; // Relative tolerance for bundled-IO verification.
  double bundle_atol = 0.01; // Absolute tolerance for bundled-IO verification.
#endif
};

// Returns the value of a "--flag=value" argument, or nullptr if arg is not
// this flag.
const char* match_value_flag(const char* flag, const char* arg) {
  const std::size_t len = std::strlen(flag);
  if (std::strncmp(arg, flag, len) == 0 && arg[len] == '=') {
    return arg + len + 1;
  }
  return nullptr;
}

// Returns true if arg is exactly the given boolean flag.
bool match_bool_flag(const char* flag, const char* arg) {
  return std::strcmp(arg, flag) == 0;
}

// Inputs support only none|summary.
PrintMode parse_input_print_mode(const char* value) {
  if (std::strcmp(value, "none") == 0) {
    return PrintMode::None;
  }
  if (std::strcmp(value, "summary") == 0) {
    return PrintMode::Summary;
  }
  ET_CHECK_MSG(
      false,
      "Unknown --print_input mode '%s'; expected 'none' or 'summary'.",
      value);
  return PrintMode::None;
}

// Outputs support none|summary|all.
PrintMode parse_output_print_mode(const char* value) {
  if (std::strcmp(value, "none") == 0) {
    return PrintMode::None;
  }
  if (std::strcmp(value, "summary") == 0) {
    return PrintMode::Summary;
  }
  if (std::strcmp(value, "all") == 0) {
    return PrintMode::All;
  }
  ET_CHECK_MSG(
      false,
      "Unknown --print_output mode '%s'; expected 'none', 'summary', or 'all'.",
      value);
  return PrintMode::None;
}

// Splits a comma-separated list into its elements.
std::vector<std::string> split_csv(const char* csv) {
  std::vector<std::string> parts;
  const char* start = csv;
  for (const char* p = csv;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (p != start) {
        parts.emplace_back(start, p - start);
      }
      if (*p == '\0') {
        break;
      }
      start = p + 1;
    }
  }
  return parts;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (const char* v = match_value_flag("--model_path", arg)) {
      ET_CHECK_MSG(*v != '\0', "--model_path was given an empty value.");
      opts.model_path = v;
    } else if (const char* v = match_value_flag("--print_input", arg)) {
      opts.print_input = parse_input_print_mode(v);
    } else if (const char* v = match_value_flag("--print_output", arg)) {
      opts.print_output = parse_output_print_mode(v);
    } else if (const char* v = match_value_flag("--inputs", arg)) {
      opts.input_files = split_csv(v);
#ifdef ET_BUNDLE_IO_ENABLED
    } else if (const char* v = match_value_flag("--bundle_rtol", arg)) {
      opts.bundle_rtol = std::atof(v);
    } else if (const char* v = match_value_flag("--bundle_atol", arg)) {
      opts.bundle_atol = std::atof(v);
#endif
    } else if (match_bool_flag("--dump-input", arg)) {
      opts.dump_input = true;
    } else if (match_bool_flag("--dump-output", arg)) {
      opts.dump_output = true;
    } else {
      ET_CHECK_MSG(false, "Unrecognized argument '%s'.", arg);
    }
  }
  ET_CHECK_MSG(
      opts.model_path != nullptr,
      "Required argument --model_path was not provided.");
  return opts;
}

// Derives the model name from a path by stripping the directory prefix and
// the ".pte" extension, e.g. "../models/babyllama.pte" -> "babyllama".
std::string model_name_from_path(const char* model_path) {
  std::string path{model_path};
  const std::size_t slash = path.find_last_of("/\\");
  if (slash != std::string::npos) {
    path = path.substr(slash + 1);
  }
  const std::size_t dot = path.find_last_of('.');
  if (dot != std::string::npos) {
    path = path.substr(0, dot);
  }
  return path;
}

// Overwrites a tensor's data with deterministic pseudo-random values. Float
// tensors are filled with uniform noise in [-1, 1]; integer tensors with small
// non-negative values (safe for token IDs and indices); bool tensors with 0/1.
// The RNG is seeded once with a constant so runs are reproducible.
void fill_random(const executorch::aten::Tensor& tensor) {
  const int64_t numel = tensor.numel();
  switch (tensor.scalar_type()) {
    case ScalarType::Float: {
      float* p = tensor.mutable_data_ptr<float>();
      for (int64_t j = 0; j < numel; ++j) {
        p[j] = 2.0f *
                (static_cast<float>(std::rand()) /
                 static_cast<float>(RAND_MAX)) -
            1.0f;
      }
      break;
    }
    case ScalarType::Long: {
      int64_t* p = tensor.mutable_data_ptr<int64_t>();
      for (int64_t j = 0; j < numel; ++j) {
        p[j] = std::rand() % 100;
      }
      break;
    }
    case ScalarType::Int: {
      int32_t* p = tensor.mutable_data_ptr<int32_t>();
      for (int64_t j = 0; j < numel; ++j) {
        p[j] = std::rand() % 100;
      }
      break;
    }
    case ScalarType::Char: {
      int8_t* p = tensor.mutable_data_ptr<int8_t>();
      for (int64_t j = 0; j < numel; ++j) {
        p[j] = static_cast<int8_t>(std::rand() % 128);
      }
      break;
    }
    case ScalarType::Bool: {
      bool* p = tensor.mutable_data_ptr<bool>();
      for (int64_t j = 0; j < numel; ++j) {
        p[j] = (std::rand() & 1) != 0;
      }
      break;
    }
    default:
      ET_LOG(
          Info,
          "Leaving input of scalar type %d unmodified (random fill "
          "unsupported).",
          static_cast<int>(tensor.scalar_type()));
      break;
  }
}

// Loads raw bytes from a .bin file into a tensor's data buffer. The file size
// must match the tensor's byte count exactly.
void load_tensor_from_file(
    const executorch::aten::Tensor& tensor,
    const std::string& file_path) {
  FILE* f = std::fopen(file_path.c_str(), "rb");
  ET_CHECK_MSG(
      f != nullptr, "Failed to open input file '%s'.", file_path.c_str());

  std::fseek(f, 0, SEEK_END);
  const long file_size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  ET_CHECK_MSG(
      file_size >= 0 && static_cast<std::size_t>(file_size) == tensor.nbytes(),
      "Input file '%s' is %ld bytes but tensor expects %zu bytes.",
      file_path.c_str(),
      file_size,
      tensor.nbytes());

  const std::size_t read =
      std::fread(tensor.mutable_data_ptr(), 1, tensor.nbytes(), f);
  std::fclose(f);
  ET_CHECK_MSG(
      read == tensor.nbytes(),
      "Read %zu of %zu bytes from '%s'.",
      read,
      tensor.nbytes(),
      file_path.c_str());
}

#ifdef ET_BUNDLE_IO_ENABLED
// Reads an entire file into memory using plain stdio. A bundled program (.bpte)
// interleaves the .pte with its reference IO, so the whole file must be
// resident in RAM to detect and unpack it. Using FILE* here (rather than
// std::filesystem) keeps the runner compatible with Xtensa newlib.
std::vector<uint8_t> read_whole_file(const char* path) {
  FILE* f = std::fopen(path, "rb");
  ET_CHECK_MSG(f != nullptr, "Failed to open file '%s'.", path);

  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  ET_CHECK_MSG(size >= 0, "Failed to size file '%s'.", path);

  std::vector<uint8_t> data(static_cast<std::size_t>(size));
  const std::size_t read = std::fread(data.data(), 1, data.size(), f);
  std::fclose(f);
  ET_CHECK_MSG(
      read == data.size(),
      "Read %zu of %zu bytes from '%s'.",
      read,
      data.size(),
      path);
  return data;
}
#endif // ET_BUNDLE_IO_ENABLED

// Writes a tensor's raw bytes to "<model_name>-<tag>-<index>.bin".
void dump_tensor(
    const executorch::aten::Tensor& tensor,
    const std::string& model_name,
    const char* tag,
    std::size_t index) {
  char filename[256];
  std::snprintf(
      filename,
      sizeof(filename),
      "%s-%s-%zu.bin",
      model_name.c_str(),
      tag,
      index);

  FILE* f = std::fopen(filename, "wb");
  if (f == nullptr) {
    ET_LOG(Error, "Failed to open '%s' for writing.", filename);
    return;
  }
  std::fwrite(tensor.const_data_ptr(), 1, tensor.nbytes(), f);
  std::fclose(f);
  ET_LOG(Info, "Wrote %s tensor %zu to '%s'.", tag, index, filename);
}

// Appends a single tensor element (in its native scalar type) to out. Returns
// the number of characters written, mirroring snprintf semantics.
int format_element(
    char* out,
    std::size_t cap,
    const executorch::aten::Tensor& tensor,
    std::size_t j) {
  switch (tensor.scalar_type()) {
    case ScalarType::Float:
      return std::snprintf(out, cap, "%g", tensor.const_data_ptr<float>()[j]);
    case ScalarType::Long:
      return std::snprintf(
          out, cap, "%" PRId64, tensor.const_data_ptr<int64_t>()[j]);
    case ScalarType::Int:
      return std::snprintf(
          out, cap, "%" PRId32, tensor.const_data_ptr<int32_t>()[j]);
    case ScalarType::Char:
      return std::snprintf(
          out, cap, "%d", static_cast<int>(tensor.const_data_ptr<int8_t>()[j]));
    case ScalarType::Bool:
      return std::snprintf(
          out, cap, "%s", tensor.const_data_ptr<bool>()[j] ? "true" : "false");
    default:
      return std::snprintf(out, cap, "?");
  }
}

// Logs a single tensor element on its own line, e.g. "output[0][3]: 42".
void log_element(
    const executorch::aten::Tensor& tensor,
    const char* tag,
    std::size_t index,
    std::size_t j) {
  char buf[64];
  format_element(buf, sizeof(buf), tensor, j);
  ET_LOG(Info, "%s[%zu][%zu]: %s", tag, index, j, buf);
}

// Number of elements packed into each summary ET_LOG line. ET_LOG truncates
// messages at a fixed internal buffer (256 chars), so a large tensor's summary
// is split across several lines instead of overflowing a single ET_LOG call.
constexpr std::size_t kSummaryElemsPerLine = 8;

// Prints a tensor via ET_LOG. All mode prints one element per line; Summary
// mode prints a compact first/last preview wrapped across lines.
void print_tensor(
    const executorch::aten::Tensor& tensor,
    const char* tag,
    std::size_t index,
    PrintMode mode,
    std::size_t edge_items) {
  const std::size_t numel = static_cast<std::size_t>(tensor.numel());

  if (mode == PrintMode::All) {
    for (std::size_t j = 0; j < numel; ++j) {
      log_element(tensor, tag, index, j);
    }
    return;
  }

  // Summary header, then the preview elements wrapped across ET_LOG lines.
  const bool elide = numel > 2 * edge_items;
  ET_LOG(
      Info,
      "%s tensor %zu [%zu elements]%s:",
      tag,
      index,
      numel,
      elide ? " (first/last)" : "");

  // Builds one bounded line at a time and flushes it via ET_LOG.
  char line[192];
  std::size_t len = 0;
  std::size_t on_line = 0;
  const auto flush = [&]() {
    if (len > 0) {
      ET_LOG(Info, "  %s", line);
      len = 0;
      on_line = 0;
    }
  };
  const auto emit = [&](std::size_t j, bool last) {
    char buf[64];
    format_element(buf, sizeof(buf), tensor, j);
    len += static_cast<std::size_t>(std::snprintf(
        line + len, sizeof(line) - len, "%s%s", buf, last ? "" : ", "));
    if (++on_line >= kSummaryElemsPerLine) {
      flush();
    }
  };

  if (!elide) {
    for (std::size_t j = 0; j < numel; ++j) {
      emit(j, j + 1 == numel);
    }
  } else {
    for (std::size_t j = 0; j < edge_items; ++j) {
      emit(j, false);
    }
    flush();
    ET_LOG(Info, "  ...");
    for (std::size_t j = numel - edge_items; j < numel; ++j) {
      emit(j, j + 1 == numel);
    }
  }
  flush();
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  const Options opts = parse_args(argc, argv);
  const std::string model_name = model_name_from_path(opts.model_path);

  // Select a data loader for the program. Bundled programs (.bpte) embed the
  // .pte alongside reference IO, so they are loaded fully into RAM and the
  // program bytes are unpacked from that buffer; plain .pte files stream from
  // disk via FileDataLoader.
  std::unique_ptr<DataLoader> loader;
  bool bundle_io = false;
#ifdef ET_BUNDLE_IO_ENABLED
  std::vector<uint8_t> file_data = read_whole_file(opts.model_path);
  bundle_io = executorch::bundled_program::is_bundled_program(
      file_data.data(), file_data.size());
  if (bundle_io) {
    ET_LOG(Info, "Bundled-IO program detected.");
    const void* program_data = nullptr;
    size_t program_data_len = 0;
    const Error status = executorch::bundled_program::get_program_data(
        file_data.data(), file_data.size(), &program_data, &program_data_len);
    ET_CHECK_MSG(
        status == Error::Ok,
        "get_program_data() failed: 0x%" PRIx32,
        (uint32_t)status);
    loader = std::make_unique<BufferDataLoader>(program_data, program_data_len);
  }
#endif // ET_BUNDLE_IO_ENABLED
  if (!bundle_io) {
    Result<FileDataLoader> loader_result =
        FileDataLoader::from(opts.model_path);
    ET_CHECK_MSG(
        loader_result.ok(),
        "FileDataLoader::from('%s') failed: 0x%" PRIx32,
        opts.model_path,
        (uint32_t)loader_result.error());
    loader = std::make_unique<FileDataLoader>(std::move(loader_result.get()));
  }

  Result<Program> program = Program::load(loader.get());
  ET_CHECK_MSG(
      program.ok(),
      "Program::load failed for '%s': 0x%" PRIx32,
      opts.model_path,
      (uint32_t)program.error());
  ET_LOG(Info, "Model file %s is loaded.", opts.model_path);

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta.error());

  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};
  MemoryAllocator temp_allocator{
      MemoryAllocator(sizeof(temp_allocator_pool), temp_allocator_pool)};

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  const std::size_t num_planned = method_meta->num_memory_planned_buffers();
  for (std::size_t id = 0; id < num_planned; ++id) {
    const std::size_t buffer_size = static_cast<std::size_t>(
        method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory{
      Span<Span<uint8_t>>(planned_spans.data(), planned_spans.size())};
  MemoryManager memory_manager{
      &method_allocator, &planned_memory, &temp_allocator};

  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());
  ET_LOG(Info, "Method loaded.");

  std::optional<executorch::extension::BufferCleanup> inputs_cleanup;

#ifdef ET_BUNDLE_IO_ENABLED
  if (bundle_io) {
    const Error status = executorch::bundled_program::load_bundled_input(
        *method, file_data.data(), kBundleTestsetIdx);
    ET_CHECK_MSG(
        status == Error::Ok,
        "load_bundled_input() failed: 0x%" PRIx32,
        (uint32_t)status);
  }
#endif // ET_BUNDLE_IO_ENABLED

  if (!bundle_io) {
    auto prepared = executorch::extension::prepare_input_tensors(*method);
    ET_CHECK_MSG(
        prepared.ok(),
        "prepare_input_tensors() failed: 0x%" PRIx32,
        (uint32_t)prepared.error());
    inputs_cleanup.emplace(std::move(prepared.get()));
  }

  std::vector<EValue> inputs(method->inputs_size());
  const Error get_inputs_status =
      method->get_inputs(inputs.data(), inputs.size());
  ET_CHECK_MSG(
      get_inputs_status == Error::Ok,
      "get_inputs() failed with status 0x%" PRIx32,
      (uint32_t)get_inputs_status);

  const bool use_files = !bundle_io && !opts.input_files.empty();
  ET_CHECK_MSG(
      !use_files || opts.input_files.size() == inputs.size(),
      "--inputs provided %zu file(s) but method expects %zu input(s).",
      opts.input_files.size(),
      inputs.size());

  std::srand(0);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].isTensor()) {
      ET_LOG(Info, "input[%zu]: non-tensor", i);
      continue;
    }
    const auto& tensor = inputs[i].toTensor();
    if (!bundle_io) {
      if (use_files) {
        load_tensor_from_file(tensor, opts.input_files[i]);
      } else {
        fill_random(tensor);
      }
    }
    if (opts.dump_input) {
      dump_tensor(tensor, model_name, "in", i);
    }
    if (opts.print_input != PrintMode::None) {
      print_tensor(tensor, "input", i, opts.print_input, kInputEdgeItems);
    }
  }

  ET_LOG(Info, "Starting model execution...");

  struct tms t_start, t_stop;
  times(&t_start);
  const Error exec_status = method->execute();
  times(&t_stop);
  ET_LOG(
      Info,
      "Execute cycles = %ld",
      static_cast<long>(t_stop.tms_utime - t_start.tms_utime));

  ET_CHECK_MSG(
      exec_status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)exec_status);
  ET_LOG(Info, "Model executed successfully.");

  std::vector<EValue> outputs(method->outputs_size());
  const Error get_outputs_status =
      method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK_MSG(
      get_outputs_status == Error::Ok,
      "get_outputs() failed with status 0x%" PRIx32,
      (uint32_t)get_outputs_status);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].isTensor()) {
      ET_LOG(Info, "output[%zu]: non-tensor", i);
      continue;
    }
    const auto& tensor = outputs[i].toTensor();
    if (opts.dump_output) {
      dump_tensor(tensor, model_name, "out", i);
    }
    if (opts.print_output != PrintMode::None) {
      print_tensor(tensor, "output", i, opts.print_output, kOutputEdgeItems);
    }
  }

#ifdef ET_BUNDLE_IO_ENABLED
  if (bundle_io) {
    const ErrorStats stats = compute_method_output_error_stats(
        *method, file_data.data(), kBundleTestsetIdx);
    if (stats.status == Error::Ok) {
      ET_LOG(Info, "=== Error stats for testset %zu ===", kBundleTestsetIdx);
      ET_LOG(Info, "  mean_absolute_error: %f", stats.mean_abs_error);
      ET_LOG(Info, "  max_absolute_error:  %f", stats.max_abs_error);
      ET_LOG(Info, "  mean_relative_error: %f", stats.mean_relative_error);
      ET_LOG(Info, "  max_relative_error:  %f", stats.max_relative_error);
    } else {
      ET_LOG(
          Error,
          "Failed to compute error stats for testset %zu: 0x%" PRIx32,
          kBundleTestsetIdx,
          (uint32_t)stats.status);
    }

    const Error verify_status = verify_method_outputs(
        *method,
        file_data.data(),
        kBundleTestsetIdx,
        opts.bundle_rtol,
        opts.bundle_atol);
    if (verify_status == Error::Ok) {
      ET_LOG(
          Info,
          "TEST: BundleIO index[%zu] Test_result: PASS",
          kBundleTestsetIdx);
    } else {
      ET_LOG(
          Error,
          "TEST: BundleIO index[%zu] Test_result: FAIL (rtol=%f atol=%f, "
          "status=0x%" PRIx32 ")",
          kBundleTestsetIdx,
          opts.bundle_rtol,
          opts.bundle_atol,
          (uint32_t)verify_status);
      return 1;
    }
  }
#endif // ET_BUNDLE_IO_ENABLED

  return 0;
}
