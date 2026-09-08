/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::kTensorDimensionLimit;
using executorch::runtime::resize_tensor;
using executorch::runtime::tensor_is_default_dim_order;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::check_linear_args;
using torch::executor::Error;
using torch::executor::get_linear_out_target_size;

namespace impl {
namespace HiFi {
namespace native {

namespace {

template <typename CTYPE>
void linear_fallback(
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    Tensor& out,
    int64_t rows,
    int64_t input_features,
    int64_t output_features) {
  const CTYPE* input_data = in.const_data_ptr<CTYPE>();
  const CTYPE* weight_data = weight.const_data_ptr<CTYPE>();
  const CTYPE* bias_data =
      bias.has_value() ? bias->const_data_ptr<CTYPE>() : nullptr;
  CTYPE* output_data = out.mutable_data_ptr<CTYPE>();

  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < output_features; ++column) {
      CTYPE value = bias_data == nullptr
          ? static_cast<CTYPE>(0)
          : bias_data[bias->numel() == 1 ? 0 : column];
      for (int64_t index = 0; index < input_features; ++index) {
        value += input_data[row * input_features + index] *
            weight_data[column * input_features + index];
      }
      output_data[row * output_features + column] = value;
    }
  }
}

} // namespace

Tensor& linear_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx, check_linear_args(in, weight, out), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, weight, out), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_linear_out_target_size(in, weight, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  const int64_t output_features = weight.size(0);
  ET_KERNEL_CHECK(
      ctx,
      !bias.has_value() ||
          (bias->scalar_type() == out.scalar_type() && bias->dim() == 1 &&
           (bias->size(0) == 1 || bias->size(0) == output_features)),
      InvalidArgument,
      out);

  const int64_t input_features = in.size(in.dim() - 1);
  const int64_t rows = in.numel() / input_features;

  if (out.numel() == 0) {
    return out;
  }

  bool optimized = true;
  if (out.scalar_type() != ScalarType::Float) {
    optimized = false;
  }
  if (input_features % 4 != 0) {
    optimized = false;
  }
  // optimized = 0;

  if (optimized) {
    // printf("""Using optimized linear implementation\n");
    const float* bias_data = nullptr;
    if (bias.has_value() && bias->numel() == output_features) {
      bias_data = bias->const_data_ptr<float>();
    } else {
      float* temporary_bias = static_cast<float*>(
          kernels::allocate_temp_memory(ctx, output_features * sizeof(float)));
      if (temporary_bias != nullptr) {
        const float bias_value =
            bias.has_value() ? *bias->const_data_ptr<float>() : 0.0f;
        for (int64_t index = 0; index < output_features; ++index) {
          temporary_bias[index] = bias_value;
        }
        bias_data = temporary_bias;
      }
    }

    if (bias_data != nullptr &&
        xa_nn_matmul_f32xf32_f32(
            out.mutable_data_ptr<float>(),
            in.const_data_ptr<float>(),
            weight.const_data_ptr<float>(),
            bias_data,
            rows,
            input_features,
            input_features,
            output_features,
            input_features,
            1,
            output_features) == 0) {
      return out;
    }
  }

  ET_SWITCH_REAL_TYPES_AND2(
      Half, BFloat16, out.scalar_type(), ctx, "linear.out", CTYPE, [&]() {
        linear_fallback<CTYPE>(
            in, weight, bias, out, rows, input_features, output_features);
      });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
