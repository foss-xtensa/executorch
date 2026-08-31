/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

Tensor& linear_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, check_linear_args(in, weight, out), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, weight, out),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, out.scalar_type() == ScalarType::Float, InvalidArgument, out);

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
          (bias->scalar_type() == ScalarType::Float && bias->dim() == 1 &&
           (bias->size(0) == 1 || bias->size(0) == output_features)),
      InvalidArgument,
      out);

  const int64_t input_features = in.size(in.dim() - 1);
  const int64_t rows = in.numel() / input_features;
  const float* input_data = in.const_data_ptr<float>();
  const float* weight_data = weight.const_data_ptr<float>();
  const float* bias_data = bias.has_value() ? bias->const_data_ptr<float>() : nullptr;
  float* output_data = out.mutable_data_ptr<float>();

  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < output_features; ++column) {
      float value = bias_data == nullptr ? 0.0f : bias_data[bias->numel() == 1 ? 0 : column];
      for (int64_t index = 0; index < input_features; ++index) {
        value += input_data[row * input_features + index] *
            weight_data[column * input_features + index];
      }
      output_data[row * output_features + column] = value;
    }
  }

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
