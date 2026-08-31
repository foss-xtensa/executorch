/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/backends/cadence/hifi/operators/operators.h>
#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include <tuple>

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::getLeadingDims;
using ::executorch::runtime::getTrailingDims;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::kTensorDimensionLimit;
using ::executorch::runtime::resize_tensor;
using ::executorch::runtime::tensor_is_default_dim_order;
using ::executorch::runtime::tensors_have_same_dim_order;
using ::torch::executor::check_layer_norm_args;
using ::torch::executor::Error;
using ::torch::executor::get_layer_norm_out_target_size;
using ::torch::executor::layer_norm_scalar;

namespace impl {
namespace HiFi {
namespace native {

namespace {

template <typename CTYPE>
void layer_norm_portable(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    CTYPE eps,
    Tensor& out,
    Tensor& mean,
    Tensor& rstd) {
  size_t dim = input.dim() - normalized_shape.size();
  size_t dim_size = input.size(dim);

  size_t leading = getLeadingDims(input, dim);
  size_t normalized = getTrailingDims(input, dim) * dim_size;

  if (leading == 0) {
    return;
  }

  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  CTYPE* mean_data = mean.mutable_data_ptr<CTYPE>();
  CTYPE* rstd_data = rstd.mutable_data_ptr<CTYPE>();

  if (normalized == 0) {
    for (const auto i : c10::irange(leading)) {
      mean_data[i] = static_cast<CTYPE>(0);
      rstd_data[i] = static_cast<CTYPE>(NAN);
    }
    return;
  }

  const CTYPE* input_data = input.const_data_ptr<CTYPE>();
  const CTYPE* weight_data =
      weight.has_value() ? weight.value().const_data_ptr<CTYPE>() : nullptr;
  const CTYPE* bias_data =
      bias.has_value() ? bias.value().const_data_ptr<CTYPE>() : nullptr;

  layer_norm_scalar<CTYPE>(
      input_data,
      weight_data,
      bias_data,
      out_data,
      mean_data,
      rstd_data,
      leading,
      normalized,
      eps);
}

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> native_layer_norm_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, rstd_out);

  ET_KERNEL_CHECK(
      ctx,
      check_layer_norm_args(
          input, normalized_shape, weight, bias, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx, tensor_is_default_dim_order(input), InvalidArgument, ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(input, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(input, weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (bias.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(input, bias.value()),
        InvalidArgument,
        ret_val);
  }

  Tensor::SizesType mean_rstd_sizes[kTensorDimensionLimit];
  size_t mean_rstd_ndim = 0;
  get_layer_norm_out_target_size(
      input, normalized_shape, mean_rstd_sizes, &mean_rstd_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(mean_out, {mean_rstd_sizes, mean_rstd_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(rstd_out, {mean_rstd_sizes, mean_rstd_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  size_t dim = input.dim() - normalized_shape.size();
  size_t dim_size = input.size(dim);
  size_t leading = getLeadingDims(input, dim);
  size_t normalized = getTrailingDims(input, dim) * dim_size;

  if (input.scalar_type() == ScalarType::Float && leading > 0 &&
      normalized > 0) {
    const float* input_data = input.const_data_ptr<float>();
    const float* weight_data =
        weight.has_value() ? weight.value().const_data_ptr<float>() : nullptr;
    const float* bias_data =
        bias.has_value() ? bias.value().const_data_ptr<float>() : nullptr;
    float* out_data = out.mutable_data_ptr<float>();
    float* mean_data = mean_out.mutable_data_ptr<float>();
    float* rstd_data = rstd_out.mutable_data_ptr<float>();

    WORD32 status = xa_nn_layer_norm_f32(
        out_data,
        mean_data,
        rstd_data,
        input_data,
        weight_data,
        bias_data,
        (WORD32)leading,
        (WORD32)normalized,
        (FLOAT32)eps);
    ET_KERNEL_CHECK(ctx, status == 0, Internal, ret_val);
    return ret_val;
  }

  // Fallback path
  ET_SWITCH_FLOATHBF16_TYPES(
      input.scalar_type(), ctx, "native_layer_norm.out", CTYPE, [&]() {
        layer_norm_portable<CTYPE>(
            input,
            normalized_shape,
            weight,
            bias,
            eps,
            out,
            mean_out,
            rstd_out);
      });

  return ret_val;
}

} // namespace native
} // namespace HiFi
} // namespace impl
