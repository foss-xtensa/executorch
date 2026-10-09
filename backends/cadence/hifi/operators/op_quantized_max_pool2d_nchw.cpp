/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <xa_nnlib_kernels_api.h>

#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

Tensor& quantized_max_pool2d_nchw_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output) {
  // NCHW layout: [N, C, H, W]
  const int32_t batch_size = input.size(0);
  const int32_t channels = input.size(1);
  const int32_t in_height = input.size(2);
  const int32_t in_width = input.size(3);

  const int32_t out_height = output.size(2);
  const int32_t out_width = output.size(3);

  const int32_t kernel_h = kernel_size[0];
  const int32_t kernel_w = kernel_size[1];
  const int32_t stride_h = stride[0];
  const int32_t stride_w = stride[1];
  const int32_t pad_h = padding[0];
  const int32_t pad_w = padding[1];

  ScalarType dtype = input.scalar_type();
  int32_t nnlib_precision;
  switch (dtype) {
    case ScalarType::Char: // int8
      nnlib_precision = PREC_SYM8S;
      break;
    case ScalarType::Byte: // uint8
      nnlib_precision = PREC_ASYM8U;
      break;
    default:
      ET_DCHECK_MSG(
          false,
          "Unsupported dtype %s for HiFi quantized_max_pool2d_nchw",
          torch::executor::toString(dtype));
      return output;
  }

  constexpr int32_t kNnlibMaxDim = 4;

  // Allocate aligned NHWC input buffer [N, H, W, C]
  const int32_t in_elems = batch_size * in_height * in_width * channels;
  void* ptr_nhwc_in_raw =
      kernels::allocate_temp_memory(ctx, (in_elems + 8) * sizeof(int8_t));
  void* p_nhwc_in = (void*)ALIGN_PTR(ptr_nhwc_in_raw, 8);

  // Allocate aligned NHWC output buffer [N, H_out, W_out, C]
  const int32_t out_elems = batch_size * out_height * out_width * channels;
  void* ptr_nhwc_out_raw =
      kernels::allocate_temp_memory(ctx, (out_elems + 8) * sizeof(int8_t));
  void* p_nhwc_out = (void*)ALIGN_PTR(ptr_nhwc_out_raw, 8);

  // Transpose input: NCHW [N,C,H,W] -> NHWC [N,H,W,C]
  WORD32 inp_shape[kNnlibMaxDim] = {batch_size, channels, in_height, in_width};
  WORD32 nhwc_in_shape[kNnlibMaxDim] = {
      batch_size, in_height, in_width, channels};
  WORD32 nchw_to_nhwc[kNnlibMaxDim] = {0, 2, 3, 1};
  xa_nn_transpose_8_8(
      (WORD8*)p_nhwc_in,
      nhwc_in_shape,
      (const WORD8*)input.const_data_ptr(),
      inp_shape,
      nchw_to_nhwc,
      kNnlibMaxDim,
      kNnlibMaxDim);

  // Compute scratch buffer size for NNLIB maxpool
  int32_t scratch_size = xa_nn_maxpool_getsize(
      channels,
      nnlib_precision,
      nnlib_precision,
      in_height,
      in_width,
      kernel_h,
      kernel_w,
      stride_w, // x_stride
      stride_h, // y_stride
      pad_w, // x_padding
      pad_h, // y_padding
      out_height,
      out_width,
      0, // inp_data_format: NHWC
      0); // out_data_format: NHWC
  scratch_size = scratch_size < 0 ? 0 : scratch_size;

  void* ptr_scratch_raw = kernels::allocate_temp_memory(ctx, scratch_size + 8);
  void* p_scratch = (void*)ALIGN_PTR(ptr_scratch_raw, 8);

  // Process each batch using NNLIB optimized maxpool kernel
  for (int32_t n = 0; n < batch_size; ++n) {
    const int32_t in_spatial = in_height * in_width * channels;
    const int32_t out_spatial = out_height * out_width * channels;

    int32_t ret;
    if (dtype == ScalarType::Char) {
      ret = xa_nn_maxpool_8(
          (WORD8*)p_nhwc_out + n * out_spatial,
          (const WORD8*)p_nhwc_in + n * in_spatial,
          in_height,
          in_width,
          channels,
          kernel_h,
          kernel_w,
          stride_w, // x_stride
          stride_h, // y_stride
          pad_w, // x_padding
          pad_h, // y_padding
          out_height,
          out_width,
          0, // inp_data_format: NHWC
          0, // out_data_format: NHWC
          p_scratch);
    } else {
      ret = xa_nn_maxpool_asym8(
          (UWORD8*)p_nhwc_out + n * out_spatial,
          (const UWORD8*)p_nhwc_in + n * in_spatial,
          in_height,
          in_width,
          channels,
          kernel_h,
          kernel_w,
          stride_w, // x_stride
          stride_h, // y_stride
          pad_w, // x_padding
          pad_h, // y_padding
          out_height,
          out_width,
          0, // inp_data_format: NHWC
          0, // out_data_format: NHWC
          p_scratch);
    }
    ET_DCHECK_MSG(ret == 0, "HiFi xa_nn_maxpool failed");
  }

  // Transpose output: NHWC [N,H_out,W_out,C] -> NCHW [N,C,H_out,W_out]
  WORD32 nhwc_out_shape[kNnlibMaxDim] = {
      batch_size, out_height, out_width, channels};
  WORD32 nchw_out_shape[kNnlibMaxDim] = {
      batch_size, channels, out_height, out_width};
  WORD32 nhwc_to_nchw[kNnlibMaxDim] = {0, 3, 1, 2};
  xa_nn_transpose_8_8(
      (WORD8*)output.mutable_data_ptr(),
      nchw_out_shape,
      (const WORD8*)p_nhwc_out,
      nhwc_out_shape,
      nhwc_to_nchw,
      kNnlibMaxDim,
      kNnlibMaxDim);

  return output;
}

} // namespace native
} // namespace HiFi
} // namespace impl
