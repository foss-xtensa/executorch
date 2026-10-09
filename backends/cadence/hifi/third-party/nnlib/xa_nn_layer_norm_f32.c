/*******************************************************************************
 * Copyright (c) 2018-2024 Cadence Design Systems, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to use this Software with Cadence processor cores only and
 * not with any other processors and platforms, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

#include <math.h>
#include <stddef.h>
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_err_chk.h"
#include "xa_type_def.h"

#if (XCHAL_HAVE_HIFI5_VFPU)
#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)
#else
#define SW_MOVDA32(a) AE_MOVDA32(a)
#endif

/*-------------------------------------------------------------------------
  Layer Normalization (FP32)
  Computes Layer Normalization over the last dimension (size N) across M rows:
    y = ( (x - mean) / sqrt(var + eps) ) * weight + bias

  Input:
    p_inp     Input tensor of size M * N (FLOAT32)
    p_weight  Scale weights of size N (FLOAT32, optional)
    p_bias    Bias offsets of size N (FLOAT32, optional)
    M         Number of rows/batches
    N         Normalized feature dimension size
    eps       Epsilon value for numerical stability
  Output:
    p_out     Output normalized tensor of size M * N (FLOAT32)
    p_mean    Mean per row of size M (FLOAT32)
    p_rstd    Reciprocal standard deviation per row of size M (FLOAT32)

  Returns:
    0 on success, -1 on invalid argument
-------------------------------------------------------------------------*/

static const union ufloat32uint32 xa_nnlib_qNaNf_local = {0x7fc00000};

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
    WORD32,
    xa_nn_layer_norm_f32,
    (FLOAT32* __restrict__ p_out,
     FLOAT32* __restrict__ p_mean,
     FLOAT32* __restrict__ p_rstd,
     const FLOAT32* __restrict__ p_inp,
     const FLOAT32* __restrict__ p_weight,
     const FLOAT32* __restrict__ p_bias,
     WORD32 M,
     WORD32 N,
     FLOAT32 eps))
#else
WORD32 xa_nn_layer_norm_f32(
    FLOAT32* __restrict__ p_out,
    FLOAT32* __restrict__ p_mean,
    FLOAT32* __restrict__ p_rstd,
    const FLOAT32* __restrict__ p_inp,
    const FLOAT32* __restrict__ p_weight,
    const FLOAT32* __restrict__ p_bias,
    WORD32 M,
    WORD32 N,
    FLOAT32 eps) {
  /* Basic parameter checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mean, -1);
  XA_NNLIB_ARG_CHK_PTR(p_rstd, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);

  XA_NNLIB_ARG_CHK_COND((M <= 0), 0);

  /* Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mean, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_rstd, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  if (p_weight != NULL) {
    XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(FLOAT32), -1);
  }
  if (p_bias != NULL) {
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  }

  /* Degenerate case: N <= 0 */
  if (N <= 0) {
    for (WORD32 i = 0; i < M; ++i) {
      p_mean[i] = 0.0f;
      p_rstd[i] = xa_nnlib_qNaNf_local.f;
    }
    return 0;
  }

  /* Precalculate N as float and scalar epsilon */
  xtfloat N_flt = (xtfloat)(FLOAT32)N;
  xtfloat eps_s = (xtfloat)eps;

  for (WORD32 i = 0; i < M; ++i) {
    const FLOAT32* p_row_inp = p_inp + (i * N);
    FLOAT32* p_row_out = p_out + (i * N);

    /* Pass 1: Reduction for Mean and Variance */
    xtfloatx2* pt_inp = (xtfloatx2*)p_row_inp;
    ae_valign inp_a = XT_LASX2PP(pt_inp);

    xtfloatx2 sumx2 = XT_CONST_S(0);
    xtfloatx2 sq_sumx2 = XT_CONST_S(0);

#pragma concurrent
    for (WORD32 j = 0; j < (N >> 1); ++j) {
      xtfloatx2 d;
      XT_LASX2IP(d, inp_a, pt_inp);
      sumx2 = XT_ADD_SX2(sumx2, d);
      XT_MADD_SX2(sq_sumx2, d, d);
    }

    /* Cross-lane horizontal reduction */
    sumx2 = XT_ADD_SX2(sumx2, XT_SEL32_LH_SX2(sumx2, sumx2));
    sq_sumx2 = XT_ADD_SX2(sq_sumx2, XT_SEL32_LH_SX2(sq_sumx2, sq_sumx2));

    xtfloat sum = XT_LOW_S(sumx2);
    xtfloat sq_sum = XT_LOW_S(sq_sumx2);

    /* Remainder reduction for odd tail */
    if (N & 1) {
      xtfloat d_inp = ((const xtfloat*)p_row_inp)[N - 1];
      sum = XT_ADD_S(sum, d_inp);
      XT_MADD_S(sq_sum, d_inp, d_inp);
    }

    /* Compute mean, variance, and reciprocal standard deviation */
    xtfloat mean = XT_DIV_S(sum, N_flt);
    xtfloat variance;
    if (N == 1) {
      variance = XT_CONST_S(0);
    } else {
      xtfloat mean_sq = XT_DIV_S(sq_sum, N_flt);
      variance = mean_sq;
      XT_MSUB_S(variance, mean, mean);
      variance = XT_MAX_S(variance, XT_CONST_S(0));
    }
    xtfloat rstd =
        XT_DIV_S(XT_CONST_S(1), XT_SQRT_S(XT_ADD_S(variance, eps_s)));

    p_mean[i] = (FLOAT32)mean;
    p_rstd[i] = (FLOAT32)rstd;

    /* Pass 2: Normalization and Affine Transformation */
    xtfloat scale_s = rstd;
    xtfloat offset_s = XT_NEG_S(XT_MUL_S(rstd, mean));
    xtfloatx2 scale_x2 =
        XT_AE_MOVXTFLOATX2_FROMINT32X2(SW_MOVDA32(XT_RFR(scale_s)));
    xtfloatx2 offset_x2 =
        XT_AE_MOVXTFLOATX2_FROMINT32X2(SW_MOVDA32(XT_RFR(offset_s)));

    pt_inp = (xtfloatx2*)p_row_inp;
    inp_a = XT_LASX2PP(pt_inp);
    xtfloatx2* pt_out = (xtfloatx2*)p_row_out;
    ae_valign out_a = AE_ZALIGN64();

    if (p_weight != NULL && p_bias != NULL) {
      xtfloatx2* pt_w = (xtfloatx2*)p_weight;
      xtfloatx2* pt_b = (xtfloatx2*)p_bias;
      ae_valign w_a = XT_LASX2PP(pt_w);
      ae_valign b_a = XT_LASX2PP(pt_b);

#pragma concurrent
      for (WORD32 j = 0; j < (N >> 1); ++j) {
        xtfloatx2 d_inpx2, d_wx2, d_bx2, eff_w, eff_b, d_outx2;

        XT_LASX2IP(d_inpx2, inp_a, pt_inp);
        XT_LASX2IP(d_wx2, w_a, pt_w);
        XT_LASX2IP(d_bx2, b_a, pt_b);

        eff_w = XT_MUL_SX2(scale_x2, d_wx2);
        eff_b = d_bx2;
        XT_MADD_SX2(eff_b, offset_x2, d_wx2);

        d_outx2 = eff_b;
        XT_MADD_SX2(d_outx2, d_inpx2, eff_w);

        XT_SASX2IP(d_outx2, out_a, pt_out);
      }
      XT_SASX2POSFP(out_a, pt_out);

      if (N & 1) {
        xtfloat d_inp = ((const xtfloat*)p_row_inp)[N - 1];
        xtfloat d_w = ((const xtfloat*)p_weight)[N - 1];
        xtfloat d_b = ((const xtfloat*)p_bias)[N - 1];
        xtfloat eff_w_s = XT_MUL_S(scale_s, d_w);
        xtfloat eff_b_s = XT_ADD_S(XT_MUL_S(offset_s, d_w), d_b);
        xtfloat d_out = XT_ADD_S(XT_MUL_S(d_inp, eff_w_s), eff_b_s);
        ((xtfloat*)p_row_out)[N - 1] = d_out;
      }
    } else if (p_weight != NULL) {
      xtfloatx2* pt_w = (xtfloatx2*)p_weight;
      ae_valign w_a = XT_LASX2PP(pt_w);

#pragma concurrent
      for (WORD32 j = 0; j < (N >> 1); ++j) {
        xtfloatx2 d_inpx2, d_wx2, eff_w, eff_b, d_outx2;

        XT_LASX2IP(d_inpx2, inp_a, pt_inp);
        XT_LASX2IP(d_wx2, w_a, pt_w);

        eff_w = XT_MUL_SX2(scale_x2, d_wx2);
        eff_b = XT_MUL_SX2(offset_x2, d_wx2);

        d_outx2 = eff_b;
        XT_MADD_SX2(d_outx2, d_inpx2, eff_w);

        XT_SASX2IP(d_outx2, out_a, pt_out);
      }
      XT_SASX2POSFP(out_a, pt_out);

      if (N & 1) {
        xtfloat d_inp = ((const xtfloat*)p_row_inp)[N - 1];
        xtfloat d_w = ((const xtfloat*)p_weight)[N - 1];
        xtfloat eff_w_s = XT_MUL_S(scale_s, d_w);
        xtfloat eff_b_s = XT_MUL_S(offset_s, d_w);
        xtfloat d_out = XT_ADD_S(XT_MUL_S(d_inp, eff_w_s), eff_b_s);
        ((xtfloat*)p_row_out)[N - 1] = d_out;
      }
    } else if (p_bias != NULL) {
      xtfloatx2* pt_b = (xtfloatx2*)p_bias;
      ae_valign b_a = XT_LASX2PP(pt_b);

#pragma concurrent
      for (WORD32 j = 0; j < (N >> 1); ++j) {
        xtfloatx2 d_inpx2, d_bx2, eff_b, d_outx2;

        XT_LASX2IP(d_inpx2, inp_a, pt_inp);
        XT_LASX2IP(d_bx2, b_a, pt_b);

        eff_b = XT_ADD_SX2(offset_x2, d_bx2);

        d_outx2 = eff_b;
        XT_MADD_SX2(d_outx2, d_inpx2, scale_x2);

        XT_SASX2IP(d_outx2, out_a, pt_out);
      }
      XT_SASX2POSFP(out_a, pt_out);

      if (N & 1) {
        xtfloat d_inp = ((const xtfloat*)p_row_inp)[N - 1];
        xtfloat d_b = ((const xtfloat*)p_bias)[N - 1];
        xtfloat eff_b_s = XT_ADD_S(offset_s, d_b);
        xtfloat d_out = XT_ADD_S(XT_MUL_S(d_inp, scale_s), eff_b_s);
        ((xtfloat*)p_row_out)[N - 1] = d_out;
      }
    } else {
      /* p_weight == NULL, p_bias == NULL */
#pragma concurrent
      for (WORD32 j = 0; j < (N >> 1); ++j) {
        xtfloatx2 d_inpx2, d_outx2;

        XT_LASX2IP(d_inpx2, inp_a, pt_inp);

        d_outx2 = offset_x2;
        XT_MADD_SX2(d_outx2, d_inpx2, scale_x2);

        XT_SASX2IP(d_outx2, out_a, pt_out);
      }
      XT_SASX2POSFP(out_a, pt_out);

      if (N & 1) {
        xtfloat d_inp = ((const xtfloat*)p_row_inp)[N - 1];
        xtfloat d_out = XT_ADD_S(XT_MUL_S(d_inp, scale_s), offset_s);
        ((xtfloat*)p_row_out)[N - 1] = d_out;
      }
    }
  }

  return 0;
}
#endif
