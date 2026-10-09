# ----------------------------------------------------------------------------
# Custom export_model to enable verification of quantization.
# Compare F32 model to Quantized model.
# ============================================================================

# pyre-unsafe

import logging
import tempfile
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from executorch.backends.cadence.aot.compiler import (
    _lower_ep_to_cadence_gen_etrecord,
    apply_pre_edge_transform_passes,
    convert_pt2,
    prepare_pt2,
)
from executorch.backends.cadence.aot.ops_registrations import *
from executorch.backends.cadence.aot.quantizer.quantizer import (
    CadenceDefaultQuantizer,
    get_cadence_default_quantizers,
)
from executorch.backends.cadence.aot.utils import save_bpte_program, save_pte_program
from executorch.backends.cadence.runtime import utils
from executorch.backends.cadence.runtime.executor import BundledProgramManager
from executorch.backends.cadence.runtime.runtime import to_nd_array
from executorch.exir import ExecutorchProgramManager
from tabulate import tabulate
from torch import nn
from torchao.quantization.pt2e.quantizer import Quantizer


class _MultilineLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        orig = record.msg
        if isinstance(orig, str) and "\n" in orig:
            header = f"[{record.levelname} {self.formatTime(record, self.datefmt)} {record.filename}:{record.lineno}]"
            # Strip leading newline if present, then indent/prefix cleanly
            content = orig.lstrip("\n")
            return f"{header}\n{content}"
        return super().format(record)


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
for handler in logging.root.handlers:
    handler.setFormatter(_MultilineLogFormatter(fmt=FORMAT))
warnings.filterwarnings("ignore")
logging.getLogger("executorch.exir.pass_manager").setLevel(logging.ERROR)


def _compute_and_report_quantization_loss(
    fp32_outputs: Any,
    quant_outputs: Any,
) -> None:
    pairs = []

    def _collect_leaf_pairs(fp32: Any, quant: Any) -> None:
        if isinstance(fp32, dict):
            for k in fp32.keys():
                _collect_leaf_pairs(fp32[k], quant[k])
        elif isinstance(fp32, (list, tuple)):
            for f_out, q_out in zip(fp32, quant):
                _collect_leaf_pairs(f_out, q_out)
        else:
            pairs.append((to_nd_array(fp32), to_nd_array(quant)))

    _collect_leaf_pairs(fp32_outputs, quant_outputs)

    rows = []
    has_multiple = len(pairs) > 1

    for idx, (fp32_arr, quant_arr) in enumerate(pairs):
        diff = np.abs(quant_arr - fp32_arr)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))

        denom = np.abs(fp32_arr) + 1e-7
        rel_diff = diff / denom
        mean_rel = float(np.mean(rel_diff))

        fp32_f64 = fp32_arr.astype(np.float64).flatten()
        quant_f64 = quant_arr.astype(np.float64).flatten()
        norm_prod = np.linalg.norm(fp32_f64) * np.linalg.norm(quant_f64)
        cos_sim = (
            float(np.dot(fp32_f64, quant_f64) / norm_prod) if norm_prod != 0 else 1.0
        )

        norm_rms_val = utils.normalized_rms(quant_arr, fp32_arr)

        row = [
            f"{max_abs:.4e}",
            f"{mean_abs:.4e}",
            f"{mean_rel:.4e}",
            f"{cos_sim:.6f}",
            f"{norm_rms_val:.4e}",
            str(tuple(quant_arr.shape)),
        ]
        if has_multiple:
            row.insert(0, str(idx))
        rows.append(row)

    headers = [
        "max_abs",
        "mean_abs",
        "mean_rel",
        "cosine_sim",
        "norm_rms",
        "shape",
    ]
    if has_multiple:
        headers.insert(0, "index")

    metric_descriptions = (
        "Quantization Loss Metrics:\n"
        "  - max_abs:    Maximum absolute point-wise difference max(|quant - fp32|)\n"
        "  - mean_abs:   Mean absolute error mean(|quant - fp32|)\n"
        "  - mean_rel:   Mean relative error mean(|quant - fp32| / (|fp32| + 1e-7))\n"
        "  - cosine_sim: Cosine similarity dot(quant, fp32) / (||quant|| * ||fp32||)\n"
        "  - norm_rms:   Normalized RMS error RMS(quant - fp32) / ||fp32||_2"
    )

    table_str = tabulate(rows, headers=headers, tablefmt="outline")
    logging.info(f"{metric_descriptions}\n\n{table_str}")


def export_and_test_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    file_name: str = "CadenceDemoModel",
    working_dir: Optional[str] = None,
    additional_quantizers: Optional[List[Quantizer]] = None,
) -> ExecutorchProgramManager:
    if working_dir is None:
        working_dir = tempfile.mkdtemp(dir="/tmp")
        logging.debug(f"Created work directory {working_dir}")

    # Native FP32 PyTorch inference
    model.eval()
    with torch.no_grad():
        fp32_outputs = model(*example_inputs)

    # Quantize model
    quantizer = CadenceDefaultQuantizer(
        quantizers=(additional_quantizers or []) + get_cadence_default_quantizers()
    )
    prepared_gm = prepare_pt2(model, example_inputs, quantizer)

    for samples in [example_inputs]:
        prepared_gm(*samples)

    converted_model = convert_pt2(prepared_gm)

    # Quantized reference inference
    with torch.no_grad():
        quant_ref_outputs = converted_model(*example_inputs)

    # Report quantization loss at export time
    _compute_and_report_quantization_loss(fp32_outputs, quant_ref_outputs)

    # Lower and export
    ep = torch.export.export(converted_model, example_inputs, strict=True)
    ep = apply_pre_edge_transform_passes(ep, quantizer)

    exec_prog: ExecutorchProgramManager = _lower_ep_to_cadence_gen_etrecord(
        ep,
        output_dir=working_dir,
    )

    forward_test_data = BundledProgramManager.bundled_program_test_data_gen(
        method="forward", inputs=example_inputs, expected_outputs=quant_ref_outputs
    )
    bundled_program_manager = BundledProgramManager([forward_test_data])
    buffer = bundled_program_manager._serialize(
        exec_prog,
        bundled_program_manager.get_method_test_suites(),
        forward_test_data,
    )

    save_pte_program(exec_prog, file_name, working_dir)
    save_bpte_program(buffer, file_name, working_dir)

    logging.debug(
        f"Executorch bundled program buffer saved to {file_name} is {len(buffer)} total bytes"
    )

    return exec_prog
