# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export SmolLM-135M to a Cadence FP32 PTE without quantization."""

import logging
import tempfile
from typing import Any, Optional, Tuple

import torch
from executorch.backends.cadence.aot.compiler import _lower_ep_to_cadence_gen_etrecord
from executorch.backends.cadence.aot.ops_registrations import *  # noqa
from executorch.backends.cadence.aot.utils import save_bpte_program, save_pte_program
from executorch.backends.cadence.runtime.executor import BundledProgramManager
from executorch.exir import ExecutorchProgramManager
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class SmolLMWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.transformer = hf_model.model
        self.lm_head = hf_model.lm_head

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)[0]
        return self.lm_head(hidden_states)


def export_model_fp32(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    file_name: str = "smollm135_fp32",
    working_dir: Optional[str] = None,
) -> ExecutorchProgramManager:
    if working_dir is None:
        working_dir = tempfile.mkdtemp(dir="/tmp")

    ref_outputs = model(*example_inputs)
    exported_program = torch.export.export(model, example_inputs, strict=True)
    exec_prog: ExecutorchProgramManager = _lower_ep_to_cadence_gen_etrecord(
        exported_program, output_dir=working_dir
    )

    forward_test_data = BundledProgramManager.bundled_program_test_data_gen(
        method="forward", inputs=example_inputs, expected_outputs=ref_outputs
    )
    bundled_program_manager = BundledProgramManager([forward_test_data])
    buffer = bundled_program_manager._serialize(
        exec_prog,
        bundled_program_manager.get_method_test_suites(),
        forward_test_data,
    )
    save_pte_program(exec_prog, file_name, working_dir)
    save_bpte_program(buffer, file_name, working_dir)
    logging.info("Exported FP32 model to %s/%s.pte", working_dir, file_name)
    return exec_prog


if __name__ == "__main__":
    torch.manual_seed(42)
    model_name_or_path = "examples/cadence/models/smollm135"
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.return_dict = False
    config.use_cache = False
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float32,
    )
    model = SmolLMWrapper(hf_model).eval()
    example_inputs = (
        torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long),
    )
    export_model_fp32(model, example_inputs)