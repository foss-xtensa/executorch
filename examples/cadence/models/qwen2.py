import logging
import os

import torch
from examples.cadence.export_test import export_and_test_model
from executorch.backends.cadence.aot.ops_registrations import *  # noqa
from transformers import AutoConfig, AutoModelForCausalLM

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class Qwen2Wrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.transformer = hf_model.model
        self.lm_head = hf_model.lm_head

    def forward(self, input_ids):
        # Index 0 contains raw hidden states; bypasses HF dict return to prevent graph breaks
        hidden_states = self.transformer(input_ids)[0]
        return self.lm_head(hidden_states)


if __name__ == "__main__":
    # Path to weights; loads locally if present, otherwise downloads automatically from Hugging Face
    path_to_weights = (
        "examples/cadence/models/qwen2"
    )
    model_name_or_path = (
        path_to_weights if os.path.exists(path_to_weights) else "Qwen/Qwen2-0.5B"
    )

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.return_dict = False
    config.use_cache = False

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float32,
    )

    model = Qwen2Wrapper(hf_model).eval()

    # Input sequence (Batch=1, SeqLen=32)
    example_inputs = (torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long),)

    export_and_test_model(model, example_inputs)
