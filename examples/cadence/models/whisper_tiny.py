import logging
import os

import torch
from examples.cadence.export_test import export_and_test_model
from executorch.backends.cadence.aot.ops_registrations import *  # noqa
from transformers import AutoConfig, WhisperModel

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class WhisperWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_features, decoder_input_ids):
        # Index 0 contains last_hidden_state; bypasses HF dict return to prevent graph breaks
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )[0]


if __name__ == "__main__":
    # Path to weights; loads locally if present, otherwise downloads automatically from Hugging Face
    path_to_weights = "/home/vsaksham/Weights/whisper_tiny"
    model_name_or_path = (
        path_to_weights if os.path.exists(path_to_weights) else "openai/whisper-tiny"
    )

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.return_dict = False
    config.use_cache = False

    hf_model = WhisperModel.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float32,
    )

    model = WhisperWrapper(hf_model).eval()

    # Inputs: (batch_size=1, num_mel_bins=80, sequence_length=3000) and decoder_input_ids
    input_features = torch.randn(1, 80, 3000, dtype=torch.float32)
    decoder_input_ids = torch.tensor(
        [[hf_model.config.decoder_start_token_id]], dtype=torch.long
    )
    example_inputs = (input_features, decoder_input_ids)

    export_and_test_model(model, example_inputs)
