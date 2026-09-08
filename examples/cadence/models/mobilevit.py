import logging
import os

import torch
from examples.cadence.export_test import export_and_test_model
from executorch.backends.cadence.aot.ops_registrations import *  # noqa
from transformers import AutoConfig, AutoModelForImageClassification

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class MobileViTWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        # Index 0 contains raw logits; bypasses HF dict return to prevent graph breaks
        return self.model(x)[0]


if __name__ == "__main__":
    # Path to weights; loads locally if present, otherwise downloads automatically from Hugging Face
    path_to_weights = "/fac/proj_audiopune/dsp3-jenkins/locspace3/Jobs/Executorch_CDNS/weights/mobilevit"
    model_name_or_path = (
        path_to_weights
        if os.path.exists(path_to_weights)
        else "apple/mobilevit-xx-small"
    )

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.return_dict = False

    hf_model = AutoModelForImageClassification.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float32,
    )

    model = MobileViTWrapper(hf_model).eval()

    # Input image (Batch=1, Channels=3, Height=256, Width=256)
    example_inputs = (torch.randn(1, 3, 256, 256, dtype=torch.float32),)

    export_and_test_model(model, example_inputs)
