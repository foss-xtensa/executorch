import logging

import torch

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

from examples.cadence.export_test import export_and_test_model
from ultralytics import YOLO


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class YOLOv5Wrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


if __name__ == "__main__":
    yolo = YOLO("examples/cadence/models/weights/yolov5nu.pt")
    pt_model = YOLOv5Wrapper(yolo.model.eval())
    example_inputs = (torch.randn(1, 3, 640, 640),)

    export_and_test_model(pt_model, example_inputs)
