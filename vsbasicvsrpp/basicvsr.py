# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class BasicVSR(BaseModel):
    """BasicVSR model for video super-resolution.

    Note that this model is used for IconVSR.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        generator (dict): Config for the generator structure.
    """

    def __init__(self, generator):
        super().__init__()

        # generator
        self.generator = MODELS.build(generator)

    def forward(self, inputs):
        """Forward tensor. Returns result of simple forward."""

        return self.generator(inputs)
