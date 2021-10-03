# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from .builder import build_backbone
from .registry import MODELS


@MODELS.register_module()
class BasicVSR(nn.Module):
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
        self.generator = build_backbone(generator)

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

    def forward(self, lq):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).

        Returns:
            Tensor: Output result.
        """
        return self.generator(lq)
