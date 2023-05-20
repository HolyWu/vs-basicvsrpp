# Copyright (c) OpenMMLab. All rights reserved.
"""Registries and utilities in MMagic.

MMagic provides 17 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

# Neural network modules inheriting `nn.Module`.
MODELS = Registry('model', parent=MMENGINE_MODELS)
