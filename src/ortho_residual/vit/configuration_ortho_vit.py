# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 The Authors of Revisiting Residual Connections. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OrthoViT model configuration"""

from transformers.models.vit.configuration_vit import ViTConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class OrthoViTConfig(ViTConfig):
    r"""
    This is the configuration class to store the configuration of an [`OrthoViTModel`].
    It is used to instantiate a ViT model with orthogonal residual connections
    according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    the ViT-base model.

    Configuration objects inherit from [`ViTConfig`] and can be used to control the model outputs.
    Read the documentation from [`ViTConfig`] for more information.

    Args:
        residual_connection (`str`, *optional*, defaults to `"linear"`):
            The type of residual connection to use. Can be "linear" or "orthogonal".
        orthogonal_method (`str`, *optional*, defaults to `"channel"`):
            The method for orthogonalization if `residual_connection` is "orthogonal".
            Can be "channel" or "global". For ViT, "channel" is typically used for token embeddings.
        residual_connection_dim (`int`, *optional*, defaults to -1):
            The dimension along which to compute orthogonality. Defaults to -1 (last dimension).
        residual_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon value for numerical stability in orthogonalization.
        residual_perturbation (`float`, *optional*, defaults to `None`):
            Magnitude of random perturbation to add to the module output before connection.

    Example:

    ```python
    >>> from modeling_ortho_vit import OrthoViTModel
    >>> from configuration_ortho_vit import OrthoViTConfig

    >>> # Initializing a ViT-base style configuration with orthogonal connections
    >>> configuration = OrthoViTConfig(residual_connection="orthogonal")

    >>> # Initializing a model (with random weights) from the ViT-base style configuration
    >>> model = OrthoViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "ortho_vit"

    def __init__(
        self,
        residual_connection="linear",
        orthogonal_method="channel", # For ViT, 'channel' typically means along the hidden_dim
        residual_connection_dim=-1,
        residual_eps=1e-6,
        residual_perturbation=None,
        elementwise_affine_ln=False, # To match user's OrthoBlock norm1/norm2
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.residual_connection = residual_connection
        self.orthogonal_method = orthogonal_method
        self.residual_connection_dim = residual_connection_dim
        self.residual_eps = residual_eps
        self.residual_perturbation = residual_perturbation
        self.elementwise_affine_ln = elementwise_affine_ln # For ViTLayer norms

    @property
    def residual_kwargs(self) -> dict:
        # This property can be used by the model to easily access all residual connection parameters
        return dict(
            method=self.residual_connection,
            orthogonal_method=self.orthogonal_method,
            dim=self.residual_connection_dim,
            perturbation=self.residual_perturbation,
            # eps is handled via a registered buffer in the layer
        )