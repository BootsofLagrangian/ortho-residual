# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# Official Implementation of Orthogonal Residual Updates

from .llama.modeling_ortho_llama import (
    OrthoLlamaConfig,
    OrthoLlamaForCausalLM,
    OrthoLlamaModel,
)
from .vit.modeling_ortho_vit import (
    OrthoViTConfig,
    OrthoViTForImageClassification,
    OrthoViTModel,
)

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification

AutoConfig.register("ortho_llama", OrthoLlamaConfig)  # for mapping config to model
OrthoLlamaConfig.register_for_auto_class()
AutoModel.register(OrthoLlamaConfig, OrthoLlamaModel)
OrthoLlamaModel.register_for_auto_class("AutoModel")
AutoModelForCausalLM.register(OrthoLlamaConfig, OrthoLlamaForCausalLM) # for mapping config to model
OrthoLlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")  # for saving modeling code

AutoConfig.register("ortho_vit", OrthoViTConfig)  # for mapping config to model
OrthoViTConfig.register_for_auto_class()
AutoModel.register(OrthoViTConfig, OrthoViTModel)
OrthoViTModel.register_for_auto_class("AutoModel")
AutoModelForImageClassification.register(OrthoViTConfig, OrthoViTForImageClassification) # for mapping config to model
OrthoViTForImageClassification.register_for_auto_class("AutoModelForImageClassification")  # for saving modeling code

__all__ = [
    "OrthoLlamaConfig",
    "OrthoLlamaForCausalLM",
    "OrthoLlamaModel",
    
    "OrthoViTConfig",
    "OrthoViTForImageClassification",
    "OrthoViTModel",
]
