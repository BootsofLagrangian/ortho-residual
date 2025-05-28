# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Callable, Optional, Tuple, Union

import torch
# import torch.utils.checkpoint
from torch import nn

# from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
# from transformers.generation import GenerationMixin
# from transformers.integrations import use_kernel_forward_from_hub
# from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
# from transformers.modeling_layers import GradientCheckpointingLayer
# from transformers.modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
#     QuestionAnsweringModelOutput,
#     SequenceClassifierOutputWithPast,
#     TokenClassifierOutput,
# )
# from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
# from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, logging

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    # LlamaConfig,
    LlamaMLP,
)

from .configuration_ortho_llama import OrthoLlamaConfig
from .residual import connect


logger = logging.get_logger(__name__)

class OrthoLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OrthoLlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        # self.hidden_size = config.hidden_size

        # self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # self.mlp = LlamaMLP(config)
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.register_buffer("residual_eps", torch.tensor([config.residual_eps], dtype=torch.float32), persistent=False)

        self._res_kwargs = config.residual_kwargs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = (
            residual + hidden_states
            if self._res_kwargs["method"] == "linear"
            else connect(residual, hidden_states, eps=self.residual_eps, **self._res_kwargs)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = (
            residual + hidden_states
            if self._res_kwargs["method"] == "linear"
            else connect(residual, hidden_states, eps=self.residual_eps, **self._res_kwargs)
        )

        # hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class OrthoLlamaPreTrainedModel(LlamaPreTrainedModel):
    config_class = OrthoLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OrthoLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True


class OrthoLlamaModel(OrthoLlamaPreTrainedModel, LlamaModel):
    config_class = OrthoLlamaConfig
    def __init__(self, config: OrthoLlamaConfig):
        super().__init__(config)
        
        for idx in range(len(self.layers)):
            self.layers[idx] = OrthoLlamaDecoderLayer(config, idx)


class OrthoLlamaForCausalLM(LlamaForCausalLM):
    config_class = OrthoLlamaConfig
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = OrthoLlamaModel(config)
        self.post_init()


__all__ = [
    "OrthoLlamaForCausalLM",
    "OrthoLlamaModel",
    "OrthoLlamaPreTrainedModel",
    # "LlamaForSequenceClassification",
    # "LlamaForQuestionAnswering",
    # "LlamaForTokenClassification",
]
