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
"""PyTorch OrthoViT model. 
From Revisiting Residual Connections: Orthogonal Residual Connections for Vision Transformers.
preprint arXiv:2505.11881

Authors: Giyeong Oh, Woohyun Cho, Siyeol Kim, Suhwan Choi, Younjae Yu
"""

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, auto_docstring
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTPatchEmbeddings, ViTSelfAttention, ViTSelfOutput, ViTAttention, ViTIntermediate, ViTPooler # Use HF implementations for these

# Assuming connect.py is in the same directory
# from .connect import connect # For use as a package
# For standalone script, you might need to adjust import path or ensure connect.py is findable
logger = logging.get_logger(__name__)
try:
    from .residual import connect
except ImportError:
    logger.warning("residual.py not found, using fallback linear connect function.")
    def connect(x, f_x, *args, **kwargs): # Linear connection fallback
        """
        Fallback connection function if connect.py is not available.
        This will simply add the input x to the function output f_x.
        This is a simple linear connection, not orthogonal.
        """
        return x + f_x 

from .configuration_ortho_vit import OrthoViTConfig # Changed from ViTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OrthoViTConfig"

# Copied from transformers.models.vit.modeling_vit.ViTOutput
# with the residual connection modified to use the `connect` function.
class OrthoViTOutput(nn.Module):
    def __init__(self, config: OrthoViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Orthogonal connection parameters are stored in config and passed to connect
        self.config = config
        # Register buffer for eps, ensuring it's on the correct device and non-persistent
        self.register_buffer("residual_eps", torch.tensor([config.residual_eps], dtype=torch.float32), persistent=False)


    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, residual_input: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Use the connect function for the residual connection
        # The 'input_tensor' to ViTOutput is the output of the intermediate layer.
        # The 'residual_input' is the output of the attention block (hidden_states before layernorm_after).
        hidden_states = connect(
            x=residual_input,
            f_x=hidden_states,
            eps=self.residual_eps,
            **self.config.residual_kwargs
        )
        return hidden_states

class OrthoViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: OrthoViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config) # Standard HF ViTAttention
        self.intermediate = ViTIntermediate(config) # Standard HF ViTIntermediate
        # self.output = ViTOutput(config) # Standard HF ViTOutput
        self.output = OrthoViTOutput(config) # Use OrthoViTOutput for the second residual connection

        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=config.elementwise_affine_ln)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=config.elementwise_affine_ln)
        self.config = config
        # Register buffer for eps, ensuring it's on the correct device and non-persistent
        self.register_buffer("residual_eps", torch.tensor([config.residual_eps], dtype=torch.float32), persistent=False)
        self.residual_kwargs = self.config.residual_kwargs # Use the config's residual_kwargs


    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        # Input to Attention block
        normed_hidden_states = self.layernorm_before(hidden_states)
        self_attention_outputs = self.attention(
            normed_hidden_states,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0] # Output of ViTAttention (includes SelfAttention + SelfOutput)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # First residual connection (after Attention)
        # The `connect` function expects the module output (f_x) and the stream input (x)
        # Here, hidden_states is x, and attention_output is f_x from the attention block
        hidden_states = connect(
            x=hidden_states, 
            f_x=attention_output,
            eps=self.residual_eps, # Use the buffer
            **self.residual_kwargs
        )
        
        # MLP block
        # In ViT, layernorm is also applied after self-attention (and first residual)
        # This `hidden_states` is now `x_{n+1}` from the attention block
        mlp_input = self.layernorm_after(hidden_states)
        intermediate_output = self.intermediate(mlp_input)

        # Second residual connection (after MLP) is handled by OrthoViTOutput
        # `hidden_states` is the residual stream input to the MLP's residual connection
        # `intermediate_output` is the direct output of the MLP's main path (before adding to residual)
        layer_output = self.output(intermediate_output, mlp_input, hidden_states) # mlp_input here is just for consistency with ViTOutput's original signature, OrthoViTOutput will use residual_input

        outputs = (layer_output,) + outputs

        return outputs


class OrthoViTEncoder(nn.Module):
    def __init__(self, config: OrthoViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([OrthoViTLayer(config) for _ in range(config.num_hidden_layers)]) # Use OrthoViTLayer
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

# @add_start_docstrings(
#     "The bare ViT MAE transformer outputting raw hidden-states without any specific head on top.",
#     # VIT_START_DOCSTRING, # OrthoViT is custom
# )
# @auto_docstring
class OrthoViTPreTrainedModel(PreTrainedModel): # Copied from ViTPreTrainedModel
    config_class = OrthoViTConfig # Use OrthoViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTEmbeddings", "OrthoViTLayer"] # Use OrthoViTLayer
    _supports_sdpa = True # Inherit from ViT
    _supports_flash_attn_2 = True # Inherit from ViT


    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine: # Only init if affine
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings): # Standard ViTEmbeddings init
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`OrthoViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ViTImageProcessor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the positional encoding if the input image resolution is different from the one
            used during pre-training.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# @add_start_docstrings(
#     "The bare OrthoViT transformer outputting raw hidden-states without any specific head on top.",
#     VIT_START_DOCSTRING,
# )
# @auto_docstring
class OrthoViTModel(OrthoViTPreTrainedModel):
    def __init__(self, config: OrthoViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = OrthoViTEncoder(config) # Use OrthoViTEncoder

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # No affine
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    # @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None, # For MAE
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output) # Final LayerNorm
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# @auto_docstring(
    # """
    # OrthoViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    # the [CLS] token) e.g. for ImageNet.
    # """
# )
class OrthoViTForImageClassification(OrthoViTPreTrainedModel):
    def __init__(self, config: OrthoViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = OrthoViTModel(config, add_pooling_layer=False) # Use OrthoViTModel

        # Classifier head (standard HF: just a Linear layer)
        # The author's original classifier was nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        # The OrthoViTModel's output (CLS token) is already layernormed by self.vit.layernorm.
        # So, if that self.vit.layernorm corresponds to the author's classifier.0 (LayerNorm),
        # then this classifier head should just be Linear.
        # If an *additional* LN is needed on the CLS token, this needs to be nn.Sequential.
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    # @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # Takes the CLS token representation for classification
        logits = self.classifier(sequence_output[:, 0, :])


        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = ["OrthoViTConfig", "OrthoViTModel", "OrthoViTForImageClassification", "OrthoViTPreTrainedModel"]