import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import Attention
from connect import ConnLoggerMixin
from gradient_checkpointing import Unsloth_Offloaded_Gradient_Checkpointer


class OrthoBlock(ConnLoggerMixin, nn.Module):
    """
    OrthoBlock with orthogonal residual connection
    """
    _global_id = 0
    def __init__(
        self, 
        hidden_size, 
        num_heads,
        mlp_ratio=4.0,
        residual_connection="orthogonal",
        orthogonal_method="global",
        residual_eps=1e-6,
        residual_perturbation=None,
        modulate=True,
        mlp_dropout=0.0,
        drop_path=0.0,
        log_interval=50,
        log_activations=True,
        gradient_checkpointing="none", # "none", "torch", "unsloth"
        **block_kwargs
    ):
        nn.Module.__init__(self)
        ConnLoggerMixin.__init__(self, log_interval=log_interval,
                                 log_activations=log_activations)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn.fused_attn = True
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_dropout)

        self.drop_path = (
            nn.Identity() if drop_path == 0.0 else
            DropPath(drop_path)
        )
        self.register_buffer(
            "residual_eps",
            torch.tensor([residual_eps], dtype=torch.float32)
        )

        self.residual_kwargs = {
            "method": residual_connection,
            "orthogonal_method": orthogonal_method,
            "perturbation": residual_perturbation,
        }
        if gradient_checkpointing:
            assert gradient_checkpointing in ("none", "torch", "unsloth"), "gradient_checkpointing should be one of 'none', 'torch', or 'unsloth'"
            self.grad_ckpt = gradient_checkpointing


    def set_step_fn(self, fn):      # 한 번만 호출
        self._get_step = fn

    def perturbation(self):
        return None
    
    def forward(self, x: torch.Tensor):
        if not self.training or self.grad_ckpt == "none":
            return self._forward_impl(x)

        # torch.utils.checkpoint 
        elif self.grad_ckpt == "torch":
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)

        elif self.grad_ckpt == "unsloth":
            def _unsloth_fn(hidden_states):
                return (self._forward_impl(hidden_states),)

            return Unsloth_Offloaded_Gradient_Checkpointer.apply(
                _unsloth_fn, x
            )[0]

    def _forward_impl(self, x: torch.Tensor):
        attn_out = self.drop_path(self.attn(self.norm1(x)))
        x = self._connect_and_collect(
            x, attn_out, 
            tag="attn", eps=self.residual_eps, **self.residual_kwargs
        )
        mlp_out = self.drop_path(self.mlp(self.norm2(x)))
        x = self._connect_and_collect(
            x, mlp_out, 
            tag="mlp", eps=self.residual_eps, **self.residual_kwargs
        )
        return x
