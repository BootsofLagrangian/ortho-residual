"""preactresnet in pytorch
https://github.com/weiaicunzai/pytorch-cifar100

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from connect import ConnLoggerMixin
from gradient_checkpointing import Unsloth_Offloaded_Gradient_Checkpointer

class PreActBasic(ConnLoggerMixin, nn.Module):

    expansion = 1
    def __init__(
        self,
        in_channels, out_channels, stride, 
        gradient_checkpointing="none",
        log_interval=50, log_activations=True, 
        residual_connection="identity", orthogonal_method="global",
        residual_eps=1e-6, residual_perturbation=None,
        residual_pattern="default", residual_rescale_mode="scalar"
    ):
        nn.Module.__init__(self)
        ConnLoggerMixin.__init__(self,
                                 log_interval=log_interval,
                                 log_activations=log_activations)
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride)
        
        self.grad_ckpt = gradient_checkpointing
        if gradient_checkpointing:
            assert gradient_checkpointing in ("none", "torch", "unsloth"), "gradient_checkpointing should be one of 'none', 'torch', or 'unsloth'"
            self.grad_ckpt = gradient_checkpointing

        self.register_buffer("residual_eps", torch.tensor(residual_eps, dtype=torch.float32))
        pattern = (residual_pattern or "default").lower().replace("-", "_")
        rescale_mode = (residual_rescale_mode or "scalar").lower().replace("-", "_")
        self._res_kwargs = dict(method=residual_connection,
                                orthogonal_method=orthogonal_method,
                                perturbation=residual_perturbation,
                                pattern=pattern)
        if pattern == "rescale_stream":
            self._res_kwargs["rescale_mode"] = rescale_mode
        self._init_pattern_state(out_channels * PreActBasic.expansion, pattern, rescale_mode)
    def _forward_impl(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return self._connect_and_collect(shortcut, res, tag="conv", eps=self.residual_eps, **self._res_kwargs)

    def forward(self, x: torch.Tensor):
        if not self.training or self.grad_ckpt == "none":
            return self._forward_impl(x)

        elif self.grad_ckpt == "torch":
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)

        elif self.grad_ckpt == "unsloth":
            def _unsloth_fn(hidden_states):
                return (self._forward_impl(hidden_states),)
    
            return Unsloth_Offloaded_Gradient_Checkpointer.apply(
                _unsloth_fn, x
            )[0]

    def _init_pattern_state(self, channel_dim: int, pattern: str, rescale_mode: str) -> None:
        if pattern == "rezero":
            self._pattern_params["conv_alpha"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rezero_constrained":
            self._pattern_params["conv_theta"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rescale_stream":
            if rescale_mode == "conv1x1":
                proj = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, bias=False)
                nn.init.zeros_(proj.weight)
                with torch.no_grad():
                    for c in range(channel_dim):
                        proj.weight[c, c, 0, 0] = 1.0
                self._pattern_modules["conv_rescale_proj"] = proj
            else:
                self._pattern_params["conv_rescale_alpha"] = nn.Parameter(torch.zeros(1))

class PreActBottleNeck(ConnLoggerMixin, nn.Module):

    expansion = 4
    def __init__(self,
        in_channels, out_channels, stride, 
        gradient_checkpointing="none",
        log_interval=50, log_activations=True, 
        residual_connection="identity", orthogonal_method="global",
        residual_eps=1e-6, residual_perturbation=None,
        residual_pattern="default", residual_rescale_mode="scalar"
    ):
        nn.Module.__init__(self)
        ConnLoggerMixin.__init__(self,
                                 log_interval=log_interval,
                                 log_activations=log_activations)

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

        self.grad_ckpt = gradient_checkpointing
        if gradient_checkpointing:
            assert gradient_checkpointing in ("none", "torch", "unsloth"), "gradient_checkpointing should be one of 'none', 'torch', or 'unsloth'"
            self.grad_ckpt = gradient_checkpointing

        self.register_buffer("residual_eps", torch.tensor(residual_eps, dtype=torch.float32))
        pattern = (residual_pattern or "default").lower().replace("-", "_")
        rescale_mode = (residual_rescale_mode or "scalar").lower().replace("-", "_")
        self._res_kwargs = dict(method=residual_connection,
                                orthogonal_method=orthogonal_method,
                                perturbation=residual_perturbation,
                                pattern=pattern)
        if pattern == "rescale_stream":
            self._res_kwargs["rescale_mode"] = rescale_mode
        channels = out_channels * PreActBottleNeck.expansion
        self._init_pattern_state(channels, pattern, rescale_mode)
    
    def _forward_impl(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return self._connect_and_collect(shortcut, res, tag="conv", eps=self.residual_eps, **self._res_kwargs)

    def forward(self, x: torch.Tensor):
        if not self.training or self.grad_ckpt == "none":
            return self._forward_impl(x)

        elif self.grad_ckpt == "torch":
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)

        elif self.grad_ckpt == "unsloth":
            def _unsloth_fn(hidden_states):
                return (self._forward_impl(hidden_states),)   # ← 튜플 반환
    
            return Unsloth_Offloaded_Gradient_Checkpointer.apply(
                _unsloth_fn, x
            )[0]

    def _init_pattern_state(self, channel_dim: int, pattern: str, rescale_mode: str) -> None:
        if pattern == "rezero":
            self._pattern_params["conv_alpha"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rezero_constrained":
            self._pattern_params["conv_theta"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rescale_stream":
            if rescale_mode == "conv1x1":
                proj = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, bias=False)
                nn.init.zeros_(proj.weight)
                with torch.no_grad():
                    for c in range(channel_dim):
                        proj.weight[c, c, 0, 0] = 1.0
                self._pattern_modules["conv_rescale_proj"] = proj
            else:
                self._pattern_params["conv_rescale_alpha"] = nn.Parameter(torch.zeros(1))

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, **kwargs):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        print("connection method:", kwargs.get("residual_connection", None), "orthogonal", kwargs.get("orthogonal_method", None))

        is_layernorm_classifier = kwargs.pop("is_layernorm_classifier", False)
        

        self.stage1 = self._make_layers(block, num_block[0], 64,  1, **kwargs)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2, **kwargs)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2, **kwargs)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2, **kwargs)

        if is_layernorm_classifier:
            self.linear = nn.Sequential(
                nn.LayerNorm(self.input_channels),
                nn.Linear(self.input_channels, num_classes)
            )
        else:
            self.linear = nn.Linear(self.input_channels, num_classes)
        # self.linear = nn.Linear(self.input_channels, num_classes)

    def _make_layers(self, block, block_num, out_channels, stride, **kwargs):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride, **kwargs))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1, **kwargs))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def pop_stats(self, *, scalarize: bool = True):
        all_stats = []
        for module in self.modules():
            if isinstance(module, ConnLoggerMixin):
                all_stats.extend(module.pop_stats(scalarize=scalarize))
        return all_stats


def preactresnet18(**kwargs):
    return PreActResNet(PreActBasic, [2, 2, 2, 2], **kwargs)

def preactresnet34(**kwargs):
    return PreActResNet(PreActBasic, [3, 4, 6, 3], **kwargs)

def preactresnet50(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3], **kwargs)

def preactresnet101(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3], **kwargs)

def preactresnet152(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3], **kwargs)

PRESET_PREACT_RESNET = {
    "resnet18": preactresnet18,
    "resnet34": preactresnet34,
    "resnet50": preactresnet50,
    "resnet101": preactresnet101,
    "resnet152": preactresnet152
}
