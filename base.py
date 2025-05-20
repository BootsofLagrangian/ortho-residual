import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from timm.layers.mlp import Mlp


class Classifier(nn.Module):
    def __init__(
        self, 
        img_size=28,
        dim=128,
        patch_size=7,
        num_heads=4,
        num_layers=4, 
        in_chans=3, 
        num_classes=10,
        class_token=False,
        reg_tokens=0,
        pos_embed="learn",
        **block_kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.dim = dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        
        # create patch embedding
        self.patchify = nn.Conv2d(
            in_channels=in_chans,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        block_class = block_kwargs.pop("block_class", TransformerBlock)
        print(f"Using block class: {block_class}, connection method:", block_kwargs.get("residual_connection", None))
        assert block_class is not None, "block_class must be provided"
        # create N blocks
        self.blocks = nn.ModuleList([
            block_class(
                hidden_size=dim,
                num_heads=num_heads,
                **block_kwargs
            ) for _ in range(num_layers)
        ])
        # final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.num_reg_tokens = reg_tokens if reg_tokens else 0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, self.num_reg_tokens, self.dim)) if reg_tokens else None
        
        self.num_tokens = self.num_patches
        assert pos_embed in ["learn", "sincos", "none"], "pos_embed must be one of ['learn', 'sincos', 'none']"
        if pos_embed == "learn":
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * .02)
        else:
            self.pos_embed = None
        
    def forward(self, x):
        # patchify
        x = self.patchify(x)  # [B, dim, img_size/patch_size, img_size/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, T, dim]
        B, T, D = x.shape
        
        # add positional embedding for patches
        if self.pos_embed is not None:
            # add positional embedding
            if self.pos_embed.shape[1] == self.num_patches:
                pos_embed = self.pos_embed.expand(B, -1, -1)
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed

        # concat cls_token and reg_tokens
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1).clone()
            x = torch.cat((cls_tokens, x), dim=1)
        if self.reg_token is not None:
            reg_tokens = self.reg_token.expand(B, -1, -1).clone()
            x = torch.cat((x, reg_tokens), dim=1)
        
        for i, block in enumerate(self.blocks):
            # apply block
            x = block(x)
    
        assert x.shape[1] == T + 1 + self.num_reg_tokens, f"Expected {T + 1 + self.num_reg_tokens} tokens, got {x.shape[1]} tokens"
        # extract cls_token for classification. discard reg_tokens
        if self.cls_token is not None:
            cls_token = x[:, 0]
        else:
            # If no cls_token, use the mean of all tokens
            cls_token = x.mean(dim=1)
        logits = self.classifier(cls_token)

        return logits

    def pop_stats(self) -> list:
        all_stats = []
        for blk in self.blocks:
            if hasattr(blk, "pop_stats"):
                all_stats.extend(blk.pop_stats())
        return all_stats
    

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn.fused_attn = True
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
        

PRESET_VIT = {
    "S": {"embed_dim": 384,  "depth": 6,  "num_heads": 6},
    "B": {"embed_dim": 768,  "depth": 12, "num_heads": 12},
    "L": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "H": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
}
