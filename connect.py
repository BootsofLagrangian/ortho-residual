import logging
import random
from typing import Optional, Sequence

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.compiler

_METRICS = (
    "x_norm2", "f_par_norm2", "f_ortho_norm2",
    "rho_par", "rho_ortho", "cos_x_out"
)

TAG2ID = {"attn": 0, "mlp": 1, "conv": 2}
ID2TAG = {v: k for k, v in TAG2ID.items()}
N_TAG   = len(TAG2ID)

@dataclass
class ConnStat:
    module_name  : str   # module name
    block_id     : int   # block id
    x_norm2      : float # residual stream scale
    f_par_norm2  : float # attention/MLP's x parallel component
    f_ortho_norm2: float # attention/MLP's x orthogonal component
    rho_par      : float # x parallel component ratio
    rho_ortho    : float # x orthogonal component ratio
    cos_x_out    : float # x and attention/MLP's cosine similarity

    @classmethod
    def from_list(cls, t: list) -> "ConnStat":
        module_name = t[0]
        assert module_name in ID2TAG, f"Unknown module name: {module_name}"
        block_id = int(t[1])
        return cls(module_name, block_id, *t[2:])     # type: ignore

@dataclass
class RawConnStat:
    dim          : int          # dimension of the residual stream
    eps          : torch.Tensor # eps for numerical stability
    x            : torch.Tensor # residual stream
    f_x          : torch.Tensor # attention/MLP/conv(if channel-wise) output

    # Optional or computed later
    stream       : Optional[torch.Tensor]

    dot          : Optional[torch.Tensor] = None # dot product
    x_norm2      : Optional[torch.Tensor] = None # residual stream scale
    f_par        : Optional[torch.Tensor] = None # attention/MLP's x parallel component
    f_ortho      : Optional[torch.Tensor] = None # attention/MLP's x orthogonal component

def _orthogonal_channel(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps: torch.Tensor) -> Tuple[torch.Tensor, RawConnStat]:
    """
    orthogonal residual connection
    x   : residual stream
    f_x : attention/MLP/conv(if channel-wise) output
    """
    # eps      = eps.to(x.device) # torch.compiler hate this
    dot      = (x * f_x).sum(dim, keepdim=True)
    x_norm2  = (x * x  ).sum(dim, keepdim=True).float() + eps
    scale = (dot / x_norm2).to(dtype=x.dtype)  # [B,1]
    f_par = scale * x
    f_ortho = f_x - f_par

    results = RawConnStat(
        dim=dim,
        eps=eps,
        x=x,
        f_x=f_x,
        stream=None,
        dot=dot,
        x_norm2=x_norm2,
        f_par=f_par,
        f_ortho=f_ortho,
    )
    return f_ortho, results

def _orthogonal_global(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps: torch.Tensor) -> Tuple[torch.Tensor, RawConnStat]:
    """
    orthogonal residual connection
    x   : residual stream
    f_x : conv output
    """    
    original_shape = x.shape
    positive_dim = dim if dim >= 0 else len(original_shape) + dim

    # eps = eps.to(x.device)
    x_view   = x.flatten(dim)          # [B, CHW...]
    f_view   = f_x.flatten(dim)        # same
    dot      = (x_view * f_view).sum(dim=dim, keepdim=True)  # [B,1]
    x_norm2  = (x_view * x_view).sum(dim=dim, keepdim=True).float() + eps

    scale = (dot / x_norm2).to(dtype=x.dtype)  # [B,1]
    unsqueeze_times = len(original_shape) - positive_dim - 1
    for _ in range(unsqueeze_times):
        scale = scale.unsqueeze(-1)
    f_par = scale * x  # broadcast
    f_ortho = f_x - f_par

    results = RawConnStat(
        dim=dim,
        eps=eps,
        x=x,
        f_x=f_x,
        stream=None,
        dot=dot,
        x_norm2=x_norm2,
        f_par=f_par,
        f_ortho=f_ortho,
    )
    return f_ortho, results

def _negative(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps: torch.Tensor) -> Tuple[torch.Tensor, RawConnStat]:
    """
    negative residual connection
    x   : residual stream
    f_x : attention/MLP/conv(if channel-wise) output
    """
    # eps = eps.to(x.device)
    stream = x - f_x
    dot = (x * f_x).sum(dim, keepdim=True)
    x_norm2 = (x * x).sum(dim, keepdim=True).float()
    x_norm2 = x_norm2.clamp_min(eps)  # numerical stability

    f_par = (dot / x_norm2).to(dtype=x.dtype) * x  # [B,1]
    f_ortho = f_x - f_par

    results = RawConnStat(
        dim=dim,
        eps=eps,
        x=x,
        f_x=f_x,
        stream=stream,
        dot=dot,
        x_norm2=x_norm2,
        f_par=f_par,
        f_ortho=f_ortho,
    )
    return stream, results

def connect(x: torch.Tensor, f_x: torch.Tensor, *, 
            method="linear", orthogonal_method="global",
            dim=-1, eps=1e-6, perturbation=None) -> Tuple[torch.Tensor, RawConnStat]:
    if perturbation is not None:
        f_x = f_x + torch.randn_like(f_x) * perturbation
    if method == "linear":
        if orthogonal_method == "negative":
            stream, results = _negative(x, f_x, dim, eps)
        else:
            stream = x + f_x
            results = RawConnStat(
                dim=dim,
                eps=eps,
                x=x,
                f_x=f_x,
                stream=stream,
                dot=None,
                x_norm2=None,
                f_par=None,
                f_ortho=None,
            )
        return stream, results
    elif method == "orthogonal":
        if orthogonal_method == "global":
            results_with_stat = _orthogonal_global(x, f_x, dim, eps)
        elif orthogonal_method == "channel":
            results_with_stat = _orthogonal_channel(x, f_x, dim, eps)
        else:
            raise ValueError(f"unknown orthogonal method: {method}")
        f_ortho, results = results_with_stat
        stream = x + f_ortho
        return stream, results
    else:
        raise ValueError(f"unknown connect method: {method}")

def _stats(results: RawConnStat) -> Tuple[torch.Tensor, ...]:
    dim = results.dim
    eps = results.eps
    
    # Calculate x_norm2 if not available
    if results.x_norm2 is None:
        x_norm2 = (results.x * results.x).sum(dim, keepdim=True).clamp_min(eps)
    else:
        x_norm2 = results.x_norm2
    
    # Calculate dot if not available
    if results.dot is None:
        dot = (results.x * results.f_x).sum(dim, keepdim=True)
    else:
        dot = results.dot
    
    # Calculate f_par and f_par_norm2
    if results.f_par is None:
        scale = (dot / x_norm2).to(dtype=results.x.dtype)
        f_par = scale * results.x
        f_par_norm2 = (f_par * f_par).sum(dim, keepdim=True).clamp_min(eps)
    else:
        f_par_norm2 = (results.f_par * results.f_par).sum(dim, keepdim=True).clamp_min(eps)
    
    # Calculate f_ortho and f_ortho_norm2
    if results.f_ortho is None:
        if results.f_par is None:
            scale = (dot / x_norm2).to(dtype=results.x.dtype)
            f_par = scale * results.x
        else:
            f_par = results.f_par
        f_ortho = results.f_x - f_par
        f_ortho_norm2 = (f_ortho * f_ortho).sum(dim, keepdim=True).clamp_min(eps)
    else:
        f_ortho_norm2 = (results.f_ortho * results.f_ortho).sum(dim, keepdim=True).clamp_min(eps)
    
    # Calculate f_x_norm2 for normalization
    f_x_norm2 = (results.f_x * results.f_x).sum(dim, keepdim=True).clamp_min(eps)
    
    # Calculate cosine similarity
    denom = torch.sqrt(x_norm2 * f_x_norm2)
    cos = dot / denom

    stat = ConnStat(
        module_name=None,
        block_id=None,
        x_norm2=x_norm2.mean().item(),  
        f_par_norm2=f_par_norm2.mean().item(),
        f_ortho_norm2=f_ortho_norm2.mean().item(),
        rho_par=(f_par_norm2 / x_norm2).mean().item(),
        rho_ortho=(f_ortho_norm2 / x_norm2).mean().item(),
        cos_x_out=cos.mean().item(),
    )
    
    return stat

def set_connect(
    module: torch.nn.Module,
    pattern: Optional[Sequence[int]] = None,
    prob: Optional[float] = None,
    default: str = "linear",
    logger: Optional[logging.Logger] = None,
):
    """
    Walk `module`, locate all sub‐modules that support an orthogonal/linear connect_method
    (either via `.connect_method` or a `._res_kwargs['method']` or `.residual_kwargs['method']`),
    and set each one’s method.

    Args:
        module:   root module to search (e.g. your ViT or ResNet).
        pattern:  if given, a list of block indices that should be 'orthogonal'; all others become 'linear'.
        prob:     if given (and pattern is None), for each block choose 'orthogonal' with this probability.
        default:  fallback method when neither pattern nor prob is set.
        logger:   optional logger to record which block gets which method.
    """
    # gather all sub‐modules that support a connect‐method
    blocks = []
    for m in module.modules():
        if hasattr(m, "connect_method") or hasattr(m, "_res_kwargs") or hasattr(m, "residual_kwargs"):
            blocks.append(m)

    for idx, blk in enumerate(blocks):
        # decide method for this block
        if pattern is not None:
            method = "orthogonal" if idx in pattern else "linear"
        elif prob is not None:
            method = "orthogonal" if random.random() < prob else "linear"
        else:
            method = default

        # apply it
        if hasattr(blk, "connect_method"):
            blk.connect_method = method
        elif hasattr(blk, "_res_kwargs"):
            blk._res_kwargs["method"] = method
        else:  # hasattr(blk, 'residual_kwargs')
            blk.residual_kwargs["method"] = method

        if logger is not None:
            logger.info(f"Block {idx}: connect_method={method}")

class ConnLoggerMixin:
    """
    ConnLoggerMixin is a mixin class for logging connection statistics in neural networks.
    It collects statistics about the connection between the input and output tensors
    during the forward pass of the network. The statistics include norms, ratios,
    and cosine similarities of the tensors involved in the connection.
    """
    _global_block_id = 0

    def __init__(self, log_interval=50, log_activations=True):
        super().__init__()
        self.log_interval   = log_interval
        self.log_activations = log_activations
        self.block_id = ConnLoggerMixin._global_block_id
        ConnLoggerMixin._global_block_id += 1
        self._step_stats: List[ConnStat] = []

    # to read the step
    def set_step_fn(self, fn):
        self._get_step = fn

    def _connect_and_collect(
        self, 
        x: torch.Tensor, 
        out: torch.Tensor, 
        *,
        tag="conv",
        method="orthogonal",
        orthogonal_method="global",
        eps=None,
        perturbation=None,
    ):
        assert tag in TAG2ID, f"Unknown tag: {tag}"
        vec_dim = 1 if tag == "conv" else -1   # channel vs hidden
        if tag == "conv":
            orthogonal_method = orthogonal_method if orthogonal_method == "negative" else "channel"

        if not torch.is_tensor(eps):
            eps = torch.tensor([1e-6], device=x.device, dtype=torch.float32)
        else:
            if eps.device != x.device:
                raise RuntimeError(f"eps on {eps.device}, x on {x.device}")   # for debug
            
        stream, results = connect(
            x, out,
            dim=vec_dim,
            method=method, orthogonal_method=orthogonal_method,
            perturbation=perturbation, eps=eps
        )
        stream = stream.to(x.dtype) # for AMP
        
        if not torch.compiler.is_compiling():
            if hasattr(self, "_get_step") and self.log_activations:
                step = self._get_step()
                if step % self.log_interval == 0 and isinstance(results, RawConnStat):
                    with torch.no_grad():
                        stat = _stats(results)
                        stat.tag = tag
                        stat.block_id = self.block_id
                        self._step_stats.append(stat)
        return stream

    def pop_stats(self) -> list:
        stats, self._step_stats = self._step_stats, []
        return stats


if __name__ == "__main__":
    # Test the connect function
    x = torch.randn(2, 3, 4)
    f = torch.randn_like(x)
    y = connect(x, f, method="linear", dim=-1)
    assert torch.allclose(y, x + f), "linear connection failed"

    y = connect(x, f, method="orthogonal", dim=-1)
    assert torch.allclose(
        ((y - x) * x).sum(-1), torch.zeros_like(x[...,0]), atol=1e-5
    ), "orthogonal connection failed"
