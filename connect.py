import logging
import random
from typing import Optional, Sequence
import os

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Callable
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from torch import _dynamo as _torch_dynamo
except ImportError:
    _torch_dynamo = None


def _disable_dynamo(fn):
    if _torch_dynamo is None:
        return fn
    return _torch_dynamo.disable(fn)


_METRICS = (
    "x_norm2", "f_par_norm2", "f_ortho_norm2",
    "rho_par", "rho_ortho", "cos_x_out"
)

TAG2ID = {"attn": 0, "mlp": 1, "conv": 2}
ID2TAG = {v: k for k, v in TAG2ID.items()}
N_TAG   = len(TAG2ID)

_KNOWN_PATTERNS = (
    "rezero_constrained",
    "rezero",
    "rescale_stream",
)

@dataclass
class ConnStat:
    module_name  : str   # module name (e.g. attn, mlp, conv)
    block_id     : int   # block id
    x_norm2      : float # residual stream scale
    f_par_norm2  : float # attention/MLP's x parallel component
    f_ortho_norm2: float # attention/MLP's x orthogonal component
    rho_par      : float # x parallel component ratio
    rho_ortho    : float # x orthogonal component ratio
    cos_x_out    : float # x and attention/MLP's cosine similarity
    extras       : Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_list(cls, t: list) -> "ConnStat":
        module_id_or_name = t[0]
        if isinstance(module_id_or_name, str):
            assert module_id_or_name in TAG2ID, f"Unknown module name: {module_id_or_name}"
            module_name = module_id_or_name
        else:
            module_idx = int(module_id_or_name)
            assert module_idx in ID2TAG, f"Unknown module id: {module_idx}"
            module_name = ID2TAG[module_idx]

        block_id = int(t[1])
        return cls(module_name, block_id, *t[2:])     # type: ignore

    def metrics(self) -> Dict[str, float]:
        """Return a dictionary of the scalar metrics stored in this object."""
        metrics = {k: getattr(self, k) for k in _METRICS}
        metrics.update({f"pattern/{k}": v for k, v in self.extras.items()})
        return metrics

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
    pattern      : str = "default"
    extras       : Dict[str, torch.Tensor] = field(default_factory=dict)

def _normalize_method_and_pattern(method: str, pattern: Optional[str]) -> Tuple[str, str]:
    if not method:
        raise ValueError("connect method must be a non-empty string")

    method_norm = method.strip().lower().replace("-", "_")
    pattern_norm = (pattern or "default").strip().lower().replace("-", "_")
    pattern_norm = pattern_norm.lstrip("_") or "default"

    inferred_pattern: Optional[str] = None
    for candidate in _KNOWN_PATTERNS:
        suffix = f"_{candidate}"
        if method_norm.endswith(suffix):
            method_norm = method_norm[: -len(suffix)]
            inferred_pattern = candidate
            break
        if method_norm == candidate:
            method_norm = "linear"
            inferred_pattern = candidate
            break

    if inferred_pattern is not None:
        if pattern_norm in ("default", "none"):
            pattern_norm = inferred_pattern
        elif pattern_norm != inferred_pattern:
            raise ValueError(
                f"pattern mismatch: pattern='{pattern_norm}' conflicts with method-derived pattern '{inferred_pattern}'"
            )

    if pattern_norm not in ("default", "none") and pattern_norm not in _KNOWN_PATTERNS:
        raise ValueError(f"unknown connection pattern: {pattern_norm}")

    base_method = method_norm or "linear"
    return base_method, pattern_norm


def _ensure_components(results: RawConnStat) -> RawConnStat:
    """Ensure dot/x_norm2/f_par/f_ortho are populated on the RawConnStat."""
    if (
        results.dot is not None
        and results.x_norm2 is not None
        and results.f_par is not None
        and results.f_ortho is not None
    ):
        return results

    dim = results.dim
    eps = results.eps
    x = results.x
    f_x = results.f_x

    dot = (x * f_x).sum(dim, keepdim=True)
    x_norm2 = (x * x).sum(dim, keepdim=True).float().clamp_min(eps)
    scale = (dot / x_norm2).to(dtype=x.dtype)
    f_par = scale * x
    f_ortho = f_x - f_par

    results.dot = dot
    results.x_norm2 = x_norm2
    results.f_par = f_par
    results.f_ortho = f_ortho
    return results

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

def _rezero(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, RawConnStat]:
    """
    ReZero residual connection
    x   : residual stream
    f_x : attention/MLP/conv(if channel-wise) output
    alpha : learnable scalar parameter
    """
    stream = x + alpha * f_x
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
        pattern="rezero",
    )
    results.extras["alpha"] = alpha.detach()
    return stream, results

def _rezero_constrained(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, RawConnStat]:
    """
    ReZero residual connection with constrained alpha
    x   : residual stream
    f_x : attention/MLP/conv(if channel-wise) output
    theta : unconstrained learnable parameter
    """
    
    # sin^2 + cos^2 = 1
    alpha = torch.cos(theta)
    beta = torch.sin(theta)
    
    dot = (x * f_x).sum(dim, keepdim=True)
    x_norm2 = (x * x).sum(dim, keepdim=True).float()
    x_norm2 = x_norm2.clamp_min(eps)  # numerical stability

    f_par = (dot / x_norm2).to(dtype=x.dtype) * x  # [B,1]
    f_ortho = f_x - f_par

    f_par = (alpha * f_par).to(dtype=x.dtype)  # scale parallel component
    f_ortho = (beta * f_ortho).to(dtype=x.dtype)  # scale orthogonal component

    stream = x + f_par + f_ortho
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
        pattern="rezero_constrained",
    )
    results.extras["theta"] = theta.detach()
    results.extras["alpha"] = alpha.detach()
    results.extras["beta"] = beta.detach()
    return stream, results


def connect(
    x: torch.Tensor,
    f_x: torch.Tensor,
    *,
    method: str = "linear",
    orthogonal_method: str = "feature",
    dim: int = -1,
    eps: float | torch.Tensor = 1e-6,
    perturbation: Optional[float] = None,
    pattern: Optional[str] = None,
    alpha: Optional[torch.Tensor] = None,
    theta: Optional[torch.Tensor] = None,
    rescale_alpha: Optional[torch.Tensor] = None,
    rescale_proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    rescale_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, RawConnStat]:
    if perturbation is not None:
        f_x = f_x + torch.randn_like(f_x) * perturbation

    method, pattern = _normalize_method_and_pattern(method, pattern)

    if isinstance(eps, float):
        eps_tensor = torch.tensor([eps], device=x.device, dtype=torch.float32)
    else:
        eps_tensor = eps.to(device=x.device, dtype=torch.float32)

    if method == "linear":
        if orthogonal_method == "negative":
            if pattern not in ("default", "none"):
                raise ValueError("negative residual connection does not support auxiliary patterns")
            stream, results = _negative(x, f_x, dim, eps_tensor)
            results.pattern = "negative"
            return stream, results

        if pattern == "rezero":
            if alpha is None:
                raise ValueError("pattern 'rezero' requires an alpha parameter")
            stream, results = _rezero(x, f_x, dim, eps_tensor, alpha)
            return stream, results

        if pattern == "rezero_constrained":
            if theta is None:
                raise ValueError("pattern 'rezero_constrained' requires a theta parameter")
            stream, results = _rezero_constrained(x, f_x, dim, eps_tensor, theta)
            return stream, results

        dot = (x * f_x).sum(dim, keepdim=True)
        x_norm2 = (x * x).sum(dim, keepdim=True).float().clamp_min(eps_tensor)
        scale = (dot / x_norm2).to(dtype=x.dtype)
        f_par = scale * x
        f_ortho = f_x - f_par
        stream = x + f_x
        results = RawConnStat(
            dim=dim,
            eps=eps_tensor,
            x=x,
            f_x=f_x,
            stream=stream,
            dot=dot,
            x_norm2=x_norm2,
            f_par=f_par,
            f_ortho=f_ortho,
        )
    elif method == "orthogonal":
        if orthogonal_method == "global":
            results_with_stat = _orthogonal_global(x, f_x, dim, eps_tensor)
        elif orthogonal_method == "feature":
            results_with_stat = _orthogonal_channel(x, f_x, dim, eps_tensor)
        else:
            raise ValueError(f"unknown orthogonal method: {orthogonal_method}")
        f_ortho, results = results_with_stat
        if pattern in ("default", "none"):
            stream = x + f_ortho
            results.stream = stream
            results.pattern = pattern
            return stream, results
    else:
        raise ValueError(f"unknown connect method: {method}")

    if pattern in ("default", "none"):
        results.pattern = pattern
        return stream, results

    if pattern == "rezero":
        if alpha is None:
            raise ValueError("pattern 'rezero' requires an alpha parameter")
        if method == "linear":
            stream, results = _rezero(x, f_x, dim, eps_tensor, alpha)
            results.pattern = pattern
            return stream, results

        if method == "orthogonal":
            results = _ensure_components(results)
            raw_f_ortho = results.f_ortho
            if raw_f_ortho is None:
                raise RuntimeError("orthogonal rezero expects f_ortho to be available")
            scaled_f_ortho = (alpha.to(dtype=raw_f_ortho.dtype) * raw_f_ortho)
            results.extras["alpha"] = alpha.detach()
            raw_norm = (raw_f_ortho * raw_f_ortho).sum(dim, keepdim=True).mean().detach()
            results.extras["f_ortho_raw_norm2"] = raw_norm
            results.f_ortho = scaled_f_ortho
            results.f_x = scaled_f_ortho
            results.stream = x + scaled_f_ortho
            results.dot = (results.x * results.f_x).sum(dim, keepdim=True)
            results.pattern = pattern
            return results.stream, results

        raise ValueError(f"pattern 'rezero' unsupported for method='{method}'")

    if pattern == "rezero_constrained":
        if theta is None:
            raise ValueError("pattern 'rezero_constrained' requires a theta parameter")

        alpha_val = torch.cos(theta)
        beta_val = torch.sin(theta)

        if method == "linear":
            stream, results = _rezero_constrained(x, f_x, dim, eps_tensor, theta)
            results.pattern = pattern
            return stream, results

        if method == "orthogonal":
            results = _ensure_components(results)
            raw_f_par = results.f_par
            raw_f_ortho = results.f_ortho
            if raw_f_ortho is None:
                raise RuntimeError("orthogonal rezero_constrained expects f_ortho to be available")

            alpha_cast = alpha_val.to(dtype=x.dtype)
            beta_cast = beta_val.to(dtype=x.dtype)
            scaled_f_par = alpha_cast * raw_f_par if raw_f_par is not None else None
            scaled_f_ortho = beta_cast * raw_f_ortho

            results.extras["theta"] = theta.detach()
            results.extras["alpha"] = alpha_val.detach()
            results.extras["beta"] = beta_val.detach()
            if raw_f_par is not None:
                raw_par_norm = (raw_f_par * raw_f_par).sum(dim, keepdim=True).mean().detach()
                results.extras["f_par_raw_norm2"] = raw_par_norm
                results.f_par = scaled_f_par
            raw_ortho_norm = (raw_f_ortho * raw_f_ortho).sum(dim, keepdim=True).mean().detach()
            results.extras["f_ortho_raw_norm2"] = raw_ortho_norm
            results.f_ortho = scaled_f_ortho
            results.f_x = scaled_f_ortho if scaled_f_par is None else (scaled_f_par + scaled_f_ortho)
            results.stream = x + results.f_x
            results.dot = (results.x * results.f_x).sum(dim, keepdim=True)
            results.pattern = pattern
            return results.stream, results

        raise ValueError(f"pattern 'rezero_constrained' unsupported for method='{method}'")

    if pattern == "rescale_stream":
        results = _ensure_components(results)
        rescale_mode = (rescale_mode or "scalar").lower()

        if rescale_mode == "conv1x1":
            if rescale_proj is None:
                raise ValueError("rescale_mode 'conv1x1' requires rescale_proj callable")
            scaled_x = rescale_proj(x)
            if isinstance(rescale_proj, nn.Module) and hasattr(rescale_proj, "weight"):
                results.extras["rescale_conv_weight_norm"] = rescale_proj.weight.detach().norm().unsqueeze(0)
            results.extras["rescale_mode"] = torch.tensor([1.0], device=x.device)
        else:
            if rescale_alpha is None:
                raise ValueError("pattern 'rescale_stream' with scalar mode requires rescale_alpha")
            scaled_x = (1 + rescale_alpha.to(dtype=x.dtype)) * x
            results.extras["rescale_alpha"] = rescale_alpha.detach()
            results.extras["rescale_mode"] = torch.tensor([0.0], device=x.device)

        delta_parallel = scaled_x - x

        raw_f_par = results.f_par
        if raw_f_par is not None:
            raw_par_norm = (raw_f_par * raw_f_par).sum(dim, keepdim=True).mean().detach()
            results.extras["f_par_raw_norm2"] = raw_par_norm

        f_ortho = results.f_ortho
        if f_ortho is None:
            f_ortho = torch.zeros_like(delta_parallel)
            results.f_ortho = f_ortho

        results.f_par = delta_parallel
        results.f_x = delta_parallel + f_ortho
        results.stream = scaled_x + f_ortho
        results.dot = (results.x * results.f_x).sum(dim, keepdim=True)

        scaled_norm2 = (scaled_x * scaled_x).sum(dim, keepdim=True).float().clamp_min(eps_tensor)
        results.extras["scaled_x_norm2"] = scaled_norm2.detach().mean()
        if results.x_norm2 is None:
            results.x_norm2 = (results.x * results.x).sum(dim, keepdim=True).float().clamp_min(eps_tensor)
        ratio = (scaled_norm2 / results.x_norm2).mean().detach()
        results.extras["stream_scale_ratio"] = ratio
        results.pattern = pattern
        return results.stream, results

    raise ValueError(f"unknown connection pattern: {pattern}")

def _stats(results: RawConnStat) -> Dict[str, torch.Tensor]:
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

    return {
        "x_norm2": x_norm2.mean().detach(),
        "f_par_norm2": f_par_norm2.mean().detach(),
        "f_ortho_norm2": f_ortho_norm2.mean().detach(),
        "rho_par": (f_par_norm2 / x_norm2).mean().detach(),
        "rho_ortho": (f_ortho_norm2 / x_norm2).mean().detach(),
        "cos_x_out": cos.mean().detach(),
    }

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
        self._step_stats: List[ConnStat] = []  # 변경: 텐서 dict 대신 ConnStat 바로 저장
        self._call_counter = 0  # 추가: 내부 호출 카운터
        self._pattern_params = nn.ParameterDict()
        self._pattern_modules = nn.ModuleDict()

    # to read the step (kept for backward compatibility, currently unused)
    def set_step_fn(self, fn):
        self._get_step = fn

    def enable_activation_logging(self):
        self.log_activations = True

    def disable_activation_logging(self):
        self.log_activations = False

    def _connect_and_collect(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        tag="conv",
        method="orthogonal",
        orthogonal_method="feature",
        eps=None,
        perturbation=None,
        pattern: str = "default",
        rescale_mode: Optional[str] = None,
    ):
        assert tag in TAG2ID, f"Unknown tag: {tag}"
        vec_dim = 1 if tag == "conv" else -1   # channel vs hidden
        if tag == "conv":
            orthogonal_method = orthogonal_method if orthogonal_method == "negative" else "feature"

        if not torch.is_tensor(eps):
            eps = torch.tensor([1e-6], device=x.device, dtype=torch.float32)
        else:
            if eps.device != x.device:
                raise RuntimeError(f"eps on {eps.device}, x on {x.device}")   # for debug
        pattern_kwargs = self._pattern_kwargs(tag, pattern, rescale_mode)
        stream, results = connect(
            x, out,
            dim=vec_dim,
            method=method, orthogonal_method=orthogonal_method,
            perturbation=perturbation, eps=eps,
            pattern=pattern,
            **pattern_kwargs,
        )
        stream = stream.to(x.dtype)
        
        # 수정: log_interval 주기로만 수집 (환경변수로 강제 가능)
        self._call_counter += 1
        # Collect only every log_interval-th call (simplified; env override removed)
        should_log = (
            self.log_activations and (self._call_counter % max(1, self.log_interval) == 0)
        )
        if should_log:
            self._store_stats(tag, self.block_id, results)
        return stream

    def _pattern_kwargs(
        self,
        tag: str,
        pattern: Optional[str],
        rescale_mode: Optional[str],
    ) -> Dict[str, Any]:
        if pattern is None:
            pattern = "default"
        pattern = pattern.lower()
        if pattern in ("default", "none"):
            return {}

        if pattern == "rezero":
            key = f"{tag}_alpha"
            if key not in self._pattern_params:
                raise RuntimeError(f"missing ReZero alpha parameter for tag '{tag}'")
            return {"alpha": self._pattern_params[key]}

        if pattern == "rezero_constrained":
            key = f"{tag}_theta"
            if key not in self._pattern_params:
                raise RuntimeError(f"missing constrained ReZero theta parameter for tag '{tag}'")
            return {"theta": self._pattern_params[key]}

        if pattern == "rescale_stream":
            mode = (rescale_mode or "scalar").lower()
            if mode == "conv1x1":
                key = f"{tag}_rescale_proj"
                if key not in self._pattern_modules:
                    raise RuntimeError(f"missing rescale projection module for tag '{tag}'")
                return {
                    "rescale_proj": self._pattern_modules[key],
                    "rescale_mode": mode,
                }
            key = f"{tag}_rescale_alpha"
            if key not in self._pattern_params:
                raise RuntimeError(f"missing rescale alpha parameter for tag '{tag}'")
            return {
                "rescale_alpha": self._pattern_params[key],
                "rescale_mode": mode,
            }

        raise ValueError(f"unknown residual pattern '{pattern}'")

    @_disable_dynamo
    def _store_stats(self, tag: str, block_id: int, results: RawConnStat):
        if not isinstance(results, RawConnStat):
            return
        with torch.no_grad():
            metrics = _stats(results)  # dict[str, tensor]
            # 즉시 스칼라화 → compile 그래프에 텐서 참조 남기지 않음
            scalar_metrics = {k: v.item() for k, v in metrics.items()}
        extras: Dict[str, float] = {}
        for key, value in (results.extras or {}).items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                if value.numel() == 1:
                    extras[key] = value.detach().item()
                else:
                    extras[f"{key}_mean"] = value.detach().float().mean().item()
            elif isinstance(value, (float, int)):
                extras[key] = float(value)
        self._step_stats.append(
            ConnStat(
                module_name=tag,
                block_id=block_id,
                extras=extras,
                **scalar_metrics,
            )
        )

    def pop_stats(self, *, scalarize: bool = True) -> list:
        cached = list(self._step_stats)
        self._step_stats.clear()
        if not scalarize:
            return []
        # 이미 ConnStat 형태
        return cached


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
