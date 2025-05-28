import torch

def _identity(x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
    """
    linear residual connection
    x   : residual stream
    f_x : attention/MLP/conv(if channel-wise) output
    """
    return f_x

def _orthogonal_channel(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    """
    orthogonal residual connection
    x   : residual stream
    f_x : attention/MLP/conv(if channel-wise) output
    """
    dot      = (x * f_x).sum(dim, keepdim=True)
    norm_x2  = (x * x  ).sum(dim, keepdim=True) + eps
    scale = (dot / norm_x2).to(x.dtype) # for amp
    proj_out = scale * x
    return f_x - proj_out

def _orthogonal_global(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    """
    orthogonal residual connection
    x   : residual stream
    f_x : conv output
    """    
    original_shape = x.shape
    positive_dim = dim if dim >= 0 else len(original_shape) + dim

    x_view   = x.flatten(dim)          # [B, CHW...]
    f_view   = f_x.flatten(dim)        # same
    dot      = (x_view * f_view).sum(dim=dim, keepdim=True)  # [B,1]
    norm_sq  = (x_view * x_view).sum(dim=dim, keepdim=True) + eps

    scale = (dot / norm_sq).to(x.dtype) # for amp
    unsqueeze_times = len(original_shape) - positive_dim - 1
    for _ in range(unsqueeze_times):
        scale = scale.unsqueeze(-1)
    proj_out = scale * x  # broadcast
    return f_x - proj_out

def connect(x, f_x, *, 
            method="linear", orthogonal_method="global",
            dim=-1, eps=1e-6, perturbation=None):
    if perturbation is not None:
        raise NotImplementedError("perturbation is not implemented yet")
    if method == "linear":
        return x + _identity(x, f_x)
    elif method == "orthogonal":
        if orthogonal_method == "global":
            return x + _orthogonal_global(x, f_x, dim, eps)
        elif orthogonal_method == "channel":
            return x + _orthogonal_channel(x, f_x, dim, eps)
        else:
            raise ValueError(f"unknown orthogonal  method: {method}")
    else:
        raise ValueError(f"unknown connect method: {method}")

