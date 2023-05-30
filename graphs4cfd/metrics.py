import torch


def r2(
        pred: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    """Compute the coefficient of determination (a.k.a index of correlation) between `pred` and `target` [https://en.wikipedia.org/wiki/Coefficient_of_determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).
    
    Args:
        pred (torch.Tensor): Predicted values. It can be a time-point or a rollout.
        target (torch.Tensor): Target values. It can be a time-point or a rollout. Dimension must match `pred`.

    Returns:
        torch.Tensor: Coefficient of determination.    
    """

    if (pred.dim()==1) or (pred.dim()==2): # Time-point or rollout
        # Remove elements that cause division by 0
        mask = (target!=target.mean().item())
        res = ((target[mask]-pred[mask])**2).sum().item()
        tot = ((target[mask]-target.mean().item())**2).sum().item()
        return 1 - res / tot
    else:
        raise RuntimeError()