import torch
from torch.distributions import Distribution, Normal, StudentT

from torch_crps.analytical.normal import crps_analytical_normal, scrps_analytical_normal
from torch_crps.analytical.studentt import (
    crps_analytical_studentt,
    scrps_analytical_studentt,
)


def crps_analytical(
    q: Distribution,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the (negatively-oriented, i.e., lower is better) CRPS in closed-form.

    Note:
        The input distribution must be either `torch.distributions.Normal` or `torch.distributions.StudentT`.
        There exists analytical solutions for other distributions, but they are not implemented, yet.
        Feel free to create an issue or pull request.

    Args:
        q: A PyTorch distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        CRPS values for each observation, of shape (num_samples,).
    """
    if isinstance(q, Normal):
        return crps_analytical_normal(q, y)
    elif isinstance(q, StudentT):
        return crps_analytical_studentt(q, y)
    else:
        raise NotImplementedError(
            f"Detected distribution of type {type(q)}, but there are only analytical solutions for "
            "`torch.distributions.Normal` or `torch.distributions.StudentT`. Either use an alternative method, e.g. "
            "`torch_crps.crps_integral` or `torch_crps.crps_ensemble`, or create an issue for the method you need."
        )


def scrps_analytical(
    q: Distribution,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the (negatively-oriented, i.e., lower is better) Scaled CRPS (SCRPS) in closed-form.

    Note:
        The input distribution must be either `torch.distributions.Normal` or `torch.distributions.StudentT`.
        There exists analytical solutions for other distributions, but they are not implemented, yet.
        Feel free to create an issue or pull request.

    Args:
        q: A PyTorch distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        SCRPS values for each observation, of shape (num_samples,).
    """
    if isinstance(q, Normal):
        return scrps_analytical_normal(q, y)
    elif isinstance(q, StudentT):
        return scrps_analytical_studentt(q, y)
    else:
        raise NotImplementedError(
            f"Detected distribution of type {type(q)}, but there are only analytical solutions for "
            "`torch.distributions.Normal` or `torch.distributions.StudentT`. Either use an alternative method, e.g. "
            "`torch_crps.scrps_integral` or `torch_crps.scrps_ensemble`, or create an issue for the method you need."
        )
