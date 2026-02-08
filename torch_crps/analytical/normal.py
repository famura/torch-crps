import torch
from torch.distributions import Normal

from torch_crps.abstract import crps_abstract, scrps_abstract


def _accuracy_normal(
    q: Normal,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute accuracy term A = E[|X - y|] for a normal distribution.

    Args:
        q: A PyTorch Normal distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        Accuracy values for each observation, of shape (num_samples,).
    """
    z = (y - q.loc) / q.scale
    standard_normal = torch.distributions.Normal(0, 1)

    cdf_z = standard_normal.cdf(z)
    pdf_z = torch.exp(standard_normal.log_prob(z))

    return q.scale * (z * (2 * cdf_z - 1) + 2 * pdf_z)


def _dispersion_normal(
    q: Normal,
) -> torch.Tensor:
    """Compute dispersion term D = E[|X - X'|] for a normal distribution.

    Args:
        q: A PyTorch Normal distribution object, typically a model's output distribution.

    Returns:
        Dispersion values for each observation, of shape (num_samples,).
    """
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=q.loc.device, dtype=q.loc.dtype))

    return 2 * q.scale / sqrt_pi


def crps_analytical_normal(
    q: Normal,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the (negatively-oriented) CRPS in closed-form assuming a normal distribution.

    See Also:
        Gneiting & Raftery; "Strictly Proper Scoring Rules, Prediction, and Estimation"; 2007.
        Equation (5) for the analytical formula for CRPS of Normal distribution.

    Args:
        q: A PyTorch Normal distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        CRPS values for each observation, of shape (num_samples,).
    """
    accuracy = _accuracy_normal(q, y)
    dispersion = _dispersion_normal(q)

    return crps_abstract(accuracy, dispersion)


def scrps_analytical_normal(
    q: Normal,
    y: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the (negatively-oriented) Scaled CRPS (SCRPS) in closed-form assuming a normal distribution.

    $$
    \text{SCRPS}(F, y) = -\frac{E[|X - y|]}{E[|X - X'|]} - 0.5 \\log \\left( E[|X - X'|] \right)
                       = \frac{A}{D} + 0.5 \\log(D)
    $$

    where $X$ and $X'$ are independent random variables drawn from the ensemble distribution, and $F(X)$ is the CDF
    of the ensemble distribution evaluated at $X$, and $y$ are the ground truth observations.

    Note:
        In contrast to the (negatively-oriented) CRPS, the SCRPS can have negative values.

    See Also:
        Bolin & Wallin; "Local scale invariance and robustness of proper scoring rules"; 2019.
        Equation (3) for the definition of the SCRPS.
        Appendix A.1 for the component formulas (Accuracy and Dispersion) for the Normal distribution

    Args:
        q: A PyTorch Normal distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        SCRPS values for each observation, of shape (num_samples,).
    """
    accuracy = _accuracy_normal(q, y)
    dispersion = _dispersion_normal(q)

    return scrps_abstract(accuracy, dispersion)
