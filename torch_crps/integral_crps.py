import torch
from torch.distributions import Distribution, StudentT

from torch_crps.analytical_crps import standardized_studentt_cdf_via_scipy


def crps_integral(
    q: Distribution,
    y: torch.Tensor,
    x_min: float = -1e2,
    x_max: float = 1e2,
    x_steps: int = 5001,
) -> torch.Tensor:
    """Compute the Continuous Ranked Probability Score (CRPS) using the a (somewhat naive) integral approach.

    Note:
        This function is not differentiable with respect to `y` due to the indicator function.

    Args:
        q: A PyTorch distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).
        x_min: Lower limit for integration for the probability space.
        x_max: Upper limit for integration for the probability space.
        x_steps: Number of steps for numerical integration.

    Returns:
        CRPS values for each observation, of shape (num_samples,).
    """

    def integrand(x: torch.Tensor) -> torch.Tensor:
        """Compute the integrand $F(x) - 1(y <= x))^2$ to be used by the torch integration function."""
        if not isinstance(q, StudentT):
            # Default case.
            cdf_value = q.cdf(x)
        else:
            # Special case for torch's StudentT distributions which do not have a cdf method implemented.
            z = (x - q.loc) / q.scale
            cdf_value = standardized_studentt_cdf_via_scipy(z, q.df)
        indicator = (y_expanded <= x).float()
        return (cdf_value - indicator) ** 2

    # Set integration limits.
    x_values = torch.linspace(
        start=torch.tensor(x_min, dtype=y.dtype, device=y.device),
        end=torch.tensor(x_max, dtype=y.dtype, device=y.device),
        steps=x_steps,
        dtype=y.dtype,
        device=y.device,
    )

    # Reshape for proper broadcasting.
    x_values = x_values.unsqueeze(-1)  # shape: (x_steps, 1)
    y_expanded = y.unsqueeze(0)  # shape: (1, num_samples)

    # Compute the integral using the trapezoidal rule.
    integral_values = integrand(x_values)
    crps_values = torch.trapezoid(integral_values, x_values.squeeze(-1), dim=0)

    return crps_values
