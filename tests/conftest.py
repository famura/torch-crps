from pathlib import Path

import pytest
import torch
from torch.distributions import Normal, StudentT

from torch_crps.analytical.studentt import standardized_studentt_cdf_via_scipy

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Check if CUDA support is available.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not supported in this setup.")


@pytest.fixture
def case_flat_1d():
    """Fixture for a simple 1D example with a scalar output, to be used with the ensemble methods."""
    return {
        "x": torch.tensor([12.0, 15.0, 16.0, 21.0]),  # only 1 forecast
        "y": torch.tensor(14.5),  # only 1 observation
        "expected_shape": torch.Size([]),
    }


@pytest.fixture
def case_batched_2d():
    """Fixture for a batched 2D example, to be used with the ensemble methods."""
    return {
        "x": torch.tensor(
            [
                [12.0, 15.0, 16.0, 21.0],  # forecast 1
                [30.0, 31.0, 33.0, 38.0],  # forecast 2
            ]
        ),
        "y": torch.tensor(
            [
                14.5,  # observation 1
                35.0,  # observation 2
            ]
        ),
        "expected_shape": torch.Size([2]),
    }


@pytest.fixture
def case_batched_3d():
    """Fixture for a complex 3D example, to be used with the ensemble methods."""
    torch.manual_seed(42)
    return {
        "x": torch.randn(2, 3, 5) * 10 + 50,
        "y": torch.randn(2, 3) * 10 + 50,
        "expected_shape": torch.Size([2, 3]),
    }


def crps_analytical_normal_gneiting(
    q: Normal,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the analytical CRPS assuming a normal distribution.

    See Also:
        Gneiting & Raftery; "Strictly Proper Scoring Rules, Prediction, and Estimation"; 2007
        Equation (5) for the analytical formula for CRPS of Normal distribution.

    Args:
        q: A PyTorch Normal distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        CRPS values for each observation, of shape (num_samples,).
    """
    # Compute standard normal CDF and PDF.
    z = (y - q.loc) / q.scale
    standard_normal = torch.distributions.Normal(0, 1)
    phi_z = standard_normal.cdf(z)  # Φ(z)
    pdf_z = torch.exp(standard_normal.log_prob(z))  # φ(z)

    # Analytical CRPS formula.
    crps = q.scale * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / torch.sqrt(torch.tensor(torch.pi)))

    return crps


def crps_analytical_studentt_jordan(
    q: StudentT,
    y: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the (negatively-oriented) CRPS in closed-form assuming a StudentT distribution.

    This is the previous implementation of the analytical CRPS for StudentT distributions.. It is provided here for
    testing and comparison purposes.

    See Also:
        Jordan et al.; "Evaluating Probabilistic Forecasts with scoringRules"; 2019.

    Args:
        q: A PyTorch StudentT distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        CRPS values for each observation, of shape (num_samples,).
    """
    # Extract degrees of freedom ν, location μ, and scale σ.
    nu, mu, sigma = q.df, q.loc, q.scale
    if torch.any(nu <= 1):
        raise ValueError("StudentT CRPS requires degrees of freedom > 1")

    # Standardize, and create standard StudentT distribution for CDF and PDF.
    z = (y - mu) / sigma
    standard_t = torch.distributions.StudentT(nu, loc=0, scale=1)

    # Compute standardized CDF F_ν(z) and PDF f_ν(z).
    cdf_z = standardized_studentt_cdf_via_scipy(z, nu)
    pdf_z = torch.exp(standard_t.log_prob(z))

    # Compute the beta function ratio: B(1/2, ν - 1/2) / B(1/2, ν/2)^2
    # Using the relationship: B(a,b) = Gamma(a) * Gamma(b) / Gamma(a+b)
    # B(1/2, ν - 1/2) / B(1/2, ν/2)^2 = ( Gamma(1/2) * Gamma(ν-1/2) / Gamma(ν) ) /
    #                                   ( Gamma(1/2) * Gamma(ν/2) / Gamma(ν/2 + 1/2) )^2
    # Simplifying to Gamma(ν - 1/2) Gamma(ν/2 + 1/2)^2 / ( Gamma(ν)Gamma(ν/2)^2 )
    # For numerical stability, we compute in log space.
    log_gamma_half = torch.lgamma(torch.tensor(0.5, dtype=nu.dtype, device=nu.device))
    log_gamma_df_minus_half = torch.lgamma(nu - 0.5)
    log_gamma_df_half = torch.lgamma(nu / 2)
    log_gamma_df_half_plus_half = torch.lgamma(nu / 2 + 0.5)

    # log[B(1/2, ν-1/2)] = log Gamma(1/2) + log Gamma(ν-1/2) - log Gamma(ν)
    # log[B(1/2, ν/2)] = log Gamma(1/2) + log Gamma(ν/2) - log Gamma(ν/2 + 1/2)
    # log[B(1/2, ν-1/2) / B(1/2, ν/2)^2] = log B(1/2, ν-1/2) - 2*log B(1/2, ν/2)
    log_beta_ratio = (
        log_gamma_half
        + log_gamma_df_minus_half
        - torch.lgamma(nu)
        - 2 * (log_gamma_half + log_gamma_df_half - log_gamma_df_half_plus_half)
    )
    beta_frac = torch.exp(log_beta_ratio)

    # Compute the CRPS for standardized values.
    crps_standard = (
        z * (2 * cdf_z - 1) + 2 * pdf_z * (nu + z**2) / (nu - 1) - (2 * torch.sqrt(nu) / (nu - 1)) * beta_frac
    )

    # Apply location-scale transformation CRPS(F_{ν,μ,σ}, y) = σ * CRPS(F_{ν}, z) with z = (y - μ) / σ.
    crps = sigma * crps_standard

    return crps
