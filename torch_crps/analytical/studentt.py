import torch
from torch.distributions import StudentT

from torch_crps.abstract import crps_abstract, scrps_abstract


def standardized_studentt_cdf_via_scipy(
    z: torch.Tensor,
    nu: torch.Tensor | float,
) -> torch.Tensor:
    """Since the `torch.distributions.StudentT` class does not have a `cdf()` method, we resort to scipy which has
    a stable implementation.

    Note:
        - The inputs `z` must be standardized.
        - This breaks differentiability and requires to move tensors to the CPU.

    Args:
        z: Standardized values at which to evaluate the CDF.
        nu: Degrees of freedom of the StudentT distribution.

    Returns:
        CDF values of the standardized StudentT distribution at `z`.
    """
    try:
        from scipy.stats import t as scipy_student_t
    except ImportError as e:
        raise ImportError(
            "scipy is required for the analytical solution for the StudentT distribution. "
            "Install `torch-crps` with the 'studentt' dependency group, e.g. `pip install torch-crps[studentt]`."
        ) from e

    z_np = z.detach().float().cpu().numpy()  # float() handles bfloat16
    nu_np = nu.detach().float().cpu().numpy() if isinstance(nu, torch.Tensor) else nu  # float() handles bfloat16

    cdf_z_np = scipy_student_t.cdf(x=z_np, df=nu_np)

    return torch.from_numpy(cdf_z_np).to(device=z.device, dtype=z.dtype)


def _accuracy_studentt(q: StudentT, y: torch.Tensor) -> torch.Tensor:
    r"""Computes the accuracy term $A = E[|Y - y|]$ for the Student-T distribution.

    $$
    A = \sigma \left[ z(2F_{\nu}(z) - 1) + 2 \frac{\nu+z^2}{\nu-1} f_{\nu}(z) \right]
    $$

    See Also:
        Jordan et al.; "Evaluating Probabilistic Forecasts with scoringRules"; 2019.

    Args:
        q: A PyTorch StudentT distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        Accuracy values for each observation, of shape (num_samples,).
    """
    nu, mu, sigma = q.df, q.loc, q.scale

    # Standardize, and create standard StudentT distribution for CDF and PDF.
    z = (y - mu) / sigma
    standard_t = StudentT(nu, loc=torch.zeros_like(mu), scale=torch.ones_like(sigma))

    # Compute standardized CDF F_ν(z) and PDF f_ν(z).
    cdf_z = standardized_studentt_cdf_via_scipy(z, nu)
    pdf_z = torch.exp(standard_t.log_prob(z))

    # A = sigma * [z * (2*F(z) - 1) + 2*f(z) * (v + z^2) / (v-1) ]
    accuracy_unscaled = z * (2 * cdf_z - 1) + 2 * pdf_z * (nu + z**2) / (nu - 1)

    accuracy = sigma * accuracy_unscaled
    return accuracy


def _dispersion_studentt(
    q: StudentT,
) -> torch.Tensor:
    r"""Computes the dispersion term $D = E[|Y - Y'|]$ for the Student-T distribution.

    See Also:
        Jordan et al.; "Evaluating Probabilistic Forecasts with scoringRules"; 2019.

    Args:
        q: A PyTorch StudentT distribution object, typically a model's output distribution.

    Returns:
        Dispersion values for each observation, of shape (num_samples,).
    """
    nu, sigma = q.df, q.scale

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

    # D = 2σ * 2 * torch.sqrt(v) / (v - 1) * beta_frac
    dispersion = 2 * sigma * 2 * torch.sqrt(nu) / (nu - 1) * beta_frac

    return dispersion


def crps_analytical_studentt(
    q: StudentT,
    y: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the (negatively-oriented) CRPS in closed-form assuming a StudentT distribution.

    This implements the closed-form formula from Jordan et al. (2019), see Appendix A.2.

    For the standardized StudentT distribution:

    $$
    \text{CRPS}(F_\nu, z) = z(2F_\nu(z) - 1) + 2f_\nu(z)\frac{\nu + z^2}{\nu - 1}
        - \frac{2\sqrt{\nu}}{\nu - 1} \frac{B(\frac{1}{2}, \nu - \frac{1}{2})}{B(\frac{1}{2}, \frac{\nu}{2})^2}
    $$

    where $z$ is the standardized value, $F_\nu$ is the CDF, $f_\nu$ is the PDF of the standard StudentT
    distribution, $\nu$ is the degrees of freedom, and $B$ is the beta function.

    For the location-scale transformed distribution:

    $$
    \text{CRPS}(F_{\nu,\mu,\sigma}, y) = \sigma \cdot \text{CRPS}\left(F_\nu, \frac{y-\mu}{\sigma}\right)
    $$

    where $\mu$ is the location parameter, $\sigma$ is the scale parameter, and $y$ is the observation.

    Note:
        This formula is only valid for degrees of freedom $\nu > 1$.

    See Also:
        Jordan et al.; "Evaluating Probabilistic Forecasts with scoringRules"; 2019.

    Args:
        q: A PyTorch StudentT distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        CRPS values for each observation, of shape (num_samples,).
    """
    if torch.any(q.df <= 1):
        raise ValueError("StudentT SCRPS requires degrees of freedom > 1")

    accuracy = _accuracy_studentt(q, y)
    dispersion = _dispersion_studentt(q)

    return crps_abstract(accuracy, dispersion)


def scrps_analytical_studentt(
    q: StudentT,
    y: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the (negatively-oriented) Scaled CRPS (SCRPS) in closed-form assuming a Student-T distribution.

    $$
    \text{SCRPS}(F, y) = -\frac{E[|X - y|]}{E[|X - X'|]} - 0.5 \log \left( E[|X - X'|] \right)
                       = \frac{A}{D} + 0.5 \log(D)
    $$

    where:

    - $F_{\nu, \mu, \sigma^2}$ is the cumulative Student-T distribution, and $F_{\nu}$ is the standardized version.
    - $A = E_F[|X - y|]$ is the accuracy term.
    - $A = \sigma [ z(2 F_{\nu}(z) - 1) +  2(\nu + z²) / (\nu*B(\nu/2, 1/2)) * F_{\nu+1}(z * \sqrt{(\nu+1)/(\nu+z²)}) ]$
    - $D = E_F[|X - X'|]$ is the dispersion term.
    - $D = \frac{ 4\sigma }{ \nu-1 } * ( \frac{ \Gamma( \nu/2 ) }{ \Gamma( (\nu-1)/2) } )^2$

    Note:
        This formula is only valid for degrees of freedom $\nu > 1$.

    See Also:
        Bolin & Wallin; "Local scale invariance and robustness of proper scoring rules"; 2019.

    Args:
        q: A PyTorch StudentT distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        SCRPS values for each observation, of shape (num_samples,).
    """
    if torch.any(q.df <= 1):
        raise ValueError("StudentT SCRPS requires degrees of freedom > 1")

    accuracy = _accuracy_studentt(q, y)
    dispersion = _dispersion_studentt(q)

    return scrps_abstract(accuracy, dispersion)
