import torch
from scipy.stats import t as scipy_student_t
from torch.distributions import Distribution, Normal, StudentT


def crps_analytical_naive_integral(
    q: Distribution,
    y: torch.Tensor,
    x_min: float = -1e2,
    x_max: float = 1e2,
    x_steps: int = 5001,
) -> torch.Tensor:
    """Compute the Continuous Ranked Probability Score (CRPS) using the naive integral method.

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
        """Compute the integrand $F(x) - 1(y <= x))^2$ to be used by the torch integration functions."""
        if not isinstance(q, StudentT):
            # Default case.
            cdf_value = q.cdf(x)
        else:
            # Special case for torch's StudentT distributions which do not have a cdf method implemented.
            z = (x - q.loc) / q.scale
            cdf_value = _standardized_studentt_cdf_via_scipy(z, q.df)
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


def crps_analytical_normal(
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
    z = (y - q.loc) / q.scale  # standardize
    standard_normal = torch.distributions.Normal(0, 1)
    phi_z = standard_normal.cdf(z)  # Φ(z)
    pdf_z = torch.exp(standard_normal.log_prob(z))  # φ(z)

    # Analytical CRPS formula.
    crps = q.scale * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / torch.sqrt(torch.tensor(torch.pi)))

    return crps


def _standardized_studentt_cdf_via_scipy(z: torch.Tensor, df: torch.Tensor | float) -> torch.Tensor:
    """Since the `torch.distributions.StudentT` class does not have a `cdf()` method, we resort to scipy which has
    a stable implementation.

    Note:
        - The inputs `z` must be standardized.
        - This breaks differentiability and requires to move tensors to the CPU.

    Args:
        z: Standardized values at which to evaluate the CDF.
        df: Degrees of freedom of the StudentT distribution.

    Returns:
        CDF values of the standardized StudentT distribution at `z`.
    """
    z_np = z.detach().cpu().numpy()
    df_np = df.detach().cpu().numpy() if isinstance(df, torch.Tensor) else df

    cdf_np = scipy_student_t.cdf(z_np, df=df_np)

    f_cdf_z = torch.from_numpy(cdf_np).to(device=z.device, dtype=z.dtype)
    return f_cdf_z


def crps_analytical_studentt(
    q: StudentT,
    y: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the analytical CRPS assuming a StudentT distribution.

    This implements the closed-form formula from Jordan et al. (2019), see Appendix A.2.

    For the standardized StudentT distribution:

    $$ \text{CRPS}(F_\nu, z) = z(2F_\nu(z) - 1) + 2f_\nu(z)\frac{\nu + z^2}{\nu - 1}
    - \frac{2\sqrt{\nu}}{\nu - 1} \frac{B(\frac{1}{2}, \nu - \frac{1}{2})}{B(\frac{1}{2}, \frac{\nu}{2})^2} $$

    where $z$ is the standardized value, $F_\nu$ is the CDF, $f_\nu$ is the PDF of the standard StudentT
    distribution, $\nu$ is the degrees of freedom, and $B$ is the beta function.

    For the location-scale transformed distribution:

    $$ \text{CRPS}(F_{\nu,\mu,\sigma}, y) = \sigma \cdot \text{CRPS}\left(F_\nu, \frac{y-\mu}{\sigma}\right) $$

    where $\mu$ is the location parameter, $\sigma$ is the scale parameter, and $y$ is the observation.

    Note:
        This formula is only valid for degrees of freedom $\nu > 1$.

    See Also:
        Jordan et al.; "Evaluating Probabilistic Forecasts with scoringRules"; 2019; Appendix A.2.

    Args:
        q: A PyTorch StudentT distribution object, typically a model's output distribution.
        y: Observed values, of shape (num_samples,).

    Returns:
        CRPS values for each observation, of shape (num_samples,).
    """
    # Extract degrees of freedom (nu), location (mu), and scale (sigma).
    df, loc, scale = q.df, q.loc, q.scale

    if torch.any(df <= 1):
        raise ValueError("StudentT CRPS requires degrees of freedom > 1")

    # Standardize, and create standard StudentT distribution for CDF and PDF.
    z = (y - loc) / scale
    standard_t = torch.distributions.StudentT(df, loc=0, scale=1)

    # Compute standardized CDF F_nu(z) and PDF f_nu(z).
    f_cdf_z = _standardized_studentt_cdf_via_scipy(z, df)
    f_z = torch.exp(standard_t.log_prob(z))

    # Compute the beta function ratio: B(1/2, nu - 1/2) / B(1/2, nu/2)^2
    # Using the relationship: B(a,b) = Gamma(a) * Gamma(b) / Gamma(a+b)
    # B(1/2, nu - 1/2) / B(1/2, nu/2)^2 = ( Gamma(1/2) * Gamma(nu-1/2) / Gamma(nu) ) /
    #                                     ( Gamma(1/2) * Gamma(nu/2) / Gamma(nu/2 + 1/2) )^2
    # Simplifying to Gamma(nu - 1/2) Gamma(nu/2 + 1/2)^2 / ( Gamma(nu)Gamma(nu/2)^2 )
    # For numerical stability, we compute in log space.
    log_gamma_half = torch.lgamma(torch.tensor(0.5, dtype=df.dtype, device=df.device))
    log_gamma_df_minus_half = torch.lgamma(df - 0.5)
    log_gamma_df_half = torch.lgamma(df / 2)
    log_gamma_df_half_plus_half = torch.lgamma(df / 2 + 0.5)

    # log[B(1/2, nu-1/2)] = log Gamma(1/2) + log Gamma(nu-1/2) - log Gamma(nu)
    # log[B(1/2, nu/2)] = log Gamma(1/2) + log Gamma(nu/2) - log Gamma(nu/2 + 1/2)
    # log[B(1/2, nu-1/2) / B(1/2, nu/2)^2] = log B(1/2, nu-1/2) - 2*log B(1/2, nu/2)
    log_beta_ratio = (
        log_gamma_half
        + log_gamma_df_minus_half
        - torch.lgamma(df)
        - 2 * (log_gamma_half + log_gamma_df_half - log_gamma_df_half_plus_half)
    )
    beta_frac = torch.exp(log_beta_ratio)

    # Compute the CRPS for standardized values.
    crps_standard = (
        z * (2 * f_cdf_z - 1) + 2 * f_z * (df + z**2) / (df - 1) - (2 * torch.sqrt(df) / (df - 1)) * beta_frac
    )

    # Apply location-scale transformation CRPS(F_{nu,mu,sigma}, y) = sigma * CRPS(F_nu, z) with z = (y - mu) / sigma.
    crps = scale * crps_standard

    return crps
