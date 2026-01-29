import torch
from torch.distributions import Distribution, Normal, StudentT


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
    """Compute the (negatively-oriented, i.e., lower is better) scaled CRPS (SCRPS) in closed-form.

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
    # Compute standard normal CDF and PDF.
    z = (y - q.loc) / q.scale  # standardize
    standard_normal = torch.distributions.Normal(0, 1)
    cdf_z = standard_normal.cdf(z)  # Φ(z)
    pdf_z = torch.exp(standard_normal.log_prob(z))  # φ(z)

    # Analytical CRPS formula.
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=z.device, dtype=z.dtype))
    crps = q.scale * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / sqrt_pi)

    return crps


def scrps_analytical_normal(
    q: Normal,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the (negatively-oriented) scaled CRPS (SCRPS) in closed-form assuming a normal distribution.

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
    # --- Dispersion Term D := E[|X - X'|] = 2σ / √π
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=y.device, dtype=y.dtype))
    dispersion = 2 * q.scale / sqrt_pi

    # --- Accuracy Term A := E[|X - y|]
    z = (y - q.loc) / q.scale  # standardize
    standard_normal = torch.distributions.Normal(0, 1)
    cdf_z = standard_normal.cdf(z)  # Φ(z)
    pdf_z = torch.exp(standard_normal.log_prob(z))  # φ(z)
    accuracy = q.scale * (z * (2 * cdf_z - 1) + 2 * pdf_z)

    # --- SCRPS (negatively-oriented) := (A / D) + 0.5 * log(D)
    scrps = accuracy / dispersion + 0.5 * torch.log(dispersion)

    return scrps


def standardized_studentt_cdf_via_scipy(
    z: torch.Tensor,
    df: torch.Tensor | float,
) -> torch.Tensor:
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
    try:
        from scipy.stats import t as scipy_student_t
    except ImportError as e:
        raise ImportError(
            "scipy is required for the analytical solution for the StudentT distribution. "
            "Install `torch-crps` with the 'studentt' dependency group, e.g. `pip install torch-crps[studentt]`."
        ) from e

    z_np = z.detach().cpu().numpy()
    df_np = df.detach().cpu().numpy() if isinstance(df, torch.Tensor) else df

    cdf_z_np = scipy_student_t.cdf(z_np, df=df_np)

    return torch.from_numpy(cdf_z_np).to(device=z.device, dtype=z.dtype)


def crps_analytical_studentt(
    q: StudentT,
    y: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the (negatively-oriented) CRPS in closed-form assuming a StudentT distribution.

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
    # Extract degrees of freedom ν, location μ, and scale σ.
    nu, mu, sigma = q.df, q.loc, q.scale
    if torch.any(nu <= 1):
        raise ValueError("StudentT CRPS requires degrees of freedom > 1")

    # Standardize, and create standard StudentT distribution for CDF and PDF.
    z = (y - mu) / sigma
    standard_t = torch.distributions.StudentT(nu, loc=0, scale=1)

    # Compute standardized CDF F_ν(z) and PDF f_ν(z).
    f_cdf_z = standardized_studentt_cdf_via_scipy(z, nu)
    f_z = torch.exp(standard_t.log_prob(z))

    # Compute the beta function ratio: B(1/2, ν - 1/2) / B(1/2, ν/2)^2
    # Using the relationship: B(a,b) = Gamma(a) * Gamma(b) / Gamma(a+b)
    # B(1/2, ν - 1/2) / B(1/2, ν/2)^2 = ( Gamma(1/2) * Gamma(ν-1/2) / Gamma(ν) ) /
    #                                     ( Gamma(1/2) * Gamma(ν/2) / Gamma(ν/2 + 1/2) )^2
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
        z * (2 * f_cdf_z - 1) + 2 * f_z * (nu + z**2) / (nu - 1) - (2 * torch.sqrt(nu) / (nu - 1)) * beta_frac
    )

    # Apply location-scale transformation CRPS(F_{ν,μ,σ}, y) = σ * CRPS(F_{ν}, z) with z = (y - μ) / σ.
    crps = sigma * crps_standard

    return crps


def scrps_analytical_studentt(
    q: StudentT,
    y: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the (negatively-oriented) scaled CRPS (SCRPS) in closed-form assuming a Student-T distribution.

    The score is calculated as:
    $$ \text{SCRPS}(F, y) = \frac{A}{D} + 0.5 \cdot \log(D) $$

    where:
    - $A = E_F[|X - y|]$ is the Accuracy term.
    - $D = E_F[|X - X'|]$ is the Dispersion term.
    - $F$ is the Student-T distribution $t(\nu, \mu, \sigma^2)$.

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
    # Extract degrees of freedom ν, location μ, and scale σ.
    nu, mu, sigma = q.df, q.loc, q.scale
    if torch.any(nu <= 1):
        raise ValueError("StudentT SCRPS requires degrees of freedom > 1")

    # Use the device of y for creating new (intermediate) tensors.
    device, dtype = y.device, y.dtype

    # --- Dispersion Term D := E[|X - X'|] = (4σ / (ν-1)) * (Γ(ν/2) / Γ((ν-1)/2))²
    # We compute in log space for numerical stability.
    log_4 = torch.log(torch.tensor(4.0, dtype=dtype, device=device))
    log_dispersion = (
        log_4 + torch.log(sigma) - torch.log(nu - 1) + 2 * (torch.lgamma(nu / 2) - torch.lgamma((nu - 1) / 2))
    )
    dispersion = torch.exp(log_dispersion)

    # --- 2. Accuracy Term A := E[|X - y|]
    # Standardize, and create standard StudentT distributions for CDFs.
    z = (y - mu) / sigma
    standard_t_nu = StudentT(nu, loc=0, scale=1)
    standard_t_nu_plus_1 = StudentT(nu + 1, loc=0, scale=1)

    # Compute Beta function term B(ν/2, 1/2)
    lgamma_half = torch.lgamma(torch.tensor(0.5, dtype=nu.dtype, device=nu.device))
    log_beta_term = torch.lgamma(nu / 2) + lgamma_half - torch.lgamma((nu + 1) / 2)
    beta_term = torch.exp(log_beta_term)

    # Compute components of the 'A' formula from Bolin & Wallin Appendix A.2
    term_A1 = z * (2 * standard_t_nu.cdf(z) - 1)

    term_A2_factor = (2 * (nu + z**2)) / (nu * beta_term)
    term_A2_cdf_arg = z * torch.sqrt((nu + 1) / (nu + z**2))
    term_A2 = term_A2_factor * standard_t_nu_plus_1.cdf(term_A2_cdf_arg)

    accuracy = sigma * (term_A1 + term_A2)

    # --- 3. SCRPS (negatively-oriented) := (A / D) + 0.5 * log(D)
    scrps = accuracy / dispersion + 0.5 * log_dispersion
    return scrps
