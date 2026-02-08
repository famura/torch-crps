import torch

from torch_crps.abstract import crps_abstract


def _accuracy_ensemble(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute accuracy term $A = E[|X - y|]$, i.e., mean absolute error, for an ensemble forecast.

    Args:
        x: The ensemble predictions, of shape (*batch_shape, dim_ensemble).
        y: The ground truth observations, of shape (*batch_shape).

    Returns:
        Accuracy values for each observation, of shape (*batch_shape).
    """
    # Unsqueeze the observation for explicit broadcasting.
    return torch.abs(x - y.unsqueeze(-1)).mean(dim=-1)


def _dispersion_ensemble_naive(
    x: torch.Tensor,
    biased: bool,
) -> torch.Tensor:
    """Compute dispersion term $D = E[|X - X'|]$ for an ensemble forecast using a naive O(m²) algorithm.

    m is the number of ensemble members.

    Args:
        x: The ensemble predictions, of shape (*batch_shape, dim_ensemble).
        biased: If True, uses the biased estimator for the dispersion term $D$, i.e., divides by m². If False, uses the
            unbiased estimator which instead divides by m * (m - 1).

    Returns:
        Dispersion values for each observation, of shape (*batch_shape).
    """
    # Create a matrix of all pairwise differences between ensemble members using broadcasting.
    x_i = x.unsqueeze(-1)  # shape: (*batch_shape, m, 1)
    x_j = x.unsqueeze(-2)  # shape: (*batch_shape, 1, m)
    pairwise_diffs = x_i - x_j  # shape: (*batch_shape, m, m)

    # Take the absolute value of every element in the matrix.
    abs_pairwise_diffs = torch.abs(pairwise_diffs)

    # Calculate the mean of the m x m matrix for each batch item, i.e, not the batch shapes.
    if biased:
        # For the biased estimator, we use the mean which divides by m².
        dispersion = abs_pairwise_diffs.mean(dim=(-2, -1))
    else:
        # For the unbiased estimator, we need to exclude the diagonal (where i=j) and divide by m(m-1).
        m = x.shape[-1]  # number of ensemble members
        dispersion = abs_pairwise_diffs.sum(dim=(-2, -1)) / (m * (m - 1))

    return dispersion


def _dispersion_ensemble(
    x: torch.Tensor,
    biased: bool,
) -> torch.Tensor:
    """Compute dispersion term $D = E[|X - X'|]$ for an ensemble forecast using an efficient O(m log m) algorithm.

    m is the number of ensemble members.

    Args:
        x: The ensemble predictions, of shape (*batch_shape, dim_ensemble).
        biased: If True, uses the biased estimator for the dispersion term $D$, i.e., divides by m². If False, uses the
            unbiased estimator which instead divides by m * (m - 1).

    Returns:
        Dispersion values for each observation, of shape (*batch_shape).
    """
    m = x.shape[-1]  # number of ensemble members

    # Sort the predictions along the ensemble member dimension.
    x_sorted, _ = torch.sort(x, dim=-1)

    # Calculate the coefficients (2i - m - 1) for the linear-time sum. These are the same for every item in the batch.
    coeffs = 2 * torch.arange(1, m + 1, device=x.device, dtype=x.dtype) - m - 1

    # Calculate the sum Σᵢ (2i - m - 1)xᵢ for each forecast in the batch along the member dimension.
    # We use the efficient O(m log m) implementation with a summation over a single dimension.
    x_sum = torch.sum(coeffs * x_sorted, dim=-1)

    # Calculate the full expectation E[|X - X'|] = 2 / m² * Σᵢ (2i - m - 1)xᵢ.
    # This is half the mean absolute difference between all pairs of predictions.
    denom = m * (m - 1) if not biased else m**2
    dispersion = 2 / denom * x_sum

    return dispersion


def crps_ensemble_naive(x: torch.Tensor, y: torch.Tensor, biased: bool = False) -> torch.Tensor:
    """Computes the Continuous Ranked Probability Score (CRPS) for an ensemble forecast.

    This implementation uses the equality

    $$ CRPS(X, y) = E[|X - y|] - 0.5 E[|X - X'|] $$

    It is designed to be fully vectorized and handle any number of leading batch dimensions in the input tensors,
    as long as they are equal for `x` and `y`.

    See Also:
        Zamo & Naveau; "Estimation of the Continuous Ranked Probability Score with Limited Information and Applications
        to Ensemble Weather Forecasts"; 2017

    Note:
        - This implementation uses an inefficient algorithm to compute the term E[|X - X'|] in O(m²) where m is
        the number of ensemble members. This is done for clarity and educational purposes.
        - This implementation exactly matches the energy formula, see (NRG) and (eNRG), in Zamo & Naveau (2017).

    Args:
        x: The ensemble predictions, of shape (*batch_shape, dim_ensemble).
        y: The ground truth observations, of shape (*batch_shape).
        biased: If True, uses the biased estimator for $D$, i.e., divides by m². If False, uses the unbiased estimator.
            The unbiased estimator divides by m * (m - 1).

    Returns:
        The calculated CRPS value for each forecast in the batch, of shape (*batch_shape).
    """
    if x.shape[:-1] != y.shape:
        raise ValueError(f"The batch dimension(s) of x {x.shape[:-1]} and y {y.shape} must be equal!")

    # Accuracy term A := E[|X - y|]
    accuracy = _accuracy_ensemble(x, y)

    # Dispersion term D := E[|X - X'|]
    dispersion = _dispersion_ensemble_naive(x, biased)

    # CRPS value := A - 0.5 * D
    return crps_abstract(accuracy, dispersion)


def crps_ensemble(x: torch.Tensor, y: torch.Tensor, biased: bool = False) -> torch.Tensor:
    r"""Computes the Continuous Ranked Probability Score (CRPS) for an ensemble forecast.

    This implementation uses the equalities

    $$
    CRPS(F, y) = E[|X - y|] - 0.5 E[|X - X'|] = E[|X - y|] + E[X] - 2 E[X F(X)]
    $$

    where $X$ and $X'$ are independent random variables drawn from the ensemble distribution, and $F(X)$ is the CDF
    of the ensemble distribution evaluated at $X$.

    It is designed to be fully vectorized and handle any number of leading batch dimensions in the input tensors,
    as long as they are equal for `x` and `y`.

    See Also:
        Zamo & Naveau; "Estimation of the Continuous Ranked Probability Score with Limited Information and Applications
        to Ensemble Weather Forecasts"; 2017

    Note:
        - This implementation uses an efficient algorithm to compute the dispersion term E[|X - X'|] in O(m log(m))
        time, where m is the number of ensemble members. This is achieved by sorting the ensemble predictions and using
        a mathematical identity to compute the mean absolute difference. You can also see this trick
        [here][https://docs.nvidia.com/physicsnemo/25.11/_modules/physicsnemo/metrics/general/crps.html]

        - This implementation exactly matches the energy formula, see (NRG) and (eNRG), in Zamo & Naveau (2017) while
        using the compuational trick which can be read from (ePWM) in the same paper. The factors &\beta_0$ and
        $\beta_1$ in (ePWM) together equal the second term, i.e., the half mean dispersion, here. In (ePWM) they pulled
        the mean out. The energy formula and the probability weighted moment formula are equivalent.

    Args:
        x: The ensemble predictions, of shape (*batch_shape, dim_ensemble).
        y: The ground truth observations, of shape (*batch_shape).
        biased: If True, uses the biased estimator for the dispersion term $D$, i.e., divides by m². If False, uses the
            unbiased estimator which instead divides by m * (m - 1).

    Returns:
        The calculated CRPS value for each forecast in the batch, of shape (*batch_shape).
    """
    if x.shape[:-1] != y.shape:
        raise ValueError(f"The batch dimension(s) of x {x.shape[:-1]} and y {y.shape} must be equal!")

    # Accuracy term A := E[|X - y|]
    accuracy = _accuracy_ensemble(x, y)

    # Dispersion term D := E[|X - X'|]
    dispersion = _dispersion_ensemble(x, biased)

    # CRPS value := A - 0.5 * D
    return crps_abstract(accuracy, dispersion)
