import torch


def crps_ensemble_naive(x: torch.Tensor, y: torch.Tensor, biased: bool = True) -> torch.Tensor:
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
        biased: If True, uses the biased estimator for E[|X - X'|]. If False, uses the unbiased estimator.
            The unbiased estimator divides by m * (m - 1) instead of m².

    Returns:
        The calculated CRPS value for each forecast in the batch, of shape (*batch_shape).
    """
    if x.shape[:-1] != y.shape:
        raise ValueError(f"The batch dimension(s) of x {x.shape[:-1]} and y {y.shape} must be equal!")

    # --- Accuracy term := E[|X - y|]

    # Compute the mean absolute error across all ensemble members. Unsqueeze the observation for explicit broadcasting.
    mae = torch.abs(x - y.unsqueeze(-1)).mean(dim=-1)

    # --- Spread term := 0.5 * E[|X - X'|]
    # This is half the mean absolute difference between all pairs of predictions.

    # Create a matrix of all pairwise differences between ensemble members using broadcasting.
    x_i = x.unsqueeze(-1)  # shape: (*batch_shape, m, 1)
    x_j = x.unsqueeze(-2)  # shape: (*batch_shape, 1, m)
    pairwise_diffs = x_i - x_j  # shape: (*batch_shape, m, m)

    # Take the absolute value of every element in the matrix.
    abs_pairwise_diffs = torch.abs(pairwise_diffs)

    # Calculate the mean of the m x m matrix for each batch item, i.e, not the batch shapes.
    if biased:
        # For the biased estimator, we use the mean which divides by m².
        mean_spread = abs_pairwise_diffs.mean(dim=(-2, -1))
    else:
        # For the unbiased estimator, we need to exclude the diagonal (where i=j) and divide by m(m-1).
        m = x.shape[-1]  # number of ensemble members
        mean_spread = abs_pairwise_diffs.sum(dim=(-2, -1)) / (m * (m - 1))

    # --- Assemble the final CRPS value.
    crps_value = mae - 0.5 * mean_spread

    return crps_value


def crps_ensemble(x: torch.Tensor, y: torch.Tensor, biased: bool = True) -> torch.Tensor:
    r"""Computes the Continuous Ranked Probability Score (CRPS) for an ensemble forecast.

    This implementation uses the equalities

    $$ CRPS(F, y) = E[|X - y|] - 0.5 E[|X - X'|] $$

    and

    $$ CRPS(F, y) = E[|X - y|] + E[X] - 2 E[X F(X)] $$

    It is designed to be fully vectorized and handle any number of leading batch dimensions in the input tensors,
    as long as they are equal for `x` and `y`.

    See Also:
        Zamo & Naveau; "Estimation of the Continuous Ranked Probability Score with Limited Information and Applications
        to Ensemble Weather Forecasts"; 2017

    Note:
        - This implementation uses an efficient algorithm to compute the term E[|X - X'|] in O(m log(m)) time, where m
        is the number of ensemble members. This is achieved by sorting the ensemble predictions and using a mathematical
        identity to compute the mean absolute difference. You can also see this trick
        [here][https://docs.nvidia.com/physicsnemo/25.11/_modules/physicsnemo/metrics/general/crps.html]
        - This implementation exactly matches the energy formula, see (NRG) and (eNRG), in Zamo & Naveau (2017) while
        using the compuational trick which can be read from (ePWM) in the same paper. The factors &\beta_0$ and
        $\beta_1$ in (ePWM) together equal the second term, i.e., the half mean spread, here. In (ePWM) they pulled
        the mean out. The energy formula and the probability weighted moment formula are equivalent.

    Args:
        x: The ensemble predictions, of shape (*batch_shape, dim_ensemble).
        y: The ground truth observations, of shape (*batch_shape).
        biased: If True, uses the biased estimator for E[|X - X'|]. If False, uses the unbiased estimator.
            The unbiased estimator divides by m * (m - 1) instead of m².

    Returns:
        The calculated CRPS value for each forecast in the batch, of shape (*batch_shape).
    """
    if x.shape[:-1] != y.shape:
        raise ValueError(f"The batch dimension(s) of x {x.shape[:-1]} and y {y.shape} must be equal!")

    # Get the number of ensemble members.
    m = x.shape[-1]

    # --- Accuracy term := E[|X - y|]

    # Compute the mean absolute error across all ensemble members. Unsqueeze the observation for explicit broadcasting.
    mae = torch.abs(x - y.unsqueeze(-1)).mean(dim=-1)

    # --- Spread term B := 0.5 * E[|X - X'|]
    # This is half the mean absolute difference between all pairs of predictions.
    # We use the efficient O(m log m) implementation with a summation over a single dimension.

    # Sort the predictions along the ensemble member dimension.
    x_sorted, _ = torch.sort(x, dim=-1)

    # Calculate the coefficients (2i - m - 1) for the linear-time sum. These are the same for every item in the batch.
    coeffs = 2 * torch.arange(1, m + 1, device=x.device, dtype=x.dtype) - m - 1

    # Calculate the sum Σᵢ (2i - m - 1)xᵢ for each forecast in the batch along the member dimension.
    x_sum = torch.sum(coeffs * x_sorted, dim=-1)

    # Calculate the full expectation E[|X - X'|] = 2 / m² * Σᵢ (2i - m - 1)xᵢ.
    denom = m * (m - 1) if not biased else m**2
    half_mean_spread = 1 / denom * x_sum  # 2 in numerator here cancels with 0.5 in the next step

    # --- Assemble the final CRPS value.
    crps_value = mae - half_mean_spread  # 0.5 already accounted for above

    return crps_value
