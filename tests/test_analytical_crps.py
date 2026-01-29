from typing import Any, Callable

import pytest
import torch
from torch.distributions import Normal, StudentT

from tests.conftest import needs_cuda
from torch_crps import (
    crps_analytical,
    crps_analytical_normal,
    scrps_analytical,
    scrps_analytical_normal,
)


@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
@pytest.mark.parametrize("crps_fcn", [crps_analytical_normal, scrps_analytical_normal], ids=["CRPS", "SCRPS"])
def test_analytical_normal_batched_smoke(use_cuda: bool, crps_fcn: Callable[..., torch.Tensor]):
    """Test that analytical solution works with batched Normal distributions."""
    torch.manual_seed(0)

    # Define a batch of 2 independent univariate Normal distributions.
    mu = torch.tensor([[0.0, 1.0], [2.0, 3.0], [-2.0, -3.0]], device="cuda" if use_cuda else "cpu")
    sigma = torch.tensor([[1.0, 0.5], [1.5, 2.0], [0.01, 0.01]], device="cuda" if use_cuda else "cpu")
    normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

    # Define observed values for each distribution in the batch.
    y = torch.tensor([[0.5, 1.5], [2.5, 3.5], [-2.0, -3.0]], device="cuda" if use_cuda else "cpu")

    # Compute CRPS using the analytical method.
    crps_analytical = crps_fcn(normal_dist, y)

    # Simple sanity check: CRPS should be non-negative.
    assert crps_analytical.shape == y.shape, "CRPS output shape should match input shape."
    assert crps_analytical.dtype in [torch.float32, torch.float64], "CRPS output dtype should be float."
    assert crps_analytical.device == y.device, "CRPS output device should match input device."
    if crps_fcn == crps_analytical_normal:
        assert torch.all(crps_analytical >= 0), "CRPS values should be non-negative."


@pytest.mark.parametrize(
    "loc, scale",
    [
        (torch.tensor(0.0), torch.tensor(1.0)),
        (torch.tensor(-1.0), torch.tensor(0.5)),
        (torch.tensor(1.0), torch.tensor(0.5)),
        (torch.tensor(10.0), torch.tensor(20.0)),
        (torch.tensor(-10.0), torch.tensor(20.0)),
        (torch.tensor(100.0), torch.tensor(5.0)),
        (torch.tensor(-100.0), torch.tensor(5.0)),
    ],
    ids=[
        "standard",
        "small-neg-mean_small-var",
        "small-pos-mean_small-var",
        "pos-mean_large-var",
        "neg-mean_large-var",
        "large-mean_medium-var",
        "large-neg-mean_medium-var",
    ],
)
@pytest.mark.parametrize("y", [torch.tensor([-95.0, -80.0, -1.0, 0.0, 0.5, 2.0, 5.0, 50.0])])
def test_studentt_convergence_to_normal(loc: torch.Tensor, scale: torch.Tensor, y: torch.Tensor, atol: float = 3e-3):
    """Test that for a high degrees of freedom, the StudentT score converges to the Normal score
    when their standard deviations are matched.

    Note:
        This test only works for the CRPS. For the SCRPS, the differences are too big.
    """
    # Create the StudentT distribution with a high degree of freedom.
    high_df = torch.tensor(1000.0)
    q_studentt = StudentT(df=high_df, loc=loc, scale=scale)

    # Calculate the standard deviation of the StudentT distribution. The variance is (df / (df - 2)) * scale^2
    student_t_std_dev = scale * torch.sqrt(high_df / (high_df - 2))

    # Create the Normal distribution with matching standard deviation.
    q_normal = Normal(loc=loc, scale=student_t_std_dev)

    # Calculate the analytical scores for both.
    score_value_studentt = crps_analytical(q_studentt, y)
    score_value_normal = crps_analytical(q_normal, y)

    # Assert that their results are nearly identical.
    assert torch.allclose(score_value_studentt, score_value_normal, atol=atol), (
        f"StudentT CRPS with high 'df' should match Normal CRPS with atol={atol}."
    )


@pytest.mark.parametrize(
    "q",
    [
        torch.distributions.Normal(loc=torch.zeros(3), scale=torch.ones(3)),
        torch.distributions.StudentT(df=5, loc=torch.zeros(3), scale=torch.ones(3)),
        "NOT_A_SUPPORTED_DISTRIBUTION",
    ],
    ids=["Normal", "StudentT", "not_supported"],
)
@pytest.mark.parametrize("crps_fcn", [crps_analytical, scrps_analytical], ids=["CRPS", "SCRPS"])
def test_analytical_interface_smoke(q: Any, crps_fcn: Callable[..., torch.Tensor]):  # noqa: ANN401
    """Test if the top-level interface function is working"""
    y = torch.zeros(3)  # can be the same for all tests

    if isinstance(q, (Normal, StudentT)):
        # Supported, should return a result.
        crps = crps_fcn(q, y)
        assert isinstance(crps, torch.Tensor)

    else:
        # Not supported, should raise an error.
        with pytest.raises(NotImplementedError):
            crps_fcn(q, y)
