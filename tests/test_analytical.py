from typing import Any, Callable

import pytest
import torch
from torch.distributions import Normal, StudentT
from typing_extensions import Literal

from tests.conftest import crps_analytical_normal_gneiting, crps_analytical_studentt_jordan, needs_cuda
from torch_crps.analytical import crps_analytical, scrps_analytical
from torch_crps.analytical.normal import crps_analytical_normal, scrps_analytical_normal
from torch_crps.analytical.studentt import (
    crps_analytical_studentt,
    scrps_analytical_studentt,
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
@pytest.mark.parametrize("y", [torch.tensor([-100.0, -10.0, -1.0, 0.0, 0.5, 2.0, 5.0, 50.0])])
@pytest.mark.parametrize("crps_fcn_type", ["CRPS", "SCRPS"], ids=["CRPS", "SCRPS"])
def test_studentt_convergence_to_normal(
    loc: torch.Tensor, scale: torch.Tensor, y: torch.Tensor, crps_fcn_type: Literal["CRPS", "SCRPS"]
):
    """Test that for a high degrees of freedom, the StudentT score converges to the Normal score
    when their standard deviations are matched.
    """
    # Create the StudentT distribution with a high degree of freedom.
    high_df = torch.tensor(1000.0)
    q_studentt = StudentT(df=high_df, loc=loc, scale=scale)

    # Calculate the standard deviation of the StudentT distribution. The variance is (df / (df - 2)) * scale^2
    student_t_std_dev = scale * torch.sqrt(high_df / (high_df - 2))

    # Create the Normal distribution with matching standard deviation.
    q_normal = Normal(loc=loc, scale=student_t_std_dev)

    # Calculate the analytical scores for both.
    if crps_fcn_type == "CRPS":
        score_value_studentt = crps_analytical(q_studentt, y)
        score_value_normal = crps_analytical(q_normal, y)
    else:
        score_value_studentt = scrps_analytical(q_studentt, y)
        score_value_normal = scrps_analytical(q_normal, y)

    # Assert that their results are nearly identical.
    # The tolerance can be quite tight now.
    atol = 6e-3 if crps_fcn_type == "CRPS" else 2e-2
    assert torch.allclose(score_value_studentt, score_value_normal, atol=atol), (
        f"StudentT {crps_fcn_type} with high 'df' should match Normal {crps_fcn_type}."
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


def test_analytical_crps_normal_consistency():
    """Test if the two ways to compute the CRPS for normal distributions give the same result:

    - old method: `crps_analytical_normal_gneiting`
    - new method: `_accuracy_normal_gneiting` and `_dispersion_normal_gneiting` packaged in `crps_analytical_normal`
    """
    torch.manual_seed(0)

    # Create a Normal distribution.
    loc = torch.tensor([0.0, 1.0, -1.0])
    scale = torch.tensor([1.0, 2.0, 0.5])
    normal_dist = torch.distributions.Normal(loc=loc, scale=scale)

    # Define observed values.
    y = torch.tensor([0.5, 2.0, -0.5])

    # Compute CRPS values.
    crps_old = crps_analytical_normal_gneiting(normal_dist, y)
    crps_new = crps_analytical_normal(normal_dist, y)

    # Assert that both methods give the same result.
    assert torch.allclose(crps_old, crps_new, atol=1e-6), "CRPS values from both methods should match."


def test_analytical_crps_studentt_consistency():
    """Test if the two ways to compute the CRPS for StudentT distributions give the same result:

    - old method: `_crps_analytical_studentt_jordan`
    - new method: `_accuracy_studentt_jordan` and `_dispersion_studentt_jordan` packaged in `crps_analytical_studentt`
    """
    torch.manual_seed(0)

    # Create a StudentT distribution.
    df = torch.tensor([3.0, 5.0, 10.0])
    loc = torch.tensor([0.0, 1.0, -1.0])
    scale = torch.tensor([1.0, 2.0, 0.5])
    studentt_dist = torch.distributions.StudentT(df=df, loc=loc, scale=scale)

    # Define observed values.
    y = torch.tensor([0.5, 2.0, -0.5])

    # Compute CRPS values.
    crps_old = crps_analytical_studentt_jordan(studentt_dist, y)
    crps_new = crps_analytical_studentt(studentt_dist, y)

    # Assert that both methods give the same result.
    assert torch.allclose(crps_old, crps_new, atol=1e-6), "CRPS values from both methods should match."


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(torch.tensor(0.5), id="df=0.5"),
        pytest.param(torch.tensor(1.0), id="df=1.0"),
        pytest.param(torch.tensor([2.0, 0.9, 3.0]), id="batch_one_below_1"),
        pytest.param(torch.tensor([1.0, 2.0]), id="batch_one_at_1"),
    ],
)
@pytest.mark.parametrize("crps_fcn", [crps_analytical_studentt, scrps_analytical_studentt], ids=["CRPS", "SCRPS"])
def test_studentt_df_leq_1_raises(df: torch.Tensor, crps_fcn: Callable[..., torch.Tensor]):
    """Test that both studentt score functions raise ValueError when any df <= 1."""
    q = StudentT(df=df, loc=torch.zeros_like(df), scale=torch.ones_like(df))
    y = torch.zeros_like(df)

    with pytest.raises(ValueError, match="degrees of freedom > 1"):
        crps_fcn(q, y)
