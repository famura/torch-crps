from typing import Any

import pytest
import torch
from torch.distributions import Normal, StudentT

from tests.conftest import needs_cuda
from torch_crps import crps_analytical, crps_analytical_normal, crps_analytical_studentt


@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_crps_analytical_normal_batched_smoke(use_cuda: bool):
    """Test that analytical solution works with batched Normal distributions."""
    torch.manual_seed(0)

    # Define a batch of 2 independent univariate Normal distributions.
    mu = torch.tensor([[0.0, 1.0], [2.0, 3.0], [-2.0, -3.0]], device="cuda" if use_cuda else "cpu")
    sigma = torch.tensor([[1.0, 0.5], [1.5, 2.0], [0.01, 0.01]], device="cuda" if use_cuda else "cpu")
    normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

    # Define observed values for each distribution in the batch.
    y = torch.tensor([[0.5, 1.5], [2.5, 3.5], [-2.0, -3.0]], device="cuda" if use_cuda else "cpu")

    # Compute CRPS using the analytical method.
    crps_analytical = crps_analytical_normal(normal_dist, y)

    # Simple sanity check: CRPS should be non-negative.
    assert crps_analytical.shape == y.shape, "CRPS output shape should match input shape."
    assert crps_analytical.dtype in [torch.float32, torch.float64], "CRPS output dtype should be float."
    assert crps_analytical.device == y.device, "CRPS output device should match input device."
    assert torch.all(crps_analytical >= 0), "CRPS values should be non-negative."


@pytest.mark.parametrize(
    "loc, scale",
    [
        (torch.tensor(0.0), torch.tensor(1.0)),
        (torch.tensor(2.0), torch.tensor(0.5)),
        (torch.tensor(-5.0), torch.tensor(10.0)),
    ],
    ids=["standard", "shifted_scaled", "neg-mean_large-var"],
)
@pytest.mark.parametrize("y", [torch.tensor([-10.0, -1.0, 0.0, 0.5, 2.0, 5.0])])
def test_studentt_convergence_to_normal(loc: torch.Tensor, scale: torch.Tensor, y: torch.Tensor):
    """Test that for a very high degrees of freedom, the StudentT CRPS converges to the Normal CRPS.
    This validates both implementations against each other.
    """
    # Create the two distributions with identical parameters.
    high_df = torch.tensor(1000.0)
    q_studentt = StudentT(df=high_df, loc=loc, scale=scale)
    q_normal = Normal(loc=loc, scale=scale)

    # Calculate the analytical CRPS for both.
    crps_studentt = crps_analytical_studentt(q_studentt, y)
    crps_normal = crps_analytical_normal(q_normal, y)

    # Assert that their results are nearly identical.
    assert torch.allclose(crps_studentt, crps_normal, atol=2e-3), (
        "StudentT CRPS with high 'df' should match Normal CRPS."
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
def test_crps_analytical_interface_smoke(q: Any):  # noqa: ANN401
    """Test if the top-level interface function is working"""
    y = torch.zeros(3)  # can be the same for all tests

    if isinstance(q, (Normal, StudentT)):
        # Supported, should return a result.
        crps = crps_analytical(q, y)
        assert isinstance(crps, torch.Tensor)

    else:
        # Not supported, should raise an error.
        with pytest.raises(NotImplementedError):
            crps_analytical(q, y)
