import pytest
import torch

from tests.conftest import needs_cuda
from torch_crps import crps_analytical_normal, crps_analytical_studentt, crps_integral


@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_crps_integral_vs_analytical_normal(use_cuda: bool):
    """Test that naive integral method matches the analytical solution for Normal distributions."""
    torch.manual_seed(0)

    # Define 4 independent univariate Normal distributions.
    mu = torch.tensor([0.0, 0.0, 3.0, -7.0], device="cuda" if use_cuda else "cpu")
    sigma = torch.tensor([1.0, 0.01, 1.5, 0.5], device="cuda" if use_cuda else "cpu")
    normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

    # Define observed values, one for each distribution.
    y = torch.tensor([0.5, 0.0, 4.5, -6.0], device="cuda" if use_cuda else "cpu")

    # Compute CRPS.
    crps_naive = crps_integral(normal_dist, y, x_min=-10, x_max=10, x_steps=10001)
    crps_analytical = crps_analytical_normal(normal_dist, y)

    # Print the results for comparison.
    print("Naive integral CRPS:", crps_naive)
    print("Analytical CRPS:", crps_analytical)
    print("Absolute difference:", torch.abs(crps_naive - crps_analytical))

    # Assert that both methods agree within numerical tolerance.
    assert torch.allclose(crps_naive, crps_analytical, atol=1e-3, rtol=5e-4), (
        f"CRPS values do not match: naive={crps_naive}, analytical={crps_analytical}"
    )
    assert crps_naive.device == crps_analytical.device == y.device, "CRPS output device should match input device."


@pytest.mark.parametrize(
    "use_cuda",
    [
        pytest.param(False, id="cpu"),
        pytest.param(True, marks=needs_cuda, id="cuda"),
    ],
)
def test_crps_integral_vs_analytical_studentt(use_cuda: bool):
    """Test that naive integral method matches the analytical solution for StudentT distributions."""
    torch.manual_seed(0)

    # Define 4 independent univariate StudentT distributions.
    df = torch.tensor([100.0, 3.0, 5.0, 5.0], device="cuda" if use_cuda else "cpu")
    mu = torch.tensor([0.0, 0.0, 3.0, -7.0], device="cuda" if use_cuda else "cpu")
    sigma = torch.tensor([1.0, 0.01, 1.5, 0.5], device="cuda" if use_cuda else "cpu")
    studentt_dist = torch.distributions.StudentT(df=df, loc=mu, scale=sigma)

    # Define observed values, one for each distribution.
    y = torch.tensor([0.5, 0.0, 4.5, -6.0], device="cuda" if use_cuda else "cpu")

    # Compute CRPS.
    crps_naive = crps_integral(studentt_dist, y, x_min=-10, x_max=10, x_steps=10001)
    crps_analytical = crps_analytical_studentt(studentt_dist, y)

    # Print the results for comparison.
    print("Naive integral CRPS:", crps_naive)
    print("Analytical CRPS:", crps_analytical)
    print("Absolute difference:", torch.abs(crps_naive - crps_analytical))

    # Assert that both methods agree within numerical tolerance.
    assert torch.allclose(crps_naive, crps_analytical, atol=1e-3, rtol=5e-4), (
        f"CRPS values do not match: naive={crps_naive}, analytical={crps_analytical}"
    )
    assert crps_naive.device == crps_analytical.device == y.device, "CRPS output device should match input device."
