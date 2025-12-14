from collections.abc import Callable

import pytest
import torch
from _pytest.fixtures import FixtureRequest

from torch_crps import crps_ensemble, crps_ensemble_naive


@pytest.mark.parametrize(
    "test_case_fixture_name",
    ["case_flat_1d", "case_batched_2d", "case_batched_3d"],
    ids=["case_flat_1d", "case_batched_2d", "case_batched_3d"],
)
@pytest.mark.parametrize("crps_fcn", [crps_ensemble_naive, crps_ensemble], ids=["naive", "default"])
@pytest.mark.parametrize("biased", [True, False], ids=["biased", "unbiased"])
def test_crps_ensemble_smoke(test_case_fixture_name: str, crps_fcn: Callable, biased: bool, request: FixtureRequest):
    """Test that naive ensemble method yield."""
    test_case_fixture: dict = request.getfixturevalue(test_case_fixture_name)
    x, y, expected_shape = test_case_fixture["x"], test_case_fixture["y"], test_case_fixture["expected_shape"]

    crps = crps_fcn(x, y, biased)

    assert isinstance(crps, torch.Tensor)
    assert crps.shape == expected_shape, "The output shape is incorrect!"
    assert crps.dtype in [torch.float32, torch.float64], "The output dtype is not float!"


@pytest.mark.parametrize(
    "batch_shape",
    [(), (3,), (3, 5)],
    ids=["case_flat_1d", "case_batched_2d", "case_batched_3d"],
)
@pytest.mark.parametrize("biased", [True, False], ids=["biased", "unbiased"])
def test_crps_ensemble_match(batch_shape: tuple[int, ...], biased: bool, dim_ensemble: int = 10):
    """Test that both implementations of crps_ensemble yield the same result."""
    torch.manual_seed(0)

    # Create a random ensemble forecast and observation.
    if len(batch_shape) > 0:
        x = torch.randn(*batch_shape, dim_ensemble)
        y = torch.randn(*batch_shape)
    else:
        x = torch.randn(dim_ensemble)
        y = torch.randn(batch_shape)

    crps_naive = crps_ensemble_naive(x, y, biased)
    crps_default = crps_ensemble(x, y, biased)

    # Assert that both methods agree within numerical tolerance.
    assert torch.allclose(crps_naive, crps_default, atol=1e-8, rtol=1e-6), (
        f"CRPS values do not match: naive={crps_naive}, default={crps_default}"
    )


def test_crps_ensemble_invalid_shapes(dim_ensemble: int = 10):
    """Test that crps_ensemble raises an error for invalid input shapes."""
    # Mismatch in the number of batch dimensions.
    x = torch.randn(2, 3, dim_ensemble)
    y = torch.randn(3)
    with pytest.raises(ValueError):
        crps_ensemble(x, y)

    # Mismatch in batch dimension sizes.
    x = torch.randn(4, 5, dim_ensemble)
    y = torch.randn(4, 6)
    with pytest.raises(ValueError):
        crps_ensemble(x, y)
