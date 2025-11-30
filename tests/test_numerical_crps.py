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
@pytest.mark.parametrize(
    "crps_fcn",
    [crps_ensemble_naive, crps_ensemble],
    ids=["naive", "default"],
)
def test_crps_ensemble_smoke(test_case_fixture_name: str, crps_fcn: Callable, request: FixtureRequest):
    """Test that naive numerical method yield."""
    test_case_fixture: dict = request.getfixturevalue(test_case_fixture_name)
    x, y, expected_shape = test_case_fixture["x"], test_case_fixture["y"], test_case_fixture["expected_shape"]

    crps = crps_fcn(x, y)

    assert isinstance(crps, torch.Tensor)
    assert crps.shape == expected_shape, "The output shape is incorrect!"
    assert crps.dtype in [torch.float32, torch.float64], "The output dtype is not float!"
