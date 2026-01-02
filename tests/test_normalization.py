from typing import Callable

import pytest
import torch

from torch_crps import (
    crps_analytical_normal_normalized,
    crps_analytical_normalized,
    crps_analytical_studentt_normalized,
    crps_ensemble_normalized,
    crps_integral_normalized,
)


@pytest.mark.parametrize(
    "wrapped_crps_fcn",
    [
        crps_analytical_normal_normalized,
        crps_analytical_normalized,
        crps_analytical_studentt_normalized,
        crps_ensemble_normalized,
        crps_integral_normalized,
    ],
    ids=[
        "crps_analytical_normal_normalized",
        "crps_analytical_normalized",
        "crps_analytical_studentt_normalized",
        "crps_ensemble_normalized",
        "crps_integral_normalized",
    ],
)
def test_nomrmalization_wrapper_input_errors(wrapped_crps_fcn: Callable, num_y: int = 3):
    """Test if the normalization wrapper handles the underlying function's arguments correctly."""
    torch.manual_seed(0)

    # Set up test cases.
    if wrapped_crps_fcn in (crps_analytical_normal_normalized, crps_analytical_normalized, crps_integral_normalized):
        q = torch.distributions.Normal(loc=torch.zeros(num_y), scale=torch.ones(num_y))
    elif wrapped_crps_fcn == crps_analytical_studentt_normalized:
        q = torch.distributions.StudentT(df=5 * torch.ones(num_y), loc=torch.zeros(num_y), scale=torch.ones(num_y))
    elif wrapped_crps_fcn == crps_ensemble_normalized:
        q = torch.randn(num_y, 10)  # dim_ensemble = 10
    else:
        raise NotImplementedError("Test case setup error.")
    y = torch.randn(num_y)

    # Case 1: y not supplied at all.
    with pytest.raises(TypeError):
        wrapped_crps_fcn(q)

    # Case 2: y supplied at wrong position (only q supplied as first arg, nothing at position 1)
    # This is essentially the same as case 1 for these functions.
    with pytest.raises(TypeError):
        wrapped_crps_fcn()

    # Case 3: y supplied but not as a tensor.
    with pytest.raises(TypeError):
        wrapped_crps_fcn(q, [0.0, 1.0, 2.0])
    with pytest.raises(TypeError):
        wrapped_crps_fcn(q, 1.0)
    with pytest.raises(TypeError):
        wrapped_crps_fcn(q, y="not a tensor")

    # Verify that valid input works.
    ncrps = wrapped_crps_fcn(q, y)
    assert isinstance(ncrps, torch.Tensor)


@pytest.mark.parametrize(
    "wrapped_crps_fcn",
    [
        crps_analytical_normal_normalized,
        crps_analytical_normalized,
        crps_analytical_studentt_normalized,
        crps_ensemble_normalized,
        crps_integral_normalized,
    ],
    ids=[
        "crps_analytical_normal_normalized",
        "crps_analytical_normalized",
        "crps_analytical_studentt_normalized",
        "crps_ensemble_normalized",
        "crps_integral_normalized",
    ],
)
@pytest.mark.parametrize("num_y", [1, 5, 100], ids=["1_obs", "5_obs", "100_obs"])
def test_nomrmalization_wrapper_output_consistency(wrapped_crps_fcn: Callable, num_y: int):
    """Test if the normalization wrapper results in normalized CRPS values."""
    torch.manual_seed(0)

    # Set up test cases.
    if wrapped_crps_fcn in (crps_analytical_normal_normalized, crps_analytical_normalized, crps_integral_normalized):
        q = torch.distributions.Normal(loc=torch.zeros(num_y), scale=torch.ones(num_y))
    elif wrapped_crps_fcn == crps_analytical_studentt_normalized:
        q = torch.distributions.StudentT(df=5 * torch.ones(num_y), loc=torch.zeros(num_y), scale=torch.ones(num_y))
    elif wrapped_crps_fcn == crps_ensemble_normalized:
        q = torch.randn(num_y, 10)  # dim_ensemble = 10
    else:
        raise NotImplementedError("Test case setup error.")

    # Calculate the CRPS and verify that it is normalized. The first case is all zeros.
    for i in range(0, 12, 2):
        y = i * torch.randn(num_y)
        ncrps = wrapped_crps_fcn(q, y)
        assert torch.all(ncrps >= 0)
        assert torch.all(ncrps <= 1)
