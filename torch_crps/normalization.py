import functools
from typing import Callable, TypeAlias

import torch

from torch_crps.analytical.dispatch import crps_analytical
from torch_crps.analytical.normal import crps_analytical_normal
from torch_crps.analytical.studentt import crps_analytical_studentt
from torch_crps.ensemble import crps_ensemble
from torch_crps.integral import crps_integral

WRAPPED_INPUT_TYPE: TypeAlias = torch.distributions.Distribution | torch.Tensor | float


def normalize_by_observation(crps_fcn: Callable) -> Callable:
    """A decorator that normalizes the output of a CRPS function by the absolute maximum of the observations `y`.

    Note:
        - The resulting value is not guaranteed to be <= 1, because the (original) CRPS value can be larger than the
        normalization factor computed from the observations `y`.
        - If the observations `y` are all close to zero, then the normalization is done by 1, so the CRPS can be > 1.

    Args:
        crps_fcn: CRPS-calculating function to be wrapped. The function must accept an argument called y which is
            at the 2nd position.

    Returns:
        CRPS-calculating function which is wrapped such that the outputs are normalized by the magnitude of the
            observations.
    """

    @functools.wraps(crps_fcn)
    def wrapper(*args: WRAPPED_INPUT_TYPE, **kwargs: WRAPPED_INPUT_TYPE) -> torch.Tensor:
        """The function returned by the decorator that normalizes and forwards to the CRPS function."""
        # Find the observation 'y' from the arguments.
        if "y" in kwargs:
            y = kwargs["y"]
        elif len(args) < 2:
            raise TypeError("The observation `y` was not found in the function arguments as there is only one.")
        elif args:
            y = args[1]
        else:
            raise TypeError("The observation `y` was not found in the function arguments.")

        # Validate that y is a tenor.
        if not isinstance(y, torch.Tensor):
            raise TypeError("The observation `y` was found in the function arguments, but is not of type torch.Tensor!")

        # Calculate the normalization factor.
        abs_max_y = y.abs().max()
        if torch.isclose(abs_max_y, torch.zeros(1, device=abs_max_y.device, dtype=abs_max_y.dtype), atol=1e-6):
            # Avoid division by values close to zero.
            abs_max_y = torch.ones(1, device=abs_max_y.device, dtype=abs_max_y.dtype)

        # Call the original CRPS function.
        crps = crps_fcn(*args, **kwargs)

        # Normalize the result.
        return crps / abs_max_y

    return wrapper


crps_analytical_obsnormalized = normalize_by_observation(crps_analytical)
crps_analytical_normal_obsnormalized = normalize_by_observation(crps_analytical_normal)
crps_analytical_studentt_obsnormalized = normalize_by_observation(crps_analytical_studentt)
crps_ensemble_obsnormalized = normalize_by_observation(crps_ensemble)
crps_integral_obsnormalized = normalize_by_observation(crps_integral)
