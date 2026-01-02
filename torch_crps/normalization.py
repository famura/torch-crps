import functools
from typing import Callable, TypeAlias

import torch

WRAPPED_INPUT_TYPE: TypeAlias = torch.distributions.Distribution | torch.Tensor | float


def normalize_by_observation(crps_fcn: Callable) -> Callable:
    """A decorator that normalizes the output of a CRPS function by the absolute maximum of the observations `y`.

    Args:
        crps_fcn: CRPS-calculating function to be wrapped. The fucntion must accept an argument called y which is
            at the 2nd position.

    Returns:
        CRPS-calculating function which is wrapped such that the outputs are normalized by the magnitude of the
            observations.
    """

    @functools.wraps(crps_fcn)
    def wrapper(*args: WRAPPED_INPUT_TYPE, **kwargs: WRAPPED_INPUT_TYPE) -> torch.Tensor:
        """The function returned by the decorator that does the normalization and the forwading to the CRPS function."""
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
        if torch.isclose(abs_max_y, torch.zeros(1, device=abs_max_y.device, dtype=abs_max_y.dtype), atol=1e-5):
            # Avoid division by values close to zero.
            abs_max_y = torch.ones(1, device=abs_max_y.device, dtype=abs_max_y.dtype)

        # Call the original CRPS function.
        crps_result = crps_fcn(*args, **kwargs)

        # Normalize the result.
        return crps_result / abs_max_y

    return wrapper
