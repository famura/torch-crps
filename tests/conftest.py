from pathlib import Path

import pytest
import torch

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Check if CUDA support is available.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not supported in this setup.")


@pytest.fixture
def case_flat_1d():
    """Fixture for a simple 1D example with a scalar output, to be used with the ensemble methods."""
    return {
        "x": torch.tensor([12.0, 15.0, 16.0, 21.0]),  # only 1 forecast
        "y": torch.tensor(14.5),  # only 1 observation
        "expected_shape": torch.Size([]),
    }


@pytest.fixture
def case_batched_2d():
    """Fixture for a batched 2D example, to be used with the ensemble methods."""
    return {
        "x": torch.tensor(
            [
                [12.0, 15.0, 16.0, 21.0],  # forecast 1
                [30.0, 31.0, 33.0, 38.0],  # forecast 2
            ]
        ),
        "y": torch.tensor(
            [
                14.5,  # observation 1
                35.0,  # observation 2
            ]
        ),
        "expected_shape": torch.Size([2]),
    }


@pytest.fixture
def case_batched_3d():
    """Fixture for a complex 3D example, to be used with the ensemble methods."""
    torch.manual_seed(42)
    return {
        "x": torch.randn(2, 3, 5) * 10 + 50,
        "y": torch.randn(2, 3) * 10 + 50,
        "expected_shape": torch.Size([2, 3]),
    }
