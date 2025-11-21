from pathlib import Path

import pytest
import torch

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Check if CUDA support is available.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not supported in this setup.")
