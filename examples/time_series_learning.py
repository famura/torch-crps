import pathlib

import matplotlib.pyplot as plt
import numpy
import seaborn
import torch
from matplotlib.figure import Figure

EXAMPLES_DIR = pathlib.Path(pathlib.Path(__file__).parent)

torch.set_default_dtype(torch.float32)


class SimpleDistriubtionalModel(torch.nn.Module):
    """A model that makes independent predictions given a sequence of inputs, yielding a StudentT distribution."""

    def __init__(self, dim_input: int, dim_output: int, hidden_size: int = 16, dof: float = 5.0) -> None:
        """Initialize the model.

        Args:
            dim_input: Input feature dimension.
            dim_output: Output feature dimension.
            hidden_size: GRU hidden state dimension.
            dof: Degrees of freedom for the StudentT distribution.
        """
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(dim_input)
        self.gru = torch.nn.GRU(dim_input, hidden_size, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, dim_output * 2)  # output mean and scale for each feature
        self.dof = dof

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.distributions.StudentT:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (num_samples, seq_len, dim_input).
            hidden: Optional (initial) hidden state for the GRU.

        Returns:
            num_samples independent StudentT distribution instances.
        """
        x = self.layer_norm(x)
        x, hidden = self.gru(x, hidden)
        x = self.linear(x)

        # Split into mean and scale parameters.
        loc, scale_raw = x.chunk(2, dim=-1)
        scale = torch.nn.functional.softplus(scale_raw) + 1e-6  # ensure positive scale

        # Create num_samples independent StudentT distributions with fixed dof.
        dist = torch.distributions.StudentT(df=self.dof, loc=loc, scale=scale)

        return dist


def load_and_split_data(dataset_name: str, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load dataset from file and split into train/test sets.

    Args:
        dataset_name: Name of the dataset file (without extension).
        normalize: Whether to normalize data to [-1, 1].

    Returns:
        Training and testing data as tensors of shape (num_samples, dim_data).
    """
    # Load and move the torch.
    data = torch.from_numpy(numpy.load(EXAMPLES_DIR / f"{dataset_name}.npy"))
    data = torch.atleast_2d(data).float().contiguous()
    if data.size(0) == 1:
        # torch.atleast_2d() adds dimensions to the front, but we want the first dim to be the length of the series.
        data = data.T

    if normalize:
        data /= max(abs(data.min()), abs(data.max()))

    # Make the first half the training set and the second half the test set.
    num_samples = len(data) // 2
    data_trn = data[:num_samples]
    data_tst = data[num_samples : num_samples * 2]

    return data_trn, data_tst


def simple_training(
    model: torch.nn.Module,
    packed_inputs: torch.Tensor,
    packed_targets: torch.Tensor,
    dataset_name: str,
    normalized_data: bool,
) -> None:
    """A bare bones training loop for the time series model that works on windowed data.

    The training loop is so simplified, it goes over the whle data set in each epoch without batching or validation.

    Args:
        model: The model to train.
        packed_inputs: Input tensor of shape (num_samples, seq_len, dim_input).
        packed_targets: Target tensor of shape (num_samples, dim_output).
        dataset_name: Name of the dataset.
        normalized_data: Whether the data is normalized.
    """
    # Use a simple heuristic for the optimization hyper-parameters.
    if dataset_name == "monthly_sunspots" and not normalized_data:  # data is in [0, 100] so we need more steps
        num_epochs = 2001
        lr = 1e-2
    else:
        num_epochs = 1001
        lr = 1e-2

    optim = torch.optim.Adam([{"params": model.parameters()}], lr=lr, eps=1e-8)

    for idx_e in range(num_epochs + 1):
        # Reset the gradients.
        optim.zero_grad(set_to_none=True)

        # Make the predictions.
        packed_predictions = model(packed_inputs)
        loss = -packed_predictions.log_prob(packed_targets).mean()

        # Call the optimizer.
        loss.backward()
        optim.step()

        if idx_e % 10 == 0:
            print(f"iter: {idx_e: >4} | loss: {loss.item()}")


def plot_results(
    dataset_name: str,
    data_trn: torch.Tensor,
    data_tst: torch.Tensor,
    predictions_trn_mean: numpy.ndarray,
    predictions_trn_std: numpy.ndarray,
    predictions_tst_mean: numpy.ndarray,
    predictions_tst_std: numpy.ndarray,
) -> Figure:
    """Create figure and plot data with predictions and uncertainty bands.

    Args:
        dataset_name: Name of the dataset for labels.
        data_trn: Training data.
        data_tst: Test data.
        predictions_trn_mean: Training predictions mean.
        predictions_trn_std: Training predictions standard deviation.
        predictions_tst_mean: Test predictions mean.
        predictions_tst_std: Test predictions standard deviation.

    Returns:
        The created figure.
    """
    if dataset_name not in ("monthly_sunspots", "mackey_glass"):
        raise NotImplementedError(f"Unknown dataset {dataset_name}! Please specify the necessary parts in the script.")

    fig, axs = plt.subplots(2, 1, figsize=(16, 9))

    # Plot training data and predictions.
    axs[0].plot(data_trn, label="data train")
    axs[0].plot(predictions_trn_mean, label="mean", color="C1")
    axs[0].fill_between(
        range(len(predictions_trn_mean)),
        predictions_trn_mean[:, 0] - 2 * predictions_trn_std[:, 0],
        predictions_trn_mean[:, 0] + 2 * predictions_trn_std[:, 0],
        alpha=0.3,
        label="±2σ",
        color="C1",
    )

    # Plot test data and predictions.
    axs[1].plot(data_tst, label="data test")
    axs[1].plot(predictions_tst_mean, label="mean", color="C1")
    axs[1].fill_between(
        range(len(predictions_tst_mean)),
        predictions_tst_mean[:, 0] - 2 * predictions_tst_std[:, 0],
        predictions_tst_mean[:, 0] + 2 * predictions_tst_std[:, 0],
        alpha=0.3,
        label="±2σ",
        color="C1",
    )

    # Set labels and legends.
    axs[1].set_xlabel("months" if dataset_name == "monthly_sunspots" else "time")
    axs[0].set_ylabel("spot count" if dataset_name == "monthly_sunspots" else "P")
    axs[1].set_ylabel("spot count" if dataset_name == "monthly_sunspots" else "P")
    axs[0].legend(loc="upper right", ncol=3)
    axs[1].legend(loc="upper right", ncol=3)

    return fig


if __name__ == "__main__":
    seaborn.set_theme()

    # Configure.
    torch.manual_seed(0)
    normalize_data = True  # scales the data to be in [-1, 1]
    dataset_name = "monthly_sunspots"  # monthly_sunspots or mackey_glass

    # Prepare the data.
    data_trn, data_tst = load_and_split_data(dataset_name, normalize_data)
    dim_data = data_trn.size(1)

    # Create the model.
    model = SimpleDistriubtionalModel(dim_input=dim_data, dim_output=dim_data)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create data set with rolling forecast scheme (i is input, t is target).
    # i t ...
    # i i t ...
    # i i i ... t
    inputs = []
    targets = []
    len_window = 10  # tested 10 and 20
    for idx in range(1, data_trn.size(0)):
        # Slice the input.
        idx_begin = max(idx - len_window, 0)
        inp = data_trn[idx_begin:idx, :].view(-1, dim_data)

        # Pad with zeros. This is not special to the models in this repo, but rather to the dataset structure.
        pad = (0, 0, len_window - inp.size(0), 0)  # from the left pad such that the input length is always 20
        inp_padded = torch.nn.functional.pad(inp, pad, mode="constant", value=0)

        # Store the data.
        inputs.append(inp_padded)
        targets.append(data_trn[idx, :].view(-1, dim_data))

    # Collect all and bring it in the form for batch processing.
    packed_inputs = torch.stack(inputs, dim=0)  # shape = (num_samples, len_window, dim_data)
    packed_targets = torch.stack(targets, dim=0)  # shape = (num_samples, dim_data)

    # Run a simple optimization loop.
    simple_training(model, packed_inputs, packed_targets, dataset_name, normalize_data)

    # Evaluate the model.
    with torch.no_grad():
        dist_trn = model(data_trn[:-1].unsqueeze(0), hidden=None)
        predictions_trn_mean = dist_trn.mean.squeeze(0).detach().numpy()
        predictions_trn_std = dist_trn.stddev.squeeze(0).detach().numpy()

        dist_tst = model(data_tst[:-1].unsqueeze(0), hidden=None)
        predictions_tst_mean = dist_tst.mean.squeeze(0).detach().numpy()
        predictions_tst_std = dist_tst.stddev.squeeze(0).detach().numpy()

    # Plot the results.
    fig = plot_results(
        dataset_name,
        data_trn,
        data_tst,
        predictions_trn_mean,
        predictions_trn_std,
        predictions_tst_mean,
        predictions_tst_std,
    )
    plt.savefig(EXAMPLES_DIR / f"time_series_learning_{dataset_name}.png", dpi=300)
    plt.show()
