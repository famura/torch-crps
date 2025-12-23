import pathlib

import matplotlib.pyplot as plt
import numpy
import seaborn
import torch
from matplotlib.figure import Figure

EXAMPLES_DIR = pathlib.Path(pathlib.Path(__file__).parent)

torch.set_default_dtype(torch.float32)


class SimpleDistributionalModel(torch.nn.Module):
    """A model that makes independent predictions given a sequence of inputs, yielding a StudentT distribution."""

    def __init__(
        self, dim_input: int, dim_output: int, hidden_size: int, dof: float = 10, dropout: float = 0.1
    ) -> None:
        """Initialize the model.

        Args:
            dim_input: Input feature dimension.
            dim_output: Output feature dimension.
            hidden_size: GRU hidden state dimension.
            dof: Degrees of freedom for the StudentT distribution.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.conv = torch.nn.Conv1d(dim_input, hidden_size, kernel_size=5, padding=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.Tanh()
        self.output_projection = torch.nn.Linear(hidden_size, dim_output * 2)  # output mean and scale for each feature
        self.dof = dof

    def forward(self, x: torch.Tensor) -> torch.distributions.StudentT:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (num_samples, seq_len, dim_input).

        Returns:
            num_samples independent StudentT distribution instances.
        """
        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Apply temporal convolution with activation
        x = torch.relu(self.conv(x))

        # Transpose back: (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)

        # Only use the last time step's feature for output prediction.
        x = x[:, -1, :]  # shape: (num_samples, hidden_size)

        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_projection(x)

        # Split into mean and scale parameters.
        loc, scale_raw = x.chunk(2, dim=-1)
        scale = torch.nn.functional.softplus(scale_raw) + 1e-4  # ensure positive scale with reasonable minimum

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
    device: torch.device,
) -> None:
    """A bare bones training loop for the time series model that works on windowed data.

    The training loop is so simplified, it goes over the whle data set in each epoch without batching or validation.

    Args:
        model: The model to train.
        packed_inputs: Input tensor of shape (num_samples, seq_len, dim_input).
        packed_targets: Target tensor of shape (num_samples, dim_output).
        dataset_name: Name of the dataset.
        normalized_data: Whether the data is normalized.
        device: Device to run training on.
    """
    # Move data to device.
    packed_inputs = packed_inputs.to(device)
    packed_targets = packed_targets.to(device)

    # Use a simple heuristic for the optimization hyper-parameters.
    if dataset_name == "monthly_sunspots" and not normalized_data:  # data is in [0, 100] so we need more steps
        num_epochs = 2001
        lr = 5e-3
    else:
        num_epochs = 4001
        lr = 5e-3

    optim = torch.optim.Adam([{"params": model.parameters()}], lr=lr, eps=1e-8)

    model.train()
    for idx_e in range(num_epochs + 1):
        # Reset the gradients.
        optim.zero_grad(set_to_none=True)

        # Make the predictions.
        packed_predictions = model(packed_inputs)
        loss = -packed_predictions.log_prob(packed_targets).mean()

        # Call the optimizer.
        loss.backward()

        # Clip gradients to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()

        if idx_e % 100 == 0:
            with torch.no_grad():
                # Get model predictions for diagnostics
                pred_mean = packed_predictions.mean.mean().item()
                pred_std = packed_predictions.stddev.mean().item()
                target_mean = packed_targets.mean().item()
                target_std = packed_targets.std().item()

                # Check variance in predictions - should not be zero if model is learning
                pred_variance = packed_predictions.mean.var().item()

            print(
                f"iter: {idx_e: >4} | loss: {loss.item():.4f} | "
                f"pred_μ: {pred_mean:.4f} tgt_μ: {target_mean:.4f} | "
                f"pred_σ: {pred_std:.4f} tgt_σ: {target_std:.4f} | "
                f"pred_var: {pred_variance:.6f}"
            )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module, data: torch.Tensor, len_window: int, device: torch.device
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Evaluate the model using rolling window predictions.

    Args:
        model: The trained model to evaluate.
        data: Input data of shape (num_samples, dim_data).
        len_window: Length of the rolling window.
        device: Device to run evaluation on.

    Returns:
        Predictions mean and standard deviation as numpy arrays of shape (num_predictions, dim_data).
    """
    model.eval()
    predictions_mean_list, predictions_std_list = [], []

    for idx in range(len_window, data.size(0)):
        idx_begin = idx - len_window
        inp = data[idx_begin:idx, :].unsqueeze(0).to(device)  # shape: (1, len_window, dim_data)
        dist = model(inp)
        predictions_mean_list.append(dist.mean.squeeze(0))
        predictions_std_list.append(dist.stddev.squeeze(0))

    predictions_mean = torch.stack(predictions_mean_list, dim=0).detach().cpu().numpy()
    predictions_std = torch.stack(predictions_std_list, dim=0).detach().cpu().numpy()

    return predictions_mean, predictions_std


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
    len_window = 20  # tested 10 and 20
    dim_hidden = 128  # increased for better expressiveness

    # Setup device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare the data.
    data_trn, data_tst = load_and_split_data(dataset_name, normalize_data)
    dim_data = data_trn.size(1)

    # Create the model and move to device
    model = SimpleDistributionalModel(dim_input=dim_data, dim_output=dim_data, hidden_size=dim_hidden)
    model = model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create data set with rolling forecast scheme (i is input, t is target).
    # Only use complete windows (no padding) to avoid confusing the model with zeros.
    inputs = []
    targets = []
    for idx in range(len_window, data_trn.size(0)):
        idx_begin = idx - len_window
        inp = data_trn[idx_begin:idx, :].view(-1, dim_data)

        # Store the data.
        inputs.append(inp)
        targets.append(data_trn[idx, :].view(-1, dim_data))

    # Collect all and bring it in the form for batch processing.
    packed_inputs = torch.stack(inputs, dim=0)  # shape = (num_samples, len_window, dim_data)
    packed_targets = torch.stack(targets, dim=0)  # shape = (num_samples, dim_data)

    # Run a simple optimization loop.
    simple_training(model, packed_inputs, packed_targets, dataset_name, normalize_data, device)

    # Evaluate the model using the same rolling window approach as training.
    predictions_trn_mean, predictions_trn_std = evaluate_model(model, data_trn, len_window, device)
    predictions_tst_mean, predictions_tst_std = evaluate_model(model, data_tst, len_window, device)

    # Plot the results (skip the first len_window points since we don't have predictions for them).
    fig = plot_results(
        dataset_name,
        data_trn[len_window:],
        data_tst[len_window:],
        predictions_trn_mean,
        predictions_trn_std,
        predictions_tst_mean,
        predictions_tst_std,
    )
    plt.savefig(EXAMPLES_DIR / f"time_series_learning_{dataset_name}.png", dpi=300)
    plt.show()
