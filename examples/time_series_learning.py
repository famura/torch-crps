"""This script loads a small time-series dataset and trains a simple distributional model on it.

The key insight here is that the CRPS loss, just like the NLL loss, allows this script to solve the task.

I tried a few variations of the model and the optimizer's parameters (not an exhaustive search). Finally, I arrived
at the insight that the NLL loss function incentivizes the model to converge to the marginal distribution of the data,
clearly a local optimum, in more scenarios than the CRPS loss.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy
import seaborn
import torch
from matplotlib.figure import Figure

from torch_crps import crps_analytical

EXAMPLES_DIR = pathlib.Path(pathlib.Path(__file__).parent)

torch.set_default_dtype(torch.float32)


class SimpleDistributionalModel(torch.nn.Module):
    """A model that makes independent predictions given a sequence of inputs, yielding a StudentT distribution."""

    def __init__(self, dim_input: int, dim_output: int, hidden_size: int) -> None:
        """Initialize the model.

        Args:
            dim_input: Input feature dimension.
            dim_output: Output feature dimension.
            hidden_size: GRU hidden state dimension.
        """
        super().__init__()
        self.gru = torch.nn.GRU(dim_input, hidden_size, num_layers=2, batch_first=True, dropout=0.0)
        self.activation = torch.nn.SiLU()
        self.output_projection = torch.nn.Linear(hidden_size, dim_output * 2)  # output mean and scale for each feature

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (num_samples, seq_len, dim_input).

        Returns:
            num_samples independent torch distribution instances.
        """
        # Pass through GRU
        x, _ = self.gru(x)  # shape: (num_samples, seq_len, hidden_size)

        # Use the last output.
        x = x[:, -1, :]  # shape: (num_samples, hidden_size)

        x = self.activation(x)
        x = self.output_projection(x)

        # Split into mean and scale parameters.
        loc, scale_raw = x.chunk(2, dim=-1)
        scale = torch.nn.functional.softplus(scale_raw) + 1e-4  # ensure positive scale with reasonable minimum

        # Create num_samples independent Nornmal distributions.
        dist = torch.distributions.Normal(loc=loc, scale=scale)

        return dist


def load_and_split_data(dataset_name: str, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load dataset from file and split into train/test sets.

    Args:
        dataset_name: Name of the dataset file (without extension).
        normalize: Whether to normalize data to [-1, 1].

    Returns:
        Training and testing data as tensors of shape (num_samples, dim_data).
    """
    # Load data and convert to torch tensor.
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
    normalize_data: bool,
    use_crps: bool,
    device: torch.device,
) -> None:
    """A bare bones training loop for the time series model that works on windowed data.

    The training loop is so simplified, it goes over the whole data set in each epoch without batching or validation.

    Args:
        model: The model to train.
        packed_inputs: Input tensor of shape (num_samples, seq_len, dim_input).
        packed_targets: Target tensor of shape (num_samples, dim_output).
        dataset_name: Name of the dataset.
        normalize_data: Whether the data is normalized.
        use_crps: If True, use CRPS loss. If false, use negative log-likelihood loss.
        device: Device to run training on.
    """
    # Move data to device.
    packed_inputs = packed_inputs.to(device)
    packed_targets = packed_targets.to(device)

    # Use a simple heuristic for the optimization hyper-parameters.
    if dataset_name == "monthly_sunspots":
        if normalize_data:
            num_epochs = 3001
            lr = 3e-3
        else:
            # The data is in [0, 100] so we need more steps.
            num_epochs = 5001
            lr = 4e-3
    else:
        num_epochs = 2001
        lr = 2e-3

    optim = torch.optim.Adam([{"params": model.parameters()}], lr=lr, eps=1e-8)

    model.train()
    for idx_e in range(num_epochs + 1):
        # Reset the gradients.
        optim.zero_grad(set_to_none=True)

        # Make the predictions.
        packed_predictions = model(packed_inputs)

        # Compute the loss, lower is better in both cases.
        if use_crps:
            loss = crps_analytical(packed_predictions, packed_targets).mean()
        else:
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


@torch.inference_mode()
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

    for idx in range(1, data.size(0)):
        idx_begin = max(idx - len_window, 0)
        inp = data[idx_begin:idx, :]

        # Pad with zeros from the left to keep input length constant.
        pad = (0, 0, len_window - inp.size(0), 0)
        inp_padded = torch.nn.functional.pad(inp, pad, mode="constant", value=0)
        inp_padded = inp_padded.unsqueeze(0).to(device)  # shape: (1, len_window, dim_data)

        dist = model(inp_padded)
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
    torch.manual_seed(0)

    # Configure.
    normalize_data = True  # scales the data to be in [-1, 1] (recommended for monthly_sunspots dataset)
    dataset_name = "monthly_sunspots"  # monthly_sunspots or mackey_glass
    use_crps = True  # if True, use CRPS loss instead of NLL
    len_window = 10  # tested 10 and 20
    dim_hidden = 64

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
    # i t ...
    # i i t ...
    # i i ... t
    inputs = []
    targets = []
    for idx in range(len_window, data_trn.size(0)):
        # Slice the input.
        idx_begin = max(idx - len_window, 0)
        inp = data_trn[idx_begin:idx, :].view(-1, dim_data)

        # Pad with zeros. This is not special to the models in this repo, but rather to the dataset structure.
        pad = (0, 0, len_window - inp.size(0), 0)  # from the left pad such that the input length is always 20
        inp_padded = torch.nn.functional.pad(inp, pad, mode="constant", value=0)

        # Store the data.
        inputs.append(inp_padded)
        targets.append(data_trn[idx, :].view(-1))

    # Collect all and bring it in the form for batch processing.
    packed_inputs = torch.stack(inputs, dim=0)  # shape = (num_samples, len_window, dim_data)
    packed_targets = torch.stack(targets, dim=0)  # shape = (num_samples, dim_data)

    # Run a simple optimization loop.
    simple_training(
        model,
        packed_inputs,
        packed_targets,
        dataset_name=dataset_name,
        normalize_data=normalize_data,
        use_crps=use_crps,
        device=device,
    )

    # Evaluate the model using the same rolling window approach as training.
    predictions_trn_mean, predictions_trn_std = evaluate_model(model, data_trn, len_window, device)
    predictions_tst_mean, predictions_tst_std = evaluate_model(model, data_tst, len_window, device)

    # Plot the results (skip the first point since we predict from index 1 onward).
    fig = plot_results(
        dataset_name,
        data_trn[1:],
        data_tst[1:],
        predictions_trn_mean,
        predictions_trn_std,
        predictions_tst_mean,
        predictions_tst_std,
    )
    loss_name = "crps" if use_crps else "nll"
    plt.savefig(EXAMPLES_DIR / f"time_series_learning_{dataset_name}_{loss_name}.png", dpi=300)
    print(f"Figure saved to {EXAMPLES_DIR / f'time_series_learning_{dataset_name}.png'}")
