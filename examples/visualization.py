import pathlib

import matplotlib.pyplot as plt
import seaborn
import torch

from torch_crps import crps_analytical, crps_ensemble, scrps_analytical, scrps_ensemble

EXAMPLES_DIR = pathlib.Path(pathlib.Path(__file__).parent)


def gamma_example(
    concentration: float,
    rate: float,
    eval_min: float,
    eval_max: float,
    num_eval_points: int,
    ensemble_size: int,
) -> None:
    """Example showing the probability density, negative log-likelihood, CRPS, and SCRPS of a Gamma distribution.

    Args:
        concentration: The concentration parameter of the Gamma distribution.
        rate: The rate parameter of the Gamma distribution.
        eval_min: The minimum grid value of y to evaluate on.
        eval_max: The maximum grid value of y to evaluate on.
        num_eval_points: The number of grid points to evaluate the functions on.
        ensemble_size: The number of ensemble estimates to use for the CRPS and SCRPS evaluation.
    """
    assert concentration > 0 and rate > 0
    assert eval_min < eval_max
    assert num_eval_points > 0
    assert ensemble_size > 0

    # Create a distribution (imagine a model's output).
    p = torch.distributions.Gamma(concentration=concentration, rate=rate)

    # Define a grid for all evaluations.
    y = torch.linspace(eval_min, eval_max, num_eval_points)

    # Evaluate the probability, negative log-probability, and the CRPS on the grid.
    p_y = p.log_prob(y).exp()
    nll_y = -p.log_prob(y)
    q_samples = p.sample((num_eval_points, ensemble_size))
    crps_y = crps_ensemble(q_samples, y)
    scrps_y = scrps_ensemble(q_samples, y)
    print(f"Evaluated p(y), NLL(p(y), y), and CRPS(p(y), y) on a grid of {num_eval_points} points")

    # Plot the evaluations.
    fig = plt.figure(figsize=(12, 8))
    y_plot = y.cpu().numpy()
    seaborn.lineplot(x=y_plot, y=p_y.cpu().numpy(), label=f"p(x) = Gamma(concentration={concentration}, rate={rate})")
    seaborn.lineplot(x=y_plot, y=nll_y.cpu().numpy(), label="NLL(p(x), y)")
    seaborn.lineplot(x=y_plot, y=crps_y.cpu().numpy(), label=f"CRPS_{ensemble_size}(p(x), y)")
    seaborn.lineplot(x=y_plot, y=scrps_y.cpu().numpy(), label=f"SCRPS_{ensemble_size}(p(x), y)")

    # Plot the mean and the median as dashed vertical lines.
    plt.axvline(p.mean.item(), color="C8", linestyle="dashed", label="mean")
    plt.axvline(p.mode.item(), color="C9", linestyle="dashed", label="median")

    # Add annotation.
    plt.xlabel("observation y")
    plt.ylabel("value")
    plt.legend(loc="upper right")

    # Save the plot.
    fig.tight_layout()
    fig.savefig(EXAMPLES_DIR / "visualization_gamma.png", dpi=300)
    print("Saved visualization to", EXAMPLES_DIR / "visualization_gamma.png")


def scale_example(
    loc: float,
    scale: float,
    num_eval_points: int,
) -> None:
    """Example showing the effect of the random variable's scale on the CRPS and SCRPS of a distribution.

    Args:
        loc: The location parameter of the Normal distribution.
        scale: The scale parameter of the Normal distribution.
        num_eval_points: The number of grid points to evaluate the functions on.
    """
    assert loc > 0 and scale > 0
    assert num_eval_points > 0

    # Create a distribution (imagine a model's output).
    p = torch.distributions.Normal(loc=loc, scale=scale)

    # Define a grid for all evaluations.
    eval_min, eval_max = loc - 4 * scale, loc + 4 * scale
    y = torch.linspace(eval_min, eval_max, num_eval_points)

    # Evaluate the probability, negative log-probability, and the CRPS on the grid.
    p_y = p.log_prob(y).exp()
    nll_y = -p.log_prob(y)
    crps_y = crps_analytical(p, y)
    scrps_y = scrps_analytical(p, y)
    print(f"Evaluated p(y), NLL(p(y), y), and CRPS(p(y), y) on a grid of {num_eval_points} points")

    # Plot the evaluations. Make the upper subplot 1/4 the height of the lower one
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 4]},
        sharex=True,
    )
    y_plot = y.cpu().numpy()

    # Upper (smaller) subplot: probability density
    ax[0].plot(y_plot, p_y.cpu().numpy(), color="C0", label=f"p(x) = Normal(loc={loc}, scale={scale})")
    ax[0].set_ylabel("p(x)")
    ax[0].legend(loc="upper right")

    # Lower (larger) subplot: NLL, CRPS, SCRPS
    ax[1].plot(y_plot, nll_y.cpu().numpy(), color="C1", label="NLL(p(x), y)")
    ax[1].plot(y_plot, crps_y.cpu().numpy(), color="C2", label="CRPS(p(x), y)")
    ax[1].plot(y_plot, scrps_y.cpu().numpy(), color="C3", label="SCRPS(p(x), y)")
    ax[1].set_xlabel("observation y")
    ax[1].set_ylabel("value")
    ax[1].legend(loc="upper center")

    # Save the plot.
    fig.tight_layout()
    fig.savefig(EXAMPLES_DIR / "visualization_normal.png", dpi=300)
    print("Saved visualization to", EXAMPLES_DIR / "visualization_normal.png")


if __name__ == "__main__":
    seaborn.set_theme()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)

    gamma_example(
        concentration=3,
        rate=4,
        eval_min=0.01,
        eval_max=2.5,
        num_eval_points=5000,
        ensemble_size=2000,
    )

    scale_example(
        loc=1000,
        scale=20,
        num_eval_points=1000,
    )
