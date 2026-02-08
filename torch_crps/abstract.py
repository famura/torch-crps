import torch


def crps_abstract(accuracy: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    """High-level function to compute the CRPS from the accuracy and dispersion terms.

    Args:
        accuracy: The accuracy term A, independent of the methods used to compute it, of shape (*batch_shape,).
        dispersion: The dispersion term D, independent of the methods used to compute it, of shape (*batch_shape,).

    Returns:
        The CRPS value for each forecast in the batch, of shape (*batch_shape,).
    """
    return accuracy - 0.5 * dispersion


def scrps_abstract(accuracy: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    """High-level function to compute the SCRPS from the accuracy and dispersion terms.

    Args:
        accuracy: The accuracy term A, independent of the methods used to compute it, of shape (*batch_shape,).
        dispersion: The dispersion term D, independent of the methods used to compute it, of shape (*batch_shape,).

    Returns:
        The SCRPS value for each forecast in the batch, of shape (*batch_shape,).
    """
    return accuracy / dispersion + 0.5 * torch.log(dispersion)
