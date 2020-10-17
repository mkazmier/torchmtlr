from math import ceil, sqrt
from typing import Optional, Union

import numpy as np
import torch


def encode_survival(time: Union[float, int, np.ndarray],
                    event: Union[int, np.ndarray],
                    bins: np.ndarray) -> torch.Tensor:
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time
        Time of event or censoring.
    event
        Event indicator (0 = censored).
    bins
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    if isinstance(time, (float, int)):
        time = np.array([time])
    if isinstance(event, int):
        event = np.array([event])

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1), dtype=torch.float)
    for i, (t, e) in enumerate(zip(time, event)):
        bin_idx = np.digitize(t, bins)
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()


def reset_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """Resets the parameters of a PyTorch module and its children."""
    for m in model.modules():
        try:
            m.reset_parameters()
        except AttributeError:
            continue
    return model


def make_time_bins(times: np.ndarray,
                   num_bins: Optional[int] = None,
                   use_quantiles: bool = True,
                   event: Optional[np.ndarray] = None) -> np.ndarray:
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array of event indicators. If specified, only samples where event == 1
        will be used to determine the time bins.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = ceil(sqrt(len(times)))
    if use_quantiles:
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    return bins


def make_optimizer(opt_cls: torch.optim.Optimizer,
                   model: torch.nn.Module,
                   **kwargs) -> torch.optim.Optimizer:
    """Creates a PyTorch optimizer for MTLR training.

    This is a helper function to instantiate an optimizer with no weight decay
    on biases (which shouldn't be regularized) and MTLR parameters (which have
    a separate regularization mechanism). Note that the `opt_cls` argument
    should be the optimizer class, not an instantiated object (e.g. optim.Adam
    instead of optim.Adam(model.parameters(), ...)).

    Parameters
    ----------
    opt_cls
        The optimizer class to instantiate.
    model
        The PyTorch module whose parameters should be optimized.
    kwargs
        Additional keyword arguments to optimizer constructor.

    Returns
    -------
    torch.optim.Optimizer
        The instantiated optimizer object.

    """
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer
