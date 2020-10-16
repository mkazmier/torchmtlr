from math import ceil, sqrt
from typing import Optional, Union

import numpy as np
import torch
from lifelines import KaplanMeierFitter


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


def normalize(data, mean=None, std=None, skip_cols=[]):
    """Normalizes the columns of Pandas DataFrame to zero mean and unit
    standard deviation."""
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    if skip_cols is not None:
        mean[skip_cols] = 0
        std[skip_cols] = 1
    return (data - mean) / std, mean, std


def integrated_brier_score(time_true, time_pred, event_observed, time_bins):
    r"""Compute the integrated Brier score for a predicted survival function.

    The integrated Brier score is defined as the mean squared error between
    the true and predicted survival functions at time t, integrated over all
    timepoints.

    Parameters
    ----------
    time_true : np.ndarray, shape=(n_samples,)
        The true time to event or censoring for each sample.
    time_pred : np.ndarray, shape=(n_samples, n_time_bins)
        The predicted survival probabilities for each sample in each time bin.
    event_observed : np.ndarray, shape=(n_samples,)
        The event indicator for each sample (1 = event, 0 = censoring).
    time_bins : np.ndarray, shape=(n_time_bins,)
        The time bins for which the survival function was computed.

    Returns
    -------
    float
        The integrated Brier score of the predictions.

    Notes
    -----
    This function uses the definition from [1]_ with inverse probability
    of censoring weighting (IPCW) to correct for censored observations.
    The weights are computed using the Kaplan-Meier estimate of the censoring
    distribution.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher, ‘Assessment
       and comparison of prognostic classification schemes for survival data’,
       Statistics in Medicine, vol. 18, no. 17‐18, pp. 2529–2545, Sep. 1999.
    """

    # compute weights for inverse probability of censoring weighting (IPCW)
    censoring_km = KaplanMeierFitter()
    censoring_km.fit(time_true, 1 - event_observed)
    weights_event = 1 / censoring_km.survival_function_at_times(
        time_true).values.reshape(-1, 1)
    weights_no_event = 1 / censoring_km.survival_function_at_times(
        time_bins).values.reshape(1, -1)
    weights_event[np.isinf(weights_event)] = 0
    weights_no_event[np.isinf(weights_no_event)] = 0

    # scores for subjects with event before time t for each time bin
    uncensored = (time_true[:, np.newaxis] <
                  time_bins) & event_observed[:, np.newaxis].astype(np.bool)
    scores_event = np.where(uncensored, (-time_pred)**2 * weights_event, 0)
    # scores for subjects with no event and no censoring before time t for each time bin
    scores_no_event = np.where((time_true[:, np.newaxis] >= time_bins),
                               (1 - time_pred)**2 * weights_no_event, 0)

    scores = np.mean(scores_event + scores_no_event, axis=0)

    # integrate over all time bins
    score = np.trapz(scores, time_bins) / time_bins.max()
    return score
