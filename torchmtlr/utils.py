from math import ceil, sqrt
from typing import Optional, Union

import numpy as np
import torch
from lifelines import KaplanMeierFitter
from einops import rearrange


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
    bin_idxs = np.digitize(time, bins)
    num_events = len(np.unique(event)) - 1
    # add extra bin [max_time, inf) at the end
    y = np.zeros((time.shape[0], num_events, bins.shape[0] + 1), dtype=np.int)
    for i, e in enumerate(event):
        bin_idx = bin_idxs[i]
        if e > 0:
            y[i, e - 1, bin_idx] = 1
        else:
            y[i, :, bin_idx:] = 1
    return torch.tensor(y.reshape(time.shape[0], -1), dtype=torch.float)


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
        times = times[event > 1]
    if num_bins is None:
        num_bins = ceil(sqrt(len(times)))
    if use_quantiles:
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    return bins


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


def pack_sequence(seq):
    return rearrange(seq, "n event time -> n (event time)")


def unpack_sequence(seq, num_events):
    return rearrange(seq, "n (event time) -> n event time", event=num_events, n=seq.size(0))


def make_synthetic_data(n_samples=8000,
                        n_noise_features=8,
                        base_hazard=.1,
                        percent_censor=.1,
                        competing=False):
    """Generates a synthetic survival dataset with linear hazard."""
    x = np.random.standard_normal((n_samples, n_noise_features + 2))
    hazards_0 = x[:, 0] + 2 * x[:, 1]
    hazards_1 = x[:, 0] + 2 * x[:, 2]
    event_time_0 = np.random.exponential(1 / (base_hazard * np.exp(hazards_0)))
    event_time_1 = np.random.exponential(1 / (base_hazard * np.exp(hazards_1)))
    event_time = np.minimum(event_time_0, event_time_1)
    censor_time = np.quantile(event_time, 1 - percent_censor)

    time = np.minimum(event_time, censor_time)
#     event = (event_time < censor_time).astype(np.int)
    event_type = np.where(event_time_0 < event_time_1, 1, 2)
    event = np.where(event_time < censor_time, event_type, 0)

    return pd.DataFrame({
        "time": time,
        "event": event,
        **{f"x{i+1}": x[:, i] for i in range(x.shape[1])}
    })
