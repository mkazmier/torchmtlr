from math import ceil, sqrt
import numpy as np
import torch
from lifelines import KaplanMeierFitter


def encode_survival(time, event, bins):
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
    time : np.ndarray
        Array of event or censoring times.
    event : np.ndarray
        Array of event indicators (0 = censored).
    bins : np.ndarray
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    time = np.clip(time, 0, bins.max())
    bin_idx = np.digitize(time, bins)
    # add extra bin [max_time, inf) at the end
    y = torch.zeros(bins.shape[0] + 1, dtype=torch.float)
    if event == 1:
        y[bin_idx] = 1
    else:
        y[bin_idx:] = 1
    return y


def reset_parameters(model):
    """Resets the parameters of a PyTorch module and its children."""
    for m in model.modules():
        try:
            m.reset_parameters()
        except AttributeError:
            continue
    return model


def make_time_bins(times, num_bins=None, use_quantiles=True):
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times : np.ndarray
        Array of survival times.
    num_bins : int, optional
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles : bool
        If True, the bin edges will correspond to quantiles of `times` (default).
        Otherwise, generates equally-spaced bins.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
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
    of censoring weighting (IPCW) to correct for censored observations. The weights
    are computed using the Kaplan-Meier estimate of the censoring distribution.

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
