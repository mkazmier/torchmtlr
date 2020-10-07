import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pack_sequence, unpack_sequence


class MTLR(nn.Module):
    """Multi-task logistic regression for individualised survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """
    def __init__(self, in_features, num_time_bins, num_events=1):
        """Initialises the module.

        Parameters
        ----------
        in_features : int
            Number of input features.
        num_time_bins : int
            The number of bins to divide the time axis into.
        """
        super().__init__()
        self.in_features = in_features
        self.num_time_bins = num_time_bins
        self.num_events = num_events

        weight = torch.zeros(self.in_features,
                             (self.num_time_bins-1) * self.num_events,
                             dtype=torch.float)
        bias = torch.zeros((self.num_time_bins-1) * self.num_events)
        self.mtlr_weight = nn.Parameter(weight)
        self.mtlr_bias = nn.Parameter(bias)

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x):
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        # (n (event time)) -> (n event time)
        out = unpack_sequence(out, self.num_events)
        # (n event time) -> (n (event time))
        return pack_sequence(torch.matmul(out, self.G))

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)


def masked_logsumexp(x, mask, dim=-1):
    """Computes logsumexp over elements of a tensor specified by a mask in
    a numerically stable way.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    mask : torch.Tensor
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim : int
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask,
                  dim=dim)) + max_val


def mtlr_neg_log_likelihood(logits, target, average=False):
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)
    return nll_total


def mtlr_cif(logits, num_events=1):
    """Generates predicted cumulative incidence function for a batch of logits.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    num_events
        The number of events (>= 1).

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    density = torch.softmax(logits, dim=1)
    # pad the predicted CIF so that
    # for each event, CIF(t=0) = 0
    padding = (1, 0, 0, 0, 0, 0)
    return F.pad(unpack_sequence(density, num_events)[..., :-1].cumsum(-1), padding)


def mtlr_cif_at_times(logits, train_times, pred_times, num_events=1):
    """Generates predicted cumulative incidence functions at arbitrary
    timepoints using linear interpolation.

    This function uses scipy.interpolate internally and returns a Numpy array,
    in contrast with `mtlr_cif`.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    train_times : Tensor or ndarray
        Time bins used for model training. Must have the same length as the
        first dimension of `pred`.
    pred_times : np.ndarray
        Array of times used to compute the CIF.
    num_events
        The number of events (>= 1).

    Returns
    -------
    np.ndarray
        The CIF for each event and each row in `pred` at `pred_times`. The
        values are linearly interpolated at timepoints not used for training.
    """
    # TODO can we use PyTorch interpolation functions here to make it
    # differentiable?
    with torch.no_grad():
        cif = mtlr_cif(logits, num_events).cpu().numpy()
    interpolator = interp1d(train_times, cif)
    return interpolator(pred_times)


def mtlr_survival(logits, num_events=1):
    """Generates predicted survival curves from a batch of logits.

    If num_events == 1, the resulting curves describe the probability of
    suvival without the event at each timepoint. If num_events > 1, the result
    is the overall survival probability (i.e. without any event) at each
    timepoint.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    num_events
        The number of events (>= 1).

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    cif = mtlr_cif(logits, num_events)
    # Sum the CIF over events to obtain overall survival curves.
    return 1 - cif.sum(dim=1)


def mtlr_survival_at_times(logits, train_times, pred_times, num_events=1):
    """Generates predicted survival curves at arbitrary timepoints using linear
    interpolation.

    This function uses scipy.interpolate internally and returns a Numpy array,
    in contrast with `mtlr_survival`.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    train_times : Tensor or ndarray
        Time bins used for model training. Must have the same length as the
        first dimension of `pred`.
    pred_times : np.ndarray
        Array of times used to compute the survival curve.
    num_events
        The number of events (>= 1).

    Returns
    -------
    np.ndarray
        The survival curve for each row in `pred` at `pred_times`. The values
        are linearly interpolated at timepoints not used for training.
    """
    # TODO can we use PyTorch interpolation functions here to make it
    # differentiable?
    with torch.no_grad():
        surv = mtlr_survival(logits, num_events).numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(np.clip(pred_times, 0, train_times.max()))


def mtlr_hazard(logits, num_events=1):
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t,
    respectively.

    Parameters
    ----------
    logits : torch.Tensor
        The predicted logits as returned by the `MTLR` module.
    num_events
        The number of events (>= 1).

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    density = unpack_sequence(torch.softmax(logits, dim=1), num_events)
    survival = mtlr_survival(logits, num_events).unsqueeze(1)
    return density[..., :-1] / (survival + 1e-15)[..., 1:]


def mtlr_risk(logits, num_events=1):
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits : torch.Tensor
        The predicted logits as returned by the `MTLR` module.
    num_events
        The number of events (>= 1).

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits, num_events)
    return torch.sum(hazard.cumsum(dim=-1), dim=-1)
