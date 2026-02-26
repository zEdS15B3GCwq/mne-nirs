# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from mne.filter import filter_data
from mne.io import BaseRaw
from mne.preprocessing.nirs import _channel_frequencies, _validate_nirs_info
from mne.utils import _validate_type, verbose
from scipy.signal import periodogram


@verbose
def peak_power(
    raw,
    time_window=10,
    threshold=0.1,
    l_freq=0.7,
    h_freq=1.5,
    l_trans_bandwidth=0.3,
    h_trans_bandwidth=0.3,
    verbose=False,
):
    """
    Compute peak spectral power metric for each channel and time window.

    As described in [1]_ and [2]_.
    This method provides a metric of data quality along the duration of
    the measurement. The user can specify the window over which the
    metric is computed.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    time_window : number
        The duration of the window over which to calculate the metric.
        Default is 10 seconds as in PHOEBE paper.
    threshold : number
        Values below this are marked as bad and annotated in the raw file.
    %(l_freq)s
    %(h_freq)s
    %(l_trans_bandwidth)s
    %(h_trans_bandwidth)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The Raw data. Optionally annotated with bad segments.
    scores : array (n_nirs, n_windows)
        Array of peak power values.
    times : list
        List of the start and end times of each window used to compute the
        peak spectral power.

    References
    ----------
    .. [1] Pollonini L et al., “PHOEBE: a method for real time mapping of
           optodes-scalp coupling in functional near-infrared spectroscopy” in
           Biomed. Opt. Express 7, 5104-5119 (2016).
    .. [2] Hernandez, Samuel Montero, and Luca Pollonini. "NIRSplot: a tool for
           quality assessment of fNIRS scans." Optics and the Brain.
           Optical Society of America, 2020.
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, "raw")

    picks = _validate_nirs_info(raw.info)
    n_wavelengths = len(np.unique(_channel_frequencies(raw.info)))

    filtered_data = filter_data(
        raw._data,
        raw.info["sfreq"],
        l_freq,
        h_freq,
        picks=picks,
        verbose=verbose,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
    )

    window_samples = int(np.ceil(time_window * raw.info["sfreq"]))
    n_windows = int(np.floor(len(raw) / window_samples))

    scores = np.zeros((len(picks), n_windows))
    times = []

    for window in range(n_windows):
        start_sample = int(window * window_samples)
        end_sample = start_sample + window_samples
        end_sample = np.min([end_sample, len(raw) - 1])

        t_start = raw.times[start_sample]
        t_stop = raw.times[end_sample]
        times.append((t_start, t_stop))

        pair_indices = np.triu_indices(n_wavelengths, k=1)
        for ii in range(0, len(picks), n_wavelengths):
            group_data = filtered_data[
                picks[ii : ii + n_wavelengths], start_sample:end_sample
            ]
            # protect against zero
            group_data = np.array([ch / (np.std(ch) or 1) for ch in group_data])
            peak_powers = []
            for jj, kk in zip(*pair_indices):
                c = np.correlate(group_data[jj], group_data[kk], "full")
                c = c / window_samples
                [f, pxx] = periodogram(c, fs=raw.info["sfreq"], window="hamming")
                peak_powers.append(max(pxx))
            pp = min(peak_powers) if peak_powers else 0.0
            scores[ii : ii + n_wavelengths, window] = pp

            if (threshold is not None) & (pp < threshold):
                raw.annotations.append(
                    t_start,
                    time_window,
                    "BAD_PeakPower",
                    ch_names=[raw.ch_names[ii : ii + n_wavelengths]],
                )
    scores = scores[np.argsort(picks)]
    return raw, scores, times
