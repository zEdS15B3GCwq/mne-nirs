# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from mne.filter import filter_data
from mne.io import BaseRaw
from mne.preprocessing.nirs import _validate_nirs_info
from mne.utils import _validate_type, verbose


@verbose
def scalp_coupling_index_windowed(
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
    Compute scalp coupling index for each channel and time window.

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

    picks = _validate_nirs_info(raw.info, fnirs="od", which="Scalp coupling index")

    # From mne-python.mne.preprocessing.nirs.scalp_coupling_index()
    # raw = raw.copy().pick(picks).load_data()
    # Wouldn't this allow simpler code below?

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

    # Determine number of wavelengths per source-detector pair
    ch_wavelengths = [raw.info["chs"][pick]["loc"][9] for pick in picks]
    n_wavelengths = len(set(ch_wavelengths))

    scores = np.zeros((len(picks), n_windows))
    times = []

    for window in range(n_windows):
        start_sample = int(window * window_samples)
        end_sample = start_sample + window_samples
        end_sample = np.min([end_sample, len(raw) - 1])

        t_start = raw.times[start_sample]
        t_stop = raw.times[end_sample]
        times.append((t_start, t_stop))

        if n_wavelengths == 2:
            # Use pairwise correlation for 2 wavelengths (backward compatibility)
            for ii in range(0, len(picks), 2):
                c1 = filtered_data[picks[ii]][start_sample:end_sample]
                c2 = filtered_data[picks[ii + 1]][start_sample:end_sample]
                # vvvv changed to match mne-python.mne.preprocessing.nirs.scalp_coupling_index
                with np.errstate(invalid="ignore"):
                    c = np.corrcoef(c1, c2)[0][1]
                if not np.isfinite(c):
                    c = 0
                # ^^^^
                scores[ii, window] = c
                scores[ii + 1, window] = c

                if (threshold is not None) & (c < threshold):
                    raw.annotations.append(
                        t_start,
                        time_window,
                        "BAD_SCI",
                        ch_names=[raw.ch_names[picks[ii] : picks[ii] + 2]],
                    )
        else:
            # For multiple wavelengths: calculate all pairwise correlations and use minimum

            # Group picks by number of wavelengths
            # Drops last incomplete group, but we're assuming valid data
            pick_iter = iter(picks)
            pick_groups = zip(*[pick_iter] * n_wavelengths)

            for group_picks in pick_groups:
                group_data = filtered_data[group_picks]

                # Calculate pairwise correlations within the group
                pair_indices = np.triu_indices(len(group_picks), k=1)
                correlations = np.zeros(pair_indices[0].shape[0])

                for n, (ii, jj) in enumerate(zip(*pair_indices)):
                    c1 = group_data[ii][start_sample:end_sample]
                    c2 = group_data[jj][start_sample:end_sample]
                    with np.errstate(invalid="ignore"):
                        c = np.corrcoef(c1, c2)[0][1]
                    if np.isfinite(c):
                        correlations[n] = c

                # Use minimum correlation as quality metric
                group_sci = correlations.min()

                # Assign the same SCI value to all channels in the group
                scores[group_picks, window] = group_sci

                if (threshold is not None) & (group_sci < threshold):
                    ch_names_in_group = [raw.ch_names[pick] for pick in group_picks]
                    raw.annotations.append(
                        t_start,
                        time_window,
                        "BAD_SCI",
                        ch_names=ch_names_in_group,
                    )

    scores = scores[np.argsort(picks)]
    return raw, scores, times
