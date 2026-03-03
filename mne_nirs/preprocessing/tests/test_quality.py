# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os

import mne  # type: ignore
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed


@pytest.fixture(name="fnirs_motor_data")
def fixture_fnirs_motor_data() -> mne.io.BaseRaw:
    """Read and return motor experiment data."""
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
    raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
    return mne.preprocessing.nirs.optical_density(raw)


@pytest.fixture(name="fnirs_labnirs_3wl_data")
def fixture_fnirs_labnirs_3wl_data() -> mne.io.BaseRaw:
    """Read and return 3-wavelength testing data."""
    pytest.importorskip("h5py")
    fname_labnirs_3wl = r"C:\Users\dev\mne_data\MNE-testing-data\SNIRF\Labnirs\labnirs_3wl_raw_recording.snirf"
    # fname_labnirs_3wl = (
    #     mne.datasets.testing.data_path(download=False)
    #     / "SNIRF"
    #     / "Labnirs"
    #     / "labnirs_3wl_raw_recording.snirf"
    # )
    raw = mne.io.read_raw_snirf(fname_labnirs_3wl)
    return mne.preprocessing.nirs.optical_density(raw)


@pytest.fixture(name="fnirs_datasets")
def fixture_fnirs_datasets(
    fnirs_motor_data, fnirs_labnirs_3wl_data
) -> list[mne.io.BaseRaw]:
    return [fnirs_motor_data, fnirs_labnirs_3wl_data]


@mne.datasets.testing.requires_testing_data
def test_peak_power_runs(fnirs_datasets: list[mne.io.BaseRaw]) -> None:
    for raw in fnirs_datasets:
        # fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
        # fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
        # raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
        # raw = mne.preprocessing.nirs.optical_density(raw)
        _, scores, _ = peak_power(raw.copy())
        assert len(scores) == len(raw.ch_names)


@mne.datasets.testing.requires_testing_data
def test_sci_windowed_runs(fnirs_datasets: list[mne.io.BaseRaw]) -> None:
    for raw in fnirs_datasets:
        _, scores, _ = scalp_coupling_index_windowed(raw.copy())
        assert len(scores) == len(raw.ch_names)


def test_sci_windowed_known_values(fnirs_motor_data: mne.io.BaseRaw):
    """Test segmented SCI with known correlation values for 2-wavelength data.

    Three channel pairs are overwritten with synthetic data that produce
    predictable scores. First, the channels are overwritten with the same
    signal to achieve perfect correlation (and keep the number of bad marks
    low). Then, various cases are tested in two, distant, 30-second windows
    (235 samples at 7.8125 Hz):

    - Pair 0 (ch 0-1): both = signal (SCI ≈ +1) throughout
    - Pair 1 (ch 2-3): W1 scale-invariant (SCI ≈ +1), W2 ch1 = -signal (SCI ≈ -1).
    - Pair 2 (ch 4-5): W1 noise + signal (SCI low), W2 noise + noise (SCI ≈ 0).

    For now, the test uses optical density data, even though SCI was meant
    to be calculated on raw voltage measurement. In the future, when the
    ``scalp_coupling_index_windowed`` method is updated to use raw data,
    this test will need to be updated as well.
    """
    raw = fnirs_motor_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 30.0
    num_channels = 6
    num_windows = 2
    windowA = 5
    windowB = 11

    rng = np.random.default_rng(seed=123456)

    # time_window=30 -> window_samples=235, n_windows=98
    w_numsamples = int(np.ceil(time_window * sfreq))
    # window A start:end with 0.5 window buffer around it for filter leak
    wA = round((windowA - 0.5) * w_numsamples), round((windowA + 1.5) * w_numsamples)
    # windows B start:end with buffer
    wB = round((windowB - 0.5) * w_numsamples), round((windowB + 1.5) * w_numsamples)

    # write the same signal in all the used channels
    signal = np.sin(2 * np.pi * 1.0 * np.arange(raw.n_times) / sfreq) - 0.5
    raw._data[0:num_channels] = signal[np.newaxis, :].copy()

    # Pair 0 (ch 0, 1)
    # signal in both, SCI≈1 throughout and no annotations in either window

    # Pair 1 (ch 2, 3):
    # window A: scale invariant (SCI ≈ +1) vs base signal in channel 3
    raw._data[2][wA[0] : wA[1]] = 0.3 * signal[wA[0] : wA[1]] + 2
    # window B: anti-correlation (SCI ≈ -1) vs base signal in channel 2
    raw._data[3][wB[0] : wB[1]] = -signal[wB[0] : wB[1]]

    # Pair 2 (ch 4, 5):
    # window A: noisy signal (low SCI) vs clean signal in channel 5
    raw._data[4][wA[0] : wA[1]] = (
        rng.normal(size=(wA[1] - wA[0],)) * 3.0 + signal[wA[0] : wA[1]]
    ) / 4
    # window B: both noise (SCI ≈ 0)
    raw._data[4][wB[0] : wB[1]] = rng.normal(size=(wB[1] - wB[0],))
    raw._data[5][wB[0] : wB[1]] = rng.random(size=(wB[1] - wB[0],)) - 0.5

    # calculate SCI quality
    raw, scores, times_out = scalp_coupling_index_windowed(
        raw, time_window=30, threshold=0.7
    )

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 98)
    assert len(times_out) == 98

    # verify SCI scores
    assert_allclose(scores[0:2, windowA], [-1, 1], atol=0.05)  # pair 0
    assert_allclose(scores[0:2, windowB], [1, 1], atol=0.05)
    assert_allclose(scores[2:4, windowA], [1, 1], atol=0.05)  # pair 1, window A
    assert_allclose(scores[2:4, windowB], [-1, -1], atol=0.05)  # pair 1, window B
    assert_allclose(scores[4:6, windowA], [0.44, 0.44], atol=0.05)  # pair 2, window A
    assert_allclose(
        scores[4:6, windowB], [-0.0471, -0.0471], atol=0.01
    )  # pair 2, window B

    # verify that BAD_SCI annotations exist for expected channels/windows
    # due to filter leak, other bad marks will be present too
    tracked_channels = raw.ch_names[:num_channels]
    target_windows = {windowA: 0, windowB: 1}

    # marks is a matrix of whether a bad mark has been found for a channel/window
    # row: channel, column: window
    marks = np.zeros((num_channels, num_windows), dtype=bool)

    for ann in raw.annotations:
        # skip if not SCI-related
        if ann["description"] != "BAD_SCI":
            continue

        # confirm window
        ann_window = round(ann["onset"] / time_window)
        col = target_windows.get(ann_window)
        if col is None:
            continue

        # confirm channel name
        for ann_name in ann["ch_names"]:
            if ann_name in tracked_channels:
                marks[tracked_channels.index(ann_name), col] = True

    expected = np.array([False, False] * 2 + [False, True] * 2 + [True, True] * 2)
    assert_array_equal(marks.ravel(), expected)


# @mne.datasets.testing.requires_testing_data
def test_sci_windowed_known_values_3wl(fnirs_labnirs_3wl_data: mne.io.BaseRaw) -> None:
    """Test segmented SCI with known correlation values for 3-wavelength data.

    This test focuses on multi-wavelength specific cases that were not covered
    in the 2-wavelength test, using the labnirs 3-wavelength SNIRF recording
    (250 samples at 19.6 Hz).

    - Group 1 (ch 0-2): all the same signal (SCI ≈ +1)
    - Group 2 (ch 3-5): one channel has inverted signal (SCI ≈ -1)
    - Group 3 (ch 6-8): one channel has noisy signal, another just noise
      (smallest SCI counts, SCI will be very small)

    For now, the test uses optical density data, even though SCI was meant
    to be calculated on raw voltage measurement. In the future, when the
    ``scalp_coupling_index_windowed`` method is updated to use raw data,
    this test will need to be updated as well.
    """
    pytest.importorskip("h5py")
    raw = fnirs_labnirs_3wl_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 4.0
    num_channels = 9
    window = 2

    rng = np.random.default_rng(seed=123456)

    w_numsamples = int(np.ceil(time_window * sfreq))
    # window start:end
    w = round(window * w_numsamples), round((window + 1) * w_numsamples)

    # write the same signal in all the used channels
    signal = np.sin(2 * np.pi * 1.0 * np.arange(raw.n_times) / sfreq) - 0.5
    # no need to use copy() for signal as the data is not modified later on
    # if part of the data is overwritten (as in the 2-wl test), copy() is needed
    raw._data[0:num_channels] = signal[np.newaxis, :]

    # Group 1 (ch 0, 1, 2): all perfectly correlated

    # Group 2 (ch 3, 4, 5): one channel has inverted signal
    raw._data[4] = -signal

    # Group 3 (ch 6, 7, 8): one channel is noisy, another just noise
    raw._data[6] = 0.1  # SCI = 0.09 if changed alone
    raw._data[8] = rng.random((raw.n_times,)) - 0.5  #  SCI = 0.14 if alone

    # calculate SCI quality
    raw, scores, times_out = scalp_coupling_index_windowed(
        raw, time_window=4, threshold=0.7
    )

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 3)
    assert len(times_out) == 3

    # verify SCI scores (in the 2nd, middle window)
    assert_allclose(scores[0:3, 1], 1, atol=0.05)  # group 0
    assert_allclose(scores[3:6, 1], -1, atol=0.05)  # group 1
    assert_allclose(scores[6:9, 1], -0.149, atol=0.01)  # group 2

    # verify that BAD_SCI annotations exist for expected channels
    tracked_channels = raw.ch_names[:num_channels]

    # marks is a vector indicating whether a bad mark has been found for a channel
    marks = np.zeros((num_channels,), dtype=bool)

    for ann in raw.annotations:
        # skip if not SCI-related
        if ann["description"] != "BAD_SCI":
            continue

        # we're only checking the middle windows
        ann_window = round(ann["onset"] / time_window)
        if ann_window != 1:
            continue

        # confirm channel name
        for ann_name in ann["ch_names"]:
            if ann_name in tracked_channels:
                print(ann)
                marks[tracked_channels.index(ann_name)] = True

    # print(scores[0:9, :])
    expected = np.array([False, True, True])
    assert_array_equal(marks, expected)


#     # Group 1 W2 has SCI < threshold; check that annotations cover all 3 channels of the group
#     group1_chs = set(raw.ch_names[0:3])
#     group1_anns = [
#         ann
#         for ann in raw.annotations
#         if ann["description"] == "BAD_SCI" and group1_chs.intersection(ann["ch_names"])
#     ]
#     assert len(group1_anns) > 0
#     for ann in group1_anns:
#         assert set(ann["ch_names"]) == group1_chs


# def test_peak_power_known_values_2wl():
#     """Test segmented PP with known spectral properties for 2-wavelength data.

#     Two channel pairs with an A/buffer/C layout at windows 10/11/12 (time_window=30 s,
#     235 samples/window, 98 windows total for fnirs_motor at 7.8125 Hz):

#     - Pair 0 (ch 0-1): W0 both in-band sinusoid (high PP), W2 ch1 zero (PP ≈ 0).
#     - Pair 1 (ch 2-3): in-band sinusoid throughout (consistently high PP).
#     """
#     fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
#     fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
#     raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
#     raw = mne.preprocessing.nirs.optical_density(raw)

#     sfreq = raw.info["sfreq"]
#     signal = np.sin(2 * np.pi * 1.0 * np.arange(raw.n_times) / sfreq)

#     w = int(np.ceil(30 * sfreq))
#     w1s, w1e = 11 * w, 12 * w  # buffer
#     w2s, w2e = 12 * w, 13 * w  # W2

#     # Pair 0 (ch 0, 1): W0 high PP, W2 ch1=0 -> PP≈0
#     raw._data[0] = signal.copy()
#     raw._data[1] = signal.copy()
#     raw._data[0][w1s:w1e] = 0.5 * signal[w1s:w1e]
#     raw._data[1][w1s:w1e] = 0.5 * signal[w1s:w1e]
#     raw._data[1][w2s:w2e] = 0.0

#     # Pair 1 (ch 2, 3): in-band signal throughout -> consistently high PP
#     raw._data[2] = signal.copy()
#     raw._data[3] = signal.copy()

#     raw, scores, times_out = peak_power(raw, time_window=30, threshold=0.75)

#     assert scores.shape == (len(raw.ch_names), 98)
#     assert len(times_out) == 98

#     # Pair 0: W0 high, W2 low (segmentation produces different values)
#     assert scores[0, 10] > 0.75
#     assert scores[0, 12] < 0.75
#     assert scores[0, 10] > scores[0, 12]

#     # Pair 1: consistently high in both target windows
#     assert scores[2, 10] > 0.75
#     assert scores[2, 12] > 0.75

#     # Pair 0 W2 must have BAD_PeakPower covering exactly the 2 channels of the pair
#     pair0_chs = set(raw.ch_names[0:2])
#     pair0_anns = [
#         ann
#         for ann in raw.annotations
#         if ann["description"] == "BAD_PeakPower"
#         and pair0_chs.intersection(ann["ch_names"])
#     ]
#     assert len(pair0_anns) > 0
#     for ann in pair0_anns:
#         assert set(ann["ch_names"]) == pair0_chs
#     # Pair 1 must not appear in any BAD_PeakPower annotation
#     pair1_chs = set(raw.ch_names[2:4])
#     pair1_anns = [
#         ann
#         for ann in raw.annotations
#         if ann["description"] == "BAD_PeakPower"
#         and pair1_chs.intersection(ann["ch_names"])
#     ]
#     assert len(pair1_anns) == 0


# @testing.requires_testing_data
# def test_peak_power_known_values_3wl():
#     """Test segmented PP with known spectral properties for 3-wavelength data.

#     Uses the labnirs 3-wavelength SNIRF recording (time_window=4 s,
#     79 samples/window, 3 windows total):

#     - Group 1 (ch 0-2): in-band sinusoid throughout -> high PP in W0 and W2.
#     - Group 2 (ch 3-5): ch3 and ch4 zero in W2, ch5 zero throughout -> PP≈0 in W2.
#     """
#     pytest.importorskip("h5py")
#     raw = mne.preprocessing.nirs.optical_density(read_raw_snirf(fname_labnirs_3wl))

#     sfreq = raw.info["sfreq"]
#     signal = np.sin(2 * np.pi * 1.0 * np.arange(raw.n_times) / sfreq)

#     w = int(np.ceil(4 * sfreq))

#     # Group 1 (ch 0, 1, 2): all in-band signal throughout -> high PP in all windows
#     raw._data[0] = signal.copy()
#     raw._data[1] = signal.copy()
#     raw._data[2] = signal.copy()

#     # Group 2 (ch 3, 4, 5): ch5 zero throughout; ch3/ch4 zero in W2 -> PP≈0 in W2
#     raw._data[3] = signal.copy()
#     raw._data[4] = signal.copy()
#     raw._data[5] = 0.0
#     raw._data[3][2 * w :] = 0.0
#     raw._data[4][2 * w :] = 0.0

#     raw, scores, times_out = peak_power(raw, time_window=4, threshold=0.75)

#     assert scores.shape == (len(raw.ch_names), 3)
#     assert len(times_out) == 3

#     # Group 1 PP > group 2 PP (group 2 W2 all-zero -> PP≈0)
#     assert scores[0, 0] > scores[3, 0]
#     assert scores[0, 2] > scores[3, 2]
#     assert scores[3, 2] < 0.75

#     # Group 2 W2 (all-zero) must have BAD_PeakPower covering all 3 channels of the group
#     group2_chs = set(raw.ch_names[3:6])
#     group2_anns = [
#         ann
#         for ann in raw.annotations
#         if ann["description"] == "BAD_PeakPower"
#         and group2_chs.intersection(ann["ch_names"])
#     ]
#     assert len(group2_anns) > 0
#     for ann in group2_anns:
#         assert set(ann["ch_names"]) == group2_chs
