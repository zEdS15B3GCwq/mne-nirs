# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from pathlib import Path

import mne  # type: ignore
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed


@pytest.fixture(name="fnirs_motor_data")
def fixture_fnirs_motor_data() -> mne.io.BaseRaw:
    """Read and return motor experiment data."""
    fnirs_data_folder = Path(mne.datasets.fnirs_motor.data_path())
    fnirs_raw_dir = fnirs_data_folder / "Participant-1"
    raw = mne.io.read_raw_nirx(str(fnirs_raw_dir), verbose=True).load_data()
    return mne.preprocessing.nirs.optical_density(raw)
    # return raw


@pytest.fixture(name="fnirs_labnirs_3wl_data")
def fixture_fnirs_labnirs_3wl_data() -> mne.io.BaseRaw:
    """Read and return 3-wavelength testing data."""
    pytest.importorskip("h5py")
    fname_labnirs_3wl = (
        mne.datasets.testing.data_path(download=False)
        / "SNIRF"
        / "Labnirs"
        / "labnirs_3wl_raw_recording.snirf"
    )
    raw = mne.io.read_raw_snirf(fname_labnirs_3wl)
    return mne.preprocessing.nirs.optical_density(raw)
    # return raw


def find_annotations(
    raw: mne.io.BaseRaw,
    description: str,
    windows: list[int],
    channel_names: list[str],
    window_time: float,
) -> np.ndarray:
    # marks is a matrix of whether a bad mark has been found for a channel/window
    # row: channel, column: window
    marks = np.zeros((len(channel_names), len(windows)), dtype=bool)

    for ann in raw.annotations:
        # skip if not expected label
        if ann["description"] != description:
            continue

        # find corresponding window
        ann_window = round(ann["onset"] / window_time)  # type: ignore
        try:
            col = windows.index(ann_window)
        except ValueError:
            continue

        # confirm channel name
        for ann_name in ann["ch_names"]:  # type: ignore
            if ann_name in channel_names:
                marks[channel_names.index(ann_name), col] = True

    print(marks)
    return marks


@pytest.fixture(name="fnirs_datasets")
def fixture_fnirs_datasets(
    fnirs_motor_data, fnirs_labnirs_3wl_data
) -> list[mne.io.BaseRaw]:
    return [fnirs_motor_data, fnirs_labnirs_3wl_data]


@mne.datasets.testing.requires_testing_data
def test_peak_power_runs(fnirs_datasets: list[mne.io.BaseRaw]) -> None:
    """Test that `peak_power` successfully runs with test data."""
    for raw in fnirs_datasets:
        _, scores, _ = peak_power(raw.copy())
        assert len(scores) == len(raw.ch_names)


@mne.datasets.testing.requires_testing_data
def test_sci_windowed_runs(fnirs_datasets: list[mne.io.BaseRaw]) -> None:
    """Test that `scalp_coupling_index_windowed` successfully runs with test data."""
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

    The scores, annotations, and whether they're associated with the correct
    windows are confirmed.

    For now, the test uses optical density data, even though SCI was meant
    to be calculated on raw voltage measurement. In the future, when the
    ``scalp_coupling_index_windowed`` method is updated to use raw data,
    this test will need to be updated as well.
    """
    raw = fnirs_motor_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 30
    num_channels = 6
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
    signal = np.sin(2.0 * np.pi * np.arange(raw.n_times) / sfreq) - 0.5
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
        raw, time_window=time_window, threshold=0.7
    )

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 98)
    assert len(times_out) == 98

    # verify SCI scores
    assert_allclose(scores[0:2, windowA], [1, 1], atol=0.05)  # pair 0
    assert_allclose(scores[0:2, windowB], [1, 1], atol=0.05)
    assert_allclose(scores[2:4, windowA], [1, 1], atol=0.05)  # pair 1, window A
    assert_allclose(scores[2:4, windowB], [-1, -1], atol=0.05)  # pair 1, window B
    assert_allclose(scores[4:6, windowA], [0.44, 0.44], atol=0.05)  # pair 2, window A
    assert_allclose(
        scores[4:6, windowB], [-0.0471, -0.0471], atol=0.01
    )  # pair 2, window B

    # verify that BAD_SCI annotations exist for expected channels/windows
    marks = find_annotations(
        raw, "BAD_SCI", [windowA, windowB], raw.ch_names[:num_channels], time_window
    )

    expected = np.array([False, False] * 2 + [False, True] * 2 + [True, True] * 2)
    assert_array_equal(marks.ravel(), expected)


@mne.datasets.testing.requires_testing_data
def test_sci_windowed_known_values_multi_wavelength(
    fnirs_labnirs_3wl_data: mne.io.BaseRaw,
) -> None:
    """Test segmented SCI with known correlation values for >=3-wavelength data.

    This test focuses on multi-wavelength specific cases that were not covered
    in the 2-wavelength test, using the labnirs 3-wavelength SNIRF recording
    (250 samples at 19.6 Hz).

    - Group 1 (ch 0-2): all the same signal (SCI ≈ +1)
    - Group 2 (ch 3-5): one channel has inverted signal (SCI ≈ -1)
    - Group 3 (ch 6-8): one channel has noisy signal, another just noise
      (smallest SCI counts, SCI will be very small)
    - Group 4 (ch 9-11): same as group 3, different order (same SCI)

    For now, the test uses optical density data, even though SCI was meant
    to be calculated on raw voltage measurement. In the future, when the
    ``scalp_coupling_index_windowed`` method is updated to use raw data,
    this test will need to be updated as well.
    """
    raw = fnirs_labnirs_3wl_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 4
    num_channels = 12

    rng = np.random.default_rng(seed=123456)

    # write the same signal in all the used channels
    signal = np.sin(2 * np.pi * 1.0 * np.arange(raw.n_times) / sfreq) - 0.5
    # no need to use copy() for signal as the data is not modified later on
    # if part of the data is overwritten (as in the 2-wl test), copy() is needed
    raw._data[0:num_channels] = signal[np.newaxis, :]

    # Group 1 (ch 0, 1, 2): all perfectly correlated

    # Group 2 (ch 3, 4, 5): one channel has inverted signal
    raw._data[4] = -signal

    # Group 3 (ch 6, 7, 8): one channel is noisy, another just noise
    rand1 = rng.random((raw.n_times,)) - 0.5
    raw._data[6] = 0.1  # SCI = 0.09 if changed alone
    raw._data[8] = rand1  #  SCI = -0.149 if alone

    # Group 4 (ch 9, 10, 11): same as group 3 just different order
    raw._data[10] = rand1
    raw._data[11] = 0.1

    # calculate SCI quality
    raw, scores, times_out = scalp_coupling_index_windowed(
        raw, time_window=time_window, threshold=0.7
    )

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 3)
    assert len(times_out) == 3

    # verify SCI scores (in the 2nd, middle window)
    assert_allclose(scores[0:3, 1], 1, atol=0.05)  # group 1
    assert_allclose(scores[3:6, 1], -1, atol=0.05)  # group 2
    assert_allclose(scores[6:9, 1], -0.149, atol=0.01)  # group 3
    assert_allclose(scores[9:12, 1], scores[6:9, 1], atol=0.01)  # group 4

    # verify that BAD_SCI annotations exist for expected channels
    marks = find_annotations(
        raw, "BAD_SCI", [1], raw.ch_names[:num_channels], time_window
    )

    expected = np.repeat([False, True, True, True], 3)
    assert_array_equal(marks.ravel(), expected)


def test_peak_power_known_values(fnirs_motor_data: mne.io.BaseRaw) -> None:
    """Test segmented PP with known spectral properties for 2-wavelength data.

    First, test channels are overwritten with a sinusoid wave. Then, test data
    with known PP scores is written in two, distant, 30-second windows in the
    same test channels (time_window=30 s, 235 samples/window at
    7.8125 Hz):

    - Pair 0 (ch 0-1): W1 both sinusoid (high PP), W2 scale (invariant, high)
    - Pair 1 (ch 2-3): W1 phase shift, W2 inverted (invariant, both high PP)
    - Pair 2 (ch 4-5): W1 other frequences (high PP), W2 noisy (lower PP)
    - Pair 3 (ch 6-7): W1 only noise, W2 only other frequencies (both PP ≈ 0)

    The scores, annotations, and whether they're associated with the correct
    windows are confirmed.

    For now, the test uses optical density data, even though PP was meant
    to be calculated on raw voltage measurement. In the future, when the
    tested method is updated to use raw data, this test will need to be
    updated as well.
    """
    raw = fnirs_motor_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 30
    num_channels = 8
    windowA = 5
    windowB = 11

    rng = np.random.default_rng(seed=123456)

    # time_window=30 -> window_samples=235, n_windows=98
    w_numsamples = int(np.ceil(time_window * sfreq))
    # window A start:end with 0.5 window buffer around it for filter leak
    wA = round((windowA - 0.5) * w_numsamples), round((windowA + 1.5) * w_numsamples)
    # windows B start:end with buffer
    wB = round((windowB - 0.5) * w_numsamples), round((windowB + 1.5) * w_numsamples)

    t = np.arange(raw.n_times) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t) - 0.5  # base "heartbeat", 1 Hz
    lf_signal = np.sin(2 * np.pi * 0.2 * t) - 0.5  # low-freq, 0.2 Hz
    hf_signal = np.sin(2 * np.pi * 2.0 * t) - 0.5  # high-freq, 2 Hz

    # write base sinusoid signal into all test channels
    raw._data[0:num_channels] = signal[np.newaxis, :].copy()

    # Pair 0 (ch 0, 1):
    # window A: base signals (perfect PP)
    # window B: scale invariant (perfect PP)
    # test data in one channel is tested against base signal in the other
    raw._data[1][wB[0] : wB[1]] = 0.3 * signal[wB[0] : wB[1]] + 2

    # Pair 1 (ch 2, 3):
    # window A: phase invariant (perfect PP)
    raw._data[2][wA[0] : wA[1]] = signal[wA[0] + 2 : wA[1] + 2]
    # window B: inverted signal (same as phase shift, perfect PP)
    raw._data[3][wB[0] : wB[1]] = -signal[wB[0] : wB[1]]

    # Pair 2 (ch 4, 5):
    # window A: other freqs are filtered out (perfect PP)
    raw._data[4][wA[0] : wA[1]] = (signal + lf_signal + hf_signal)[wA[0] : wA[1]]
    # window B: noisy signal (lower PP)
    raw._data[5][wB[0] : wB[1]] = (
        rng.normal(size=(wB[1] - wB[0],)) * 2 + signal[wB[0] : wB[1]]
    )

    # Pair 3 (ch 6, 7):
    # window A: only noise (PP ≈ 0)
    raw._data[6][wA[0] : wA[1]] = rng.normal(size=(wB[1] - wB[0],))
    # window B: only other frequencies (PP ≈ 0)
    raw._data[7][wB[0] : wB[1]] = lf_signal[wB[0] : wB[1]] + hf_signal[wB[0] : wB[1]]

    # calculate PP quality
    raw, scores, times_out = peak_power(raw, time_window=time_window, threshold=0.1)

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 98)
    assert len(times_out) == 98

    # verify scores
    assert_allclose(scores[0:4, [windowA, windowB]], 10, atol=0.05)  # pairs 0 and 1
    assert_allclose(scores[4:6, windowA], [10, 10], atol=0.05)  # pair 2, window A
    assert_allclose(scores[4:6, windowB], [3.5, 3.5], atol=0.05)  # pair 2, window B
    assert_allclose(scores[6:8, windowA], [0, 0], atol=0.05)  # pair 3, window A
    assert_allclose(scores[6:8, windowB], [0, 0], atol=0.05)  # pair 3, window B

    # verify that BAD_PeakPower annotations exist for expected channels/windows
    marks = find_annotations(
        raw,
        "BAD_PeakPower",
        [windowA, windowB],
        raw.ch_names[:num_channels],
        time_window,
    )

    expected = np.array([False, False] * 6 + [True, True] * 2)
    assert_array_equal(marks.ravel(), expected)


@mne.datasets.testing.requires_testing_data
def test_peak_power_known_values_multi_wavelength(
    fnirs_labnirs_3wl_data: mne.io.BaseRaw,
) -> None:
    """Test segmented PP with known spectral properties for >=3-wavelength data.

    This test focuses on multi-wavelength specific cases that were not covered
    in the 2-wavelength test, using the labnirs 3-wavelength SNIRF recording
    (250 samples at 19.6 Hz). The recording is long enough for three 4-second
    windows, and tests are evaluated in the middle one.

    - Group 1 (ch 0-2): in-band sinusoid throughout (perfect PP)
    - Group 2 (ch 3-5): signal + noisy signal + noise (smallest wins, PP≈0)
    - Group 3 (ch 6-8): group 2 with different order (same PP)
    """
    raw = fnirs_labnirs_3wl_data.copy()
    sfreq = raw.info["sfreq"]

    time_window = 4
    num_channels = 9

    rng = np.random.default_rng(seed=123456)
    t = np.arange(raw.n_times) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t) - 0.5  # base "heartbeat", 1 Hz
    noisy = signal + rng.normal(size=(raw.n_times,)) * 2
    noise = rng.random(size=(raw.n_times,)) - 0.5

    # write the same signal in all the used channels
    # no need to use copy() for signal as the data is not modified later on
    # if part of the data is overwritten (as in the 2-wl test), copy() is needed
    raw._data[0:num_channels] = signal[np.newaxis, :]

    # Group 1 (ch 0, 1, 2): all perfectly correlated

    # Group 2 (ch 3, 4, 5):
    # base signal + noisy signal + noise
    # test data in one channel is tested against base signal in the other
    # raw._data[4] = noisy
    # raw._data[5] = noise

    # Group 3 (ch 6, 7, 8): group 2 reordered
    raw._data[6] = noise
    raw._data[8] = noisy

    # calculate PP quality
    raw, scores, times_out = peak_power(raw, time_window=time_window, threshold=0.1)

    # verify output shape
    assert scores.shape == (len(raw.ch_names), 3)
    assert len(times_out) == 3

    # verify scores
    print(scores)
    assert False
    assert_allclose(scores[0:3, 1], 10, atol=0.05)  # pairs 0 and 1
    # assert_allclose(scores[3:6, 1], [10, 10], atol=0.05)  # pair 2, window A
    # assert_allclose(scores[4:6, 1], [3.5, 3.5], atol=0.05)  # pair 2, window B
    # assert_allclose(scores[6:8, 1], [0, 0], atol=0.05)  # pair 3, window A
    # assert_allclose(scores[6:8, 1], [0, 0], atol=0.05)  # pair 3, window B


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


def test_sci_windowed_annotations_target_correct_channels():
    """BAD_SCI annotations must name the channels with poor SCI.

    _validate_nirs_info / _check_channels_ordered returns picks sorted
    alphabetically by channel name, which can differ from the row order
    in raw._data.  The buggy code uses the loop variable ``ii`` (an index
    into ``picks``) directly as a channel index into ``raw.ch_names``,
    causing annotations to be attached to the wrong channels.
    """
    sfreq = 10.0
    n_samples = 400  # 40 s at 10 Hz → 4 windows of 10 s

    # Channels stored in NON-alphabetical order: S2_D1 first, S1_D1 second.
    # _validate_nirs_info / _check_channels_ordered sorts picks alphabetically,
    # so it returns picks = [2, 3, 0, 1] instead of [0, 1, 2, 3].
    ch_names = ["S2_D1 760", "S2_D1 850", "S1_D1 760", "S1_D1 850"]
    ch_types = ["fnirs_od"] * 4

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    for ch in info["chs"]:
        # Wavelength must be stored in loc[9] (required by _validate_nirs_info).
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n_samples)

    # S2_D1 (raw._data rows 0, 1): identical  → SCI = +1.0  (GOOD, above threshold)
    # S1_D1 (raw._data rows 2, 3): anti-corr  → SCI = −1.0  (BAD,  below threshold)
    data = np.array([signal, signal, signal, -signal], dtype=float)

    raw = mne.io.RawArray(data, info)
    raw_out, _, _ = scalp_coupling_index_windowed(raw, time_window=10, threshold=0.1)

    bad_annotations = [
        ann for ann in raw_out.annotations if ann["description"] == "BAD_SCI"
    ]
    assert len(bad_annotations) > 0, "Expected BAD_SCI annotations for the S1_D1 pair"

    for ann in bad_annotations:
        annotated = ann["ch_names"]  # tuple of channel names for this annotation
        # The bad pair is S1_D1; the bug wrongly blames S2_D1 (raw.ch_names[0:2]).
        assert "S1_D1 760" in annotated, (
            f"BAD_SCI should name 'S1_D1 760' (bad channel), got {annotated}"
        )
        assert "S1_D1 850" in annotated, (
            f"BAD_SCI should name 'S1_D1 850' (bad channel), got {annotated}"
        )
        assert "S2_D1 760" not in annotated, (
            f"'S2_D1 760' is a good channel and must not appear in BAD_SCI, got {annotated}"
        )
        assert "S2_D1 850" not in annotated, (
            f"'S2_D1 850' is a good channel and must not appear in BAD_SCI, got {annotated}"
        )


@mne.datasets.testing.requires_testing_data
def test_order(fnirs_labnirs_3wl_data: mne.io.BaseRaw):
    raw = fnirs_labnirs_3wl_data.copy()
    sfreq = raw.info["sfreq"]
    scalp_coupling_index_windowed(raw)
    assert False
