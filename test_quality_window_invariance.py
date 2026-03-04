import matplotlib.pyplot as plt
import mne
import numpy as np

from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = fnirs_data_folder / "Participant-1"
raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
raw = mne.preprocessing.nirs.optical_density(raw)

rng = np.random.default_rng(seed=123456)
t = np.arange(raw.n_times) / raw.info["sfreq"]
signal = np.sin(2.0 * np.pi * t) - 0.5
noisy = (signal + rng.normal(size=(raw.n_times,)) * 3) / 4
noise = rng.random(size=(raw.n_times,)) - 0.5

# Channel pair 1: clean sinusoid data (perfect score)
raw._data[0:6] = signal
# Channel pair 2: noisy data
raw._data[3] = 0.1
# Channel pair 3: noise
raw._data[5] = np.sin(2.0 * np.pi * (t + 0.2)) - 0.5

# calculate SCI quality
_, sci_45, sci_times_45 = peak_power(raw, time_window=45, threshold=0.7)
_, sci_30, sci_times_30 = peak_power(raw, time_window=30, threshold=0.7)
_, sci_15, sci_times_15 = peak_power(raw, time_window=15, threshold=0.7)

f = plt.figure()
plt.subplot(1, 3, 1)
plt.plot([t[0] for t in sci_times_45], sci_45[0, :].ravel(), label="45")
plt.plot([t[0] for t in sci_times_30], sci_30[0, :].ravel(), label="30")
plt.plot([t[0] for t in sci_times_15], sci_15[0, :].ravel(), label="15")
plt.legend()
plt.subplot(1, 3, 2)
plt.plot([t[0] for t in sci_times_45], sci_45[2, :].ravel(), label="45")
plt.plot([t[0] for t in sci_times_30], sci_30[2, :].ravel(), label="30")
plt.plot([t[0] for t in sci_times_15], sci_15[2, :].ravel(), label="15")
plt.legend()
plt.subplot(1, 3, 3)
plt.plot([t[0] for t in sci_times_45], sci_45[4, :].ravel(), label="45")
plt.plot([t[0] for t in sci_times_30], sci_30[4, :].ravel(), label="30")
plt.plot([t[0] for t in sci_times_15], sci_15[4, :].ravel(), label="15")
plt.legend()
plt.show()
