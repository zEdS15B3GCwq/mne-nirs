import mne
from scipy.io import savemat

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = fnirs_data_folder / "Participant-1"
raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
od = mne.preprocessing.nirs.optical_density(raw)

savemat(
    "fnirs_motor_participant1.mat",
    {
        "raw": raw.get_data(),
        "od": od.get_data(),
        "Fs": raw.info["sfreq"],
        "n_times": raw.n_times,
    },
)
