from mne.io import read_raw_snirf

from mne_nirs.visualisation import plot_3d_montage

raw = read_raw_snirf(
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\MNE-NIRS\20220217\20220217_nirx_15_3_recording.snirf"
)
plot_3d_montage(raw.info, view_map={"left-lat": [1, 2, 3]})
