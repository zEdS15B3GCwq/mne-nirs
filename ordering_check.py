import mne

files = [
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\FieldTrip\220307_opticaldensity.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\GowerLabs\lumomat-1-1-0.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\Kernel\Flow2\Portal_2024_10_23\c345d04_2.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\Kernel\Flow2\Portal_2024_10_23\c345d04_3.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\Kernel\Flow2\Portal_2024_10_23\c345d04_5.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\Kernel\Flow50\Portal_2021_11\hb.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\Kernel\Flow50\Portal_2021_11\td_moments.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\Labnirs\labnirs_3wl_raw_recording.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\MNE-NIRS\20220217\20220217_nirx_15_3_recording.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\NIRx\NIRSport2\1.0.3\2021-04-23_005.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\NIRx\NIRSport2\1.0.3\2021-05-05_001.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\NIRx\NIRSport2\2021.9\2021-10-01_002.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\SfNIRS\snirf_homer3\1.0.3\nirx_15_3_recording.snirf",
    r"C:\Users\Dev\mne_data\MNE-testing-data\SNIRF\SfNIRS\snirf_homer3\1.0.3\snirf_1_3_nirx_15_2_recording_w_short.snirf",
]

for path in files:
    try:
        raw = mne.io.read_raw_snirf(path, verbose=False)
    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        continue
    ch_names = raw.ch_names
    if ch_names != sorted(ch_names):
        print(f"UNSORTED: {path}")
        print(ch_names)
        print()
