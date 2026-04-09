# sphinx_gallery_thumbnail_number = 5

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from mne.preprocessing.nirs import beer_lambert_law, optical_density
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

import mne_nirs
from mne_nirs.channels import get_long_channels, get_short_channels
from mne_nirs.datasets import fnirs_motor_group
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.io.fold import fold_landmark_specificity
from mne_nirs.statistics import run_glm, statsmodels_to_results
from mne_nirs.visualisation import (
    plot_glm_surface_projection,
    plot_nirs_source_detector,
)

root = mne_nirs.datasets.audio_or_visual_speech.data_path()
dataset = BIDSPath(
    root=root,
    suffix="nirs",
    extension=".snirf",
    subject="04",
    task="AudioVisualBroadVsRestricted",
    datatype="nirs",
    session="01",
)
raw = mne.io.read_raw_snirf(dataset.fpath)
raw.annotations.rename(
    {"1.0": "Audio", "2.0": "Video", "3.0": "Control", "15.0": "Ends"}
)

# Download anatomical locations
subjects_dir = str(mne.datasets.sample.data_path()) + "/subjects"
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels = mne.read_labels_from_annot(
    "fsaverage", "HCPMMP1", "lh", subjects_dir=subjects_dir
)
labels_combined = mne.read_labels_from_annot(
    "fsaverage", "HCPMMP1_combined", "lh", subjects_dir=subjects_dir
)

brain = mne.viz.Brain(
    "fsaverage", subjects_dir=subjects_dir, background="w", cortex="0.5"
)
brain.add_sensors(
    raw.info, trans="fsaverage", fnirs=["channels", "pairs", "sources", "detectors"]
)
brain.show_view(azimuth=180, elevation=80, distance=450)

view_map = {
    "left-lat": np.r_[np.arange(1, 27), 28],
    "caudal": np.r_[27, np.arange(43, 53)],
    "right-lat": np.r_[np.arange(29, 43), 44],
}

fig_montage = mne_nirs.visualisation.plot_3d_montage(
    raw.info, view_map=view_map, subjects_dir=subjects_dir
)

# print(raw.ch_names)
# pairs = [name.split(" ")[0] for name in raw.ch_names]
# print(set(pairs))
# print(set(pairs[::2]))
# print(set(pairs) == set(pairs[::2]))
