"""
Functions to perform ICA (human input)

Needs to:
Do ICA on raw object using random seeds (with option for random seed input)
Allow selection of component by hand
Allow rerun with new random seed
Then un-ICAs the raw
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
from mne import export as export
from mne_icalabel import label_components
import os
from mne.preprocessing import (ICA, corrmap, create_ecg_epochs,
                               create_eog_epochs)

def Auto_ICA(raw, n_comp=None, seed=123, l_freq=1., h_freq=100, removeCFA=False):
    filt_raw = raw.copy().pick(picks=["eeg"]).filter(l_freq=l_freq, h_freq=h_freq, n_jobs="cuda")
    filt_raw = filt_raw.set_eeg_reference("average")
    originalRaw = raw.copy()

    if n_comp == None:
        n_comp = raw.info["nchan"] - 1

    ica = ICA(n_components=n_comp, max_iter='auto', random_state=seed, method="infomax", fit_params=dict(extended=True))
    ica.fit(filt_raw)
    total_explained_ratio = ica.get_explained_variance_ratio(
        filt_raw,
        ch_type='eeg'
    )

    compRatios = {}
    for component in range(n_comp-1):
        explained_var_ratio = ica.get_explained_variance_ratio(
            filt_raw,
            components=[component],
            ch_type='eeg'
        )
        compRatios[component] = 100*explained_var_ratio['eeg']/total_explained_ratio['eeg']

    filt_raw.load_data()

    ic_labels = label_components(filt_raw, ica, method="iclabel")

    labels = ic_labels["labels"]

    print(labels)

    exclude_idx_with_CFA_removed = [idx for idx, label in enumerate(labels) if label not in ["brain"]]
    print("exclude_idx_with_CFA_removed", exclude_idx_with_CFA_removed)

    exclude_idx_without_CFA_removed = [idx for idx, label in enumerate(labels) if label not in ["brain", "heart beat"]]
    print("exclude_idx_without_CFA_removed", exclude_idx_without_CFA_removed)


    if removeCFA:
        icaraw_with_CFA_removed = ica.apply(originalRaw, exclude=exclude_idx_with_CFA_removed)
    else:
        icaraw_with_CFA_removed = None
    icaraw_without_CFA_removed = ica.apply(originalRaw, exclude=exclude_idx_without_CFA_removed)


    return icaraw_with_CFA_removed, icaraw_without_CFA_removed


def Human_ICA(raw, n_comp, seed=69):
    filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    ica = ICA(n_components=n_comp, max_iter='auto', random_state=seed)
    ica.fit(filt_raw)
    ica.plot_sources(inst = filt_raw, show_scrollbars=True)
    ica.plot_components(inst = filt_raw, cmap = "interactive", nrows=3, ncols=7)
    print(ica.exclude)
    ica.apply(raw)
    return raw, seed

def plot_data(data_dict):
    # Set up figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # Set up scrollable plot
    axcolor = 'lightgoldenrodyellow'
    axpos = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
    spos = plt.Slider(axpos, 'Position', 0, 1, valinit=0)

    # Set up checkboxes
    labels = list(data_dict.keys())
    check = CheckButtons(ax=ax, labels=labels, actives=[True] * len(labels))

    # Initialize plot data
    x = np.arange(len(list(data_dict.values())[0]))
    bottom = np.zeros_like(x)
    plots = []

    # Plot data
    for i, (label, data) in enumerate(data_dict.items()):
        p = ax.bar(x, data, bottom=bottom, label=label)
        bottom += data
        plots.append(p)

    # Set up legend and labels
    ax.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    # Set up checkbox update function
    def update_checkboxes(label):
        for i, (l, p) in enumerate(zip(labels, plots)):
            if label == l:
                p[0].set_visible(not p[0].get_visible())
        plt.draw()

    # Attach checkbox update function
    check.on_clicked(update_checkboxes)

    # Set up scrollbar update function
    def update(val):
        pos = spos.val
        ax.set_xlim(pos, pos + 1)
        plt.draw()

    # Attach scrollbar update function
    spos.on_changed(update)

    # Show plot and wait for user to close window
    plt.show()

    # Return checked labels
    return [labels[i] for i, c in enumerate(check.get_status()) if c]
