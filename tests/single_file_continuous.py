import quantities as pq
from evoked.base import Recording
from evoked.io import process_single_file, load_single_file
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import matplotlib as mpl


recordings = {
    "sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_ieeg.vhdr": Recording(
        id="patient 31",
        stimulus=[41],
        order="grouped",      
        event_label="electrical_stimulation",
        repeats=440,
        layout="continuous",
        stimulus_unit=pq.uA,
        block_index=0,
    )
}

ccep = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\ccep_data\sub-ccepAgeUMCU31\ses-1\ieeg\sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_ieeg.vhdr"
abf = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\evoked_push\evoked\src\evoked\data\2025_03_02_0000.abf"

# df = process_single_file(filename=ccep, recordings=recordings, epoch=[0,30],target_frequency=100.0)

# print(df)

raw = load_single_file(ccep, 0)

def plot_raw_stacked(raw, channels, start=52.0, stop=72.0, offset=None, event_label="electrical_stimulation"):
    sfreq = raw.info["sfreq"]
    picks = raw.copy().pick(channels)

    start_idx = int(start * sfreq)
    stop_idx = int(stop * sfreq)

    data = picks.get_data(start=start_idx, stop=stop_idx)

    times = picks.times[start_idx:stop_idx]

    if offset is None:
        # scale = np.median(
        # np.percentile(data, 95, axis=1)
        # - np.percentile(data, 5, axis=1)
        # )
        offset = 1.25 * np.median(np.ptp(data, axis=1))
        #offset = 2 * scale

    fig, ax = plt.subplots(figsize=(12, 1.25 * len(channels)))

    for i, ch in enumerate(picks.ch_names):
        ax.plot(times, data[i] - i * offset, lw=0.5, color="black")
        ax.text(times[0]-0.1, -i * offset, f"ch {i+1}", ha="right", va="center")

    for onset, label in zip(raw.annotations.onset, raw.annotations.description):
        if label == event_label and start <= onset <= stop:
            #ax.axvline(onset, color="red", lw=0.3, linestyle="dashed")
            ax.plot(
                onset,
                1.01,
                marker="v",
                color="red",
                markersize=5,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


fig, ax = plot_raw_stacked(raw, channels=['T05', 'T06', 'T07', 'T08'])
fig.savefig("multichannel_fig1_new1.png", dpi=600, bbox_inches="tight")


block = load_single_file(abf, 0)

def plot_neo_block(block, channel=0, rc_params=None):
    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(8,5))

        for i, segment in enumerate(block.segments):
            signal = segment.analogsignals[channel]

            time = np.asarray(signal.times.rescale("s")).squeeze()
            time = time - time[0]

            value = np.asarray(signal).squeeze()
            value = np.clip(value, -3.5, 3)

            xmin = 0.0
            xmax = 0.015

            mask = (time >= xmin) & (time <= xmax)
            ax.plot(time[mask], value[mask], color="black", lw=0.5, alpha=0.7)

        # stimulus marker at 0.1 ms
        ax.plot(
            0.0005,
            1.01,
            marker="v",
            color="red",
            markersize=5,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

        # text = AnchoredText("Segments 1-33", loc='upper right', frameon=False)
        # ax.add_artist(text)

        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel(
        #     f"Response ({block.segments[0].analogsignals[channel].units})"
        # )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax

fig1, ax1 = plot_neo_block(block)
fig1.savefig("segments_fig1_new1.png",dpi=600, bbox_inches="tight")