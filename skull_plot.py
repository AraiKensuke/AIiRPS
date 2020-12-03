import mne
import numpy as _N
import matplotlib.pyplot as _plt

_RPS        = 1
_STROOP     = 2
_GONOGO     = 3

def do_skull_plot(vec20, fname, dataset):
    # Create some dummy metadata
    sfreq = 1

    if (dataset == _RPS) or (dataset == _STROOP):
        n_channels = 21
        ch_names=["P3", "C3", "F3", "Fz", "F4",
                  "C4", "P4", "Cz", "Pz", "A1",
                  "Fp1","Fp2","T3", "T5", "O1",
                  "O2", "F7", "F8", "A2", "T6",
                  "T4"]
        ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg"]
    else:
        n_channels = 31

        ch_names = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "T5'", "T6'", "O1'", "O2'", "P3\"", "P4\"", "PZ\"", "OZ", "I", "CB1\"", "CB2\"", "CB1", "CB2"]
        ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info.set_montage("standard_1020")

    data = _N.random.randn(1, n_channels, sfreq * 1)
    #X =X(:,[1:8 10:21]);
    data[0, 0:8, 0] = vec20[0:8]
    data[0, 9:21, 0] = vec20[8:20]
    data[0, 8, 0]   = 0.5*(vec20[0] + vec20[6])
    custom_epochs = mne.EpochsArray(data, info)

    # Create the Evoked object
    tmin = 0
    evoked_array = mne.EvokedArray(data[0], info, tmin)

    # Generate some random data: 10 epochs, 5 channels, 2 seconds per epoch
    mp = evoked_array.plot_topomap(ch_type='eeg', times=[0], colorbar=True)
    mp.savefig(fname)


"""
"""
"""
"""
"""
"""
def do_skull_plot_all_EVs(all_vecs, ps, ch_names, fname, eeg_date, fL, fH, montage="standard_1020", close=False):
    # Create some dummy metadata
    n_channels = len(ch_names)

    if montage == "standard_1020":
        n_channels = 20
        ch_names=["P3", "C3", "F3", "Fz", "F4",
                  "C4", "P4", "Pz", "A1", "Fp1",
                  "Fp2","T3", "T5", "O1", "O2", 
                  "F7", "F8", "A2", "T6", "T4"]
        ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg"]
    elif montage == "standard_1020_19ch":
        n_channels = 19
        ch_names = ["Fp1", "Fp2", "F7", "F3","Fz", "F4", 
                    "F8", "T3", "C3", "Cz", "C4", "T4", 
                    "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
        ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg",]
        montage = "standard_1020"
    else:
        n_channels = 32

        ch_names = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "T5'", "T6'", "O1'", "O2'", "P3\"", "P4\"", "PZ\"", "OZ", "I", "CB1\"", "CB2\"", "CB1", "CB2", "CZ"]
        ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]

    print(n_channels)
    sfreq = 1
    ch_types=["eeg"] * n_channels
#              "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
#              "eeg", "eeg", "eeg", "eeg", "eeg"]
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info.set_montage(montage)

    nEVs = all_vecs.shape[1]
    print(nEVs)
    print(all_vecs.shape)

    data = _N.zeros((1, n_channels, sfreq * nEVs))
    print(data.shape)
    #X =X(:,[1:8 10:21]);
    for ns in range(nEVs):
        # data[0, 0:8, ns] = all_vec20[0:8, ns]
        # data[0, 9:21, ns] = all_vec20[8:20, ns]
        # data[0, 8, ns]   = 0.5*(all_vec20[0, ns] + all_vec20[6, ns])
        if montage == "standard_1020":
            data[0, 0:n_channels, ns] = all_vecs[0:n_channels, ns]
        else:
            data[0, 0:n_channels-1, ns] = all_vecs[0:n_channels-1, ns]
    if montage != "standard_1020":
        data[0, n_channels-1, ns] = _N.mean(all_vecs[:, ns], axis=0)
    custom_epochs = mne.EpochsArray(data, info)

    # Create the Evoked object
    tmin = 0
    #data[0, :, nEVs-1] = 0
    evoked_array = mne.EvokedArray(data[0], info, tmin)

    all_evs = _N.arange(nEVs)


    fig, axes = _plt.subplots(figsize=(1.9*nEVs, 2.8), nrows=1, ncols=nEVs)

    # Here we look at EEG channels, and use a custom head sphere to get all the
    # sensors to be well within the drawn head surface
    #for axes_row, ch_type in zip(axes, ('mag', 'eeg')):
    #for ax, extr in zip(axes_row, extrapolations):

    for ev in range(nEVs):
        if nEVs > 1:
            ax = axes[ev]
        else:
            ax = axes
        evoked_array.plot_topomap(all_evs[ev], ch_type="eeg", axes=ax, show=False, colorbar=False)
#0.1, ch_type=ch_type, size=2, extrapolate=extr,
#                            axes=ax, show=False, colorbar=False,
#                            sphere=(0., 0., 0., 0.09))
        ax.set_title('ev %(ev1)d[%(p)d]' % {"ev1" : (ev+1), "p" : ps[ev]})
    _plt.suptitle("%(fn)s   [%(fL)d-%(fH)d]Hz" % {"fn" : eeg_date, "fL" : fL, "fH" : fH})
    _plt.savefig(fname)
    if close:
        _plt.close()


    # Generate some random data: 10 epochs, 5 channels, 2 seconds per epoch
    #mp = evoked_array.plot_topomap(all_evs, ch_type='eeg', colorbar=True, ncols=nEVs)
    #mp.savefig(fname)
