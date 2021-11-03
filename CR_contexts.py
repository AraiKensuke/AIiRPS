#!/usr/bin/python

import numpy as _N
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import AIiRPS.utils.read_taisen as _rd
from filter import gauKer
from scipy.signal import savgol_filter
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, shuffle_discrete_contiguous_regions, mtfftc
import AIiRPS.skull_plot as _sp
import os
import sys
from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
from filter import gauKer
import GCoh.eeg_util as _eu
import AIiRPS.rpsms as rpsms
import GCoh.preprocess_ver as _ppv

import AIiRPS.constants as _cnst
from AIiRPS.utils.dir_util import getResultFN
import GCoh.datconfig as datconf
import scipy.signal as _ssig

def x_tr_func(arr):
    return _N.log(_N.array(arr)*0.2)
    #return arr

#x_tr_func = _N.log

__1st__ = 0
__2nd__ = 1
__ALL__ = 2

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1
shfl_type_str   = _N.array(["keep_cont", "no_keep_cont"])

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

#"20210526_1416-25",

partIDs  = ["20210526_1416-25", "20210609_1747-07",
            "20210609_1230-28", "20210609_1321-35",
            "20210609_1248-16", "20210526_1358-27",
            "20200108_1642-20", "20200109_1504-32"]

#partIDs  = ["20200812_1252-50", "20200812_1331-06",
#            "20200818_1546-13", "20200818_1603-42"]

#            "20200818_1624-01", "20200818_1644-09"]

# partIDs  = ["20210609_1248-16", "20210609_1747-07",
#             "20210609_1230-28", "20210609_1321-35",
#             "20210609_1248-16", "20210526_1358-27",
#             "20200108_1642-20", "20200109_1504-32",
#             "20200812_1252-50", "20200812_1331-06",
#             "20200818_1546-13", "20200818_1603-42",
#             "20200818_1624-01", "20200818_1644-09"]

#partIDs  = ["20200109_1504-32"]

# ths      = [_N.array([0, 1, 2]), _N.array([0, 1, 2]),
#             _N.array([0]), _N.array([0, 1, 2]),
#             _N.array([0, 1, 2]), _N.array([0, 1, 2]),
#             _N.array([1, 2]), _N.array([0, 1, 2]),
#             _N.array([0, 1, 2]), _N.array([0, 1, 2]),
#             _N.array([0, 1, 2]), _N.array([0]),
#             _N.array([0, 1, 2]), _N.array([0, 1, 2])]            
        
                                         

fnt_tck = 15
fnt_lbl = 17

armv_ver = 1
gcoh_ver =2

manual_cluster = False

Fs=300
win, slideby      = _ppv.get_win_slideby(gcoh_ver)

t_offset = 0  #  ms offset behv_sig_ts
stop_early   = 0#180
ev_n   = 0

show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

frng = [35, 45]
#frng = [15, 25]
#frng  = [32, 48]
#frng  = [34, 46]
#frng = [7, 15]
#frng = [10, 20]
#frng  = [30, 40]
#frng  = [32, 48]

all_pc_pvs = []
all_pc_pvs_flatten = [[], []]

NFFT = 256
N_psd       = 241
all_pwr1    = _N.zeros(N_psd)
all_pwr2    = _N.zeros(N_psd)
all_pwr12   = _N.zeros(N_psd)        

fs_use = _N.linspace(0, 1.2, N_psd)

MULT   = 1.   #  kernel width
lags_sec=50

ii = -1
for partID in partIDs:
    ii += 1
    rpsm_key = rpsms.rpsm_partID_as_key[partID]

    sections = 4

    pikdir     = datconf.getResultFN(datconf._RPS, "%(dir)s/v%(av)d%(gv)d" % {"dir" : rpsm_key, "av" : armv_ver, "gv" : gcoh_ver})
    label          = 71
    outdir         = pikdir#"%(pd)s/%(lb)d_%(st)s_%(sec)d_%(toff)d" % {"pd" : pikdir, "lb" : label, "st" : shfl_type_str[shfl_type], "toff" : t_offset, "sec" : sections}

    lm       = depickle("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d_%(lb)d.dmp" % {"rk" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir, "lb" : label})

    if not os.access(outdir, os.F_OK):
         os.mkdir(outdir)

    _behv_sigs_all   = _N.array(lm["behv"])
         
    bigchg_behv_sig_ts = lm["behv_ts"][0:_behv_sigs_all.shape[1]-stop_early] - t_offset         
    win_gcoh      = lm["win_gcoh"]
    slide_gcoh    = lm["slide_gcoh"]
    win_spec      = lm["win_spec"]
    slide_spec    = lm["slide_spec"]
    eeg_date      = lm["gcoh_fn"]
    ts_spectrograms = lm["ts_spectrograms"]
    ts_gcoh         = lm["ts_gcoh"]#[:-1]
    L_gcoh  = ts_gcoh.shape[0]

    thin          = False
    startOff      = __1st__
    half          = __2nd__
    startOff      = startOff if thin else __1st__
    skip          = 2 if thin else 1

    eeg_smp_dt   = lm["eeg_smp_dt"]
    ###########    This treats the first start_offset moves as non-existent
    

    ignore_stored = False

    ###  BIGCHG
    _behv_sigs_all   = _N.array(lm["behv"])

    #fig = _plt.figure(figsize=(14, 5))

    ic = -1

    labels = ["cond=WTL", "cond=HUM hand", "cond=AI hand"]


    beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
    endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]

    be_inds = _N.array(_N.linspace(beginInd, endInd, sections+1), dtype=_N.int)

    t0 = beginInd
    t1 = (endInd - beginInd)//2 + beginInd
    t2 = endInd
    
    for context in [_cnst._WTL]:#, _cnst._HUMRPS, _cnst._AIRPS]:
        ic += 1
        use_behv     = _N.array([_cnst._WTL, _cnst._AIRPS])#, _cnst._AIRPS])
        _behv_sigs   = _N.sum(_behv_sigs_all[use_behv], axis=0)

        _bigchg_behv_sigs    = _behv_sigs[0:_behv_sigs.shape[0]-stop_early]
        bigchg_behv_sigs    = _bigchg_behv_sigs

        bigchg_behv_sig_ts = lm["behv_ts"][0:_behv_sigs.shape[0]-stop_early] - t_offset

        avg_behv_sig = bigchg_behv_sigs  # 12/5/2020
        cwtl_avg_behv_sig_interp = _N.interp(ts_gcoh, bigchg_behv_sig_ts/1000., avg_behv_sig)

    #     _plt.plot(cwtl_avg_behv_sig_interp + ic*0.5, label=labels[ic])
    # _plt.ylim(-0.5, 1.7)
    # _plt.yticks([])
    # _plt.legend()
    # _plt.savefig("CRs")

    

    hnd_dat = lm["hnd_dat"]
    
    mn_mvtm = _N.mean(_N.diff(hnd_dat[:, 3])) / 1000  #  move duration (in seconds)
    mn_mvtm1 = _N.mean(_N.diff(hnd_dat[0:hnd_dat.shape[0]//2, 3])) / 1000  #  move duration (in seconds)
    sd_mvtm1 = _N.std(_N.diff(hnd_dat[0:hnd_dat.shape[0]//2, 3])) / 1000
    mn_mvtm2 = _N.mean(_N.diff(hnd_dat[hnd_dat.shape[0]//2:, 3])) / 1000  #  move duration (in seconds)
    sd_mvtm2 = _N.std(_N.diff(hnd_dat[hnd_dat.shape[0]//2:, 3])) / 1000  #  move duration (in seconds)

    y = cwtl_avg_behv_sig_interp - _N.min(cwtl_avg_behv_sig_interp)
    fs, pwr = _ssig.welch(y[t0:t2], fs=(mn_mvtm/(slideby/300.)), nfft=NFFT)
    lpwr = _N.log(pwr)
    i1Hz      = _N.where((fs[0:-1] < 0.5) & (fs[1:] >= 0.5))[0][0]
    lpwr -= lpwr[i1Hz]
    all_pwr12 += _N.interp(fs_use, fs, lpwr)

    fs, pwr = _ssig.welch(y[t0:t1], fs=(mn_mvtm/(slideby/300.)), nfft=NFFT)
    lpwr = _N.log(pwr)
    i1Hz      = _N.where((fs[0:-1] < 0.5) & (fs[1:] >= 0.5))[0][0]
    lpwr -= lpwr[i1Hz]
    all_pwr1 += _N.interp(fs_use, fs, lpwr)

    
    fs, pwr = _ssig.welch(y[t1:t2], fs=(mn_mvtm/(slideby/300.)), nfft=NFFT)
    lpwr = _N.log(pwr)
    i1Hz      = _N.where((fs[0:-1] < 0.5) & (fs[1:] >= 0.5))[0][0]
    lpwr -= lpwr[i1Hz]
    all_pwr2 += _N.interp(fs_use, fs, lpwr)


_plt.plot(x_tr_func(fs_use[1:]), all_pwr12[1:], color="black", marker=".", ms=2)
_plt.axvline(x=x_tr_func(0.083), ls="--")
_plt.axvline(x=x_tr_func(0.091), ls=":")        
_plt.axvline(x=x_tr_func(0.1), ls="--")    
_plt.axvline(x=x_tr_func(0.111), ls=":")
_plt.axvline(x=x_tr_func(0.125), ls="--")
_plt.axvline(x=x_tr_func(0.143), ls=":")
_plt.axvline(x=x_tr_func(0.1666), ls="--")
_plt.axvline(x=x_tr_func(0.2), ls=":")
_plt.axvline(x=x_tr_func(0.25), ls="--")
_plt.axvline(x=x_tr_func(0.33333), ls=":")
_plt.axvline(x=x_tr_func(0.5), ls="--")
_plt.axvline(x=x_tr_func(1), ls=":")
_plt.xticks(x_tr_func([0.083, 0.091, 0.1, 0.111, 0.125, 0.143, 0.1666, 0.2, 0.25, 0.3333, 0.5, 1]),
                [r"$\frac{1}{12}$",  r"$\frac{1}{11}$",  r"$\frac{1}{10}$",  r"$\frac{1}{9}$",  r"$\frac{1}{8}$",  r"$\frac{1}{7}$",  r"$\frac{1}{6}$",  r"$\frac{1}{5}$",  r"$\frac{1}{4}$",  r"$\frac{1}{3}$",  r"$\frac{1}{2}$",  r"$\frac{1}{1}$"], fontsize=20)
_plt.xlim(_N.log(0.07*0.2), _N.log(0.6*0.2))

pkl = {}
pkl["fs"] =  fs_use[1:]
pkl["all_pwr12"] = all_pwr12[1:]
pkl["all_pwr1"]  = all_pwr1[1:]
pkl["all_pwr2"]  = all_pwr2[1:]

dmp = open("dCRs.dmp", "wb")
pickle.dump(pkl, dmp, -1)
dmp.close()

