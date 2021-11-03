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

gk_fot = gauKer(1)
gk_fot = gk_fot / _N.sum(gk_fot)

def find_osc_timescale(ACfun, lags):
    ACfun   = _N.convolve(ACfun, gk_fot, mode="same")  #  smooth out AC function
    dACfun  = _N.diff(ACfun)
    maxes = _N.where((dACfun[0:-1] > 0) & (dACfun[1:] <= 0))[0]
    mins  = _N.where((dACfun[0:-1] < 0) & (dACfun[1:] >= 0))[0]
    mnMaxes = -1
    mnMins  = -1
    mnInt   = 0
    nTerms  = 0
    if (len(maxes) >= 2):
        mnMaxes = _N.diff(maxes)[0]  #  intervals
        #mnMaxes = _N.mean(_N.diff(maxes))  #  intervals
        mnInt += mnMaxes
        nTerms += 1
    # if (len(mins) >= 2):
    #     mnMins  = _N.mean(_N.diff(mins))
    #     mnInt += mnMaxes
    #     nTerms += 1
    if nTerms > 0:
        return mnInt / nTerms
    else:
        return -1
"""
def find_osc_timescale(ACfun, lags):
    ACfun   = _N.convolve(ACfun, gk_fot, mode="same")
    hlf_ACfun = ACfun[lags:]
    dACfun  = _N.diff(hlf_ACfun)
    maxes = _N.where((dACfun[0:-1] > 0) & (dACfun[1:] <= 0))[0]
    mnMaxes = -1
    mnMins  = -1
    mnInt   = 0
    nTerms  = 0
    if len(maxes) > 0:
        print(maxes[ACfun[maxes].argmax()])
        return maxes[ACfun[maxes].argmax()]
    else:
        return -1
"""

#"20210526_1416-25", 
partIDs  = ["20210609_1747-07"]#, "20210609_1230-28", "20210609_1321-35", "20210609_1248-16", "20210526_1358-27", "20200108_1642-20", "20200109_1504-32", "20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]
#partIDs  = ["20210526_1416-25", "20210526_1318-12",
partIDs  = ["20210526_1416-25"]
partIDs  = ["20210526_1318-12", "20210526_1416-25",
            "20210526_1358-27", "20210526_1503-39",            
            "20210609_1321-35", "20210609_1747-07",
            "20210609_1248-16", "20210609_1230-28", 
            "20200108_1642-20", "20200109_1504-32",
            "20200812_1252-50", "20200812_1331-06",
            "20200818_1546-13", "20200818_1603-42",
            "20200818_1624-01", "20200818_1644-09"]

#partIDs  = ["20210609_1747-07"]#"20210526_1416-25"]#, "20210609_1747-07", "20210609_1230-28", "20210609_1321-35", "20210609_1248-16", "20210526_1358-27", "20200108_1642-20", "20200109_1504-32", "20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]

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

#frng = [35, 45]
#frng = [15, 25]
#frng  = [32, 48]
#frng  = [34, 46]
frng = [7, 15]
#frng = [10, 20]
#frng  = [30, 40]
#frng  = [32, 48]

all_pc_pvs = []
all_pc_pvs_flatten = [[], []]

MULT   = 1   #  kernel width
fxdGK  = None
lags_sec=50

if fxdGK is not None:
    sMULT = "G%d" % fxdGK
else:
    sMULT = "%.1f" % MULT
_fp = open("times_O_%(1)d_%(2)d_%(lags)d_%(mult)s.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : sMULT}, "w")
_fp_F = open("times_O_%(1)d_%(2)d_%(lags)d_%(mult)s_flip.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : sMULT}, "w")
_fp_m_gi_ts = open("times_gmint_v_ACts_%(1)d_%(2)d_%(lags)d_%(mult)s.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : sMULT}, "w")
_fp_m_gi_ts_F = open("times_gmint_v_ACts_%(1)d_%(2)d_%(lags)d_%(mult)s_flip.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : sMULT}, "w")

for partID in partIDs:
    rpsm_key = rpsms.rpsm_partID_as_key[partID]

    sections = 4

    pikdir     = datconf.getResultFN(datconf._RPS, "%(dir)s/v%(av)d%(gv)d" % {"dir" : rpsm_key, "av" : armv_ver, "gv" : gcoh_ver})
    label          = 53
    outdir         = pikdir#"%(pd)s/%(lb)d_%(st)s_%(sec)d_%(toff)d" % {"pd" : pikdir, "lb" : label, "st" : shfl_type_str[shfl_type], "toff" : t_offset, "sec" : sections}

    #print("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d_%(lb)d.dmp" % {"rk" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir, "lb" : label})
    lm       = depickle("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d_%(lb)d.dmp" % {"rk" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir, "lb" : label})

    if not os.access(outdir, os.F_OK):
         os.mkdir(outdir)

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

    use_behv     = _N.array([_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS])
    #use_behv     = _N.array([_ME_WTL, _ME_RPS])
    #use_behv     = _N.array([_ME_RPS])
    _behv_sigs   = _N.sum(_behv_sigs_all[use_behv], axis=0)

    #_behv_sigs   = _behv_sigs_all[0]

    _bigchg_behv_sigs    = _behv_sigs[0:_behv_sigs.shape[0]-stop_early]
    hlfL = _bigchg_behv_sigs.shape[0]//2

    real_evs = lm["EIGVS"]   #  shape different
    nChs     = real_evs.shape[3]

    #TRs      = _N.array([1, 1, 3, 10, 30, 40, 50, 60])  # more tries for higher K
    TRs      = _N.array([1, 1, 10, 20, 30, 40, 50, 60])  # more tries for higher K

    fs_gcoh = lm["fs"]
    fs_spec = lm["fs_spectrograms"]

    #  smoothing the 1vR autocorrelation.  Because 
    hnd_dat = lm["hnd_dat"]
    mn_mvtm = _N.mean(_N.diff(hnd_dat[:, 3])) / 1000  #  move duration (in seconds)
    num_samples_smooth = (300/slideby * mn_mvtm) * MULT  #  how many bins of 1vR is that?

    if fxdGK is not None:
        gkInt = fxdGK
    else:
        gkInt = int(_N.round(num_samples_smooth))
    print("..............      gkInt %d" % gkInt)
    #print("...........  %(dt)s    %(mn).2f   %(gkI)d" % {"dt" : partID, "mn" : mn_mvtm, "gkI" : gkInt})
    gk = gauKer(gkInt)    #  not interested in things of single-move timescales
    gk /= _N.sum(gk)


    t_btwn_rps = _N.mean(_N.diff(hnd_dat[:, 3]))
    clrs  = ["black", "orange", "blue", "green", "red", "lightblue", "grey", "pink", "yellow"]

    fi = 0

    pl_num      = -1

    slideby_sec = slideby/300
    #xticksD = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    xticksD = _N.arange(-lags_sec, lags_sec+1, 10)
    lags = int(lags_sec / slideby_sec)+1  #  (slideby/300)*lags


    cwtl = 0

    fL = frng[0]#
    fH = frng[1]

    irngs = _N.where((fs_spec > fL) & (fs_spec < fH))[0]
    sL    = irngs[0]
    sH    = irngs[-1]    

    irngs = _N.where((fs_gcoh > fL) & (fs_gcoh < fH))[0]
    iL    = irngs[0]
    iH    = irngs[-1]    

    minK    = 1
    maxK    = 8
    try_Ks  = _N.arange(minK, maxK+1)
    TRs      = _N.array([1, 4, 10, 20, 30, 40, 50, 60, 70])  # more tries for higher K

    flip = False
    nStates, _rmpd_lab = find_or_retrieve_GMM_labels(datconf._RPS, partID, "%(gf)s_gcoh%(evn)d_%(w)d_%(s)d_v%(av)d%(gv)d" % {"gf" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, real_evs[ev_n], iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=try_Ks, TRs=TRs, manual_cluster=manual_cluster, ignore_stored=ignore_stored, do_pca=True, min_var_expld=0.95)

    nObsv_of_pattern_in_sect = _N.zeros((sections, nStates), dtype=_N.int)
    use_this                 = _N.zeros((sections, nStates), dtype=_N.bool)
    rmpd_lab  = None

    fig = _plt.figure(figsize=(14, 13))
    
    fp = _fp_F if flip else _fp
    fp_m_gi_ts = _fp_m_gi_ts_F if flip else _fp_m_gi_ts 
    sflip = "" if not flip else "_flip"
    
    dNGS_timescales = _N.empty(sections)
    GCOH_timescales = _N.empty((nStates, sections))
    
    bigchg_behv_sigs    = _bigchg_behv_sigs[::-1] if flip else _bigchg_behv_sigs
    
    bigchg_behv_sig_ts = lm["behv_ts"][0:_behv_sigs.shape[0]-stop_early] - t_offset
    
    stateBin          = _N.zeros(L_gcoh, dtype=_N.int)
    
    
    avg_behv_sig = bigchg_behv_sigs  # 12/5/2020
    cwtl_avg_behv_sig_interp = _N.interp(ts_gcoh, bigchg_behv_sig_ts/1000., avg_behv_sig)
    
    beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
    endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]
    
    be_inds = _N.array(_N.linspace(beginInd, endInd, sections+1), dtype=_N.int)
    time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)
    
    dNGS_acs = _N.empty((sections, 2*lags+1))
    fac01s   = _N.empty((nStates, sections, 2*lags+1))
    
    wgt_states = _N.zeros((nStates, sections))
    
    for hlf in range(sections):
        i0 = be_inds[hlf]
        i1 = be_inds[hlf+1]
        x, dNGS_ac = _eu.autocorrelate(cwtl_avg_behv_sig_interp[i0:i1] - _N.mean(cwtl_avg_behv_sig_interp[i0:i1]), lags)
        acdNGS_AMP = _N.std(dNGS_ac[0:lags-5])
        acdNGS_mn  = _N.mean(dNGS_ac[0:lags-5])
        
        dNGS_acs[hlf] = (dNGS_ac - acdNGS_mn) / acdNGS_AMP
        
        dNGS_timescales[hlf] = find_osc_timescale(dNGS_ac[lags-5:], lags)
        toobig = _N.where(dNGS_acs[hlf] > 2.8)[0]
        dNGS_acs[hlf, toobig] = 2.8   #  cut it off
        
        
    for ns in range(nStates):
        rmpd_lab = _rmpd_lab

        stateInds = _N.where(rmpd_lab == ns)[0]

        stateBin[:] = 0
        stateBin[stateInds] = 1   #  neural signal
        fstateBin = _N.convolve(stateBin, gk, mode="same")
        _plt.plot(stateBin[be_inds[0]:be_inds[-1]] + 4.5*ns, color="black")
        _plt.plot(fstateBin[be_inds[0]:be_inds[-1]] + 4.5*ns + 1.5, color="black")
        dx = (be_inds[-1] - be_inds[0]) / hnd_dat.shape[0]
        _plt.scatter(_N.arange(0, be_inds[-1] - be_inds[0], dx), _N.ones(hnd_dat.shape[0])*(4.5*ns+1.2), s=1, color="black")
    _plt.suptitle(partID)
    _plt.xlim(-1, be_inds[-1]-be_inds[0]+1)
    fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    _plt.savefig("smoothed_GCOH_%(pID)s_%(1)d_%(2)d_%(lags)d_%(mult)s.png" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : sMULT, "pID" : partID})
