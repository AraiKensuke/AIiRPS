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

_WTL = 0
_HUMRPS = 1
_AIRPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1
covs    = _N.array(["WTL", "RPS"])
shfl_type_str   = _N.array(["keep_cont", "no_keep_cont"])

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

gk_fot = gauKer(2)
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
        mnMaxes = _N.mean(_N.diff(maxes))
        mnInt += mnMaxes
        nTerms += 1
    if (len(mins) >= 2):
        mnMins  = _N.mean(_N.diff(mins))
        mnInt += mnMaxes
        nTerms += 1
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

#dat     = "Apr112020_21_26_01"#"Apr312020_16_53_03"
#dat      = "Jan082020_16_56_08"
# dat      = "Jan092020_14_55_38"
#dat     = "Jan092020_14_00_00"#"Apr312020_16_53_03"
#
#dat     = "Aug122020_13_19_23"#"Apr312020_16_53_03"
#dat     = "May042020_22_23_04"#"Apr312020_16_53_03"
#dat     = "May052020_21_08_01"
#dat     = "May082020_23_34_58"
#dat =      "Apr152020_20_34_20"
#dat =      "Apr242020_00_00_00"
#dat =      "Apr182020_22_02_03"
#dat =      "Apr242020_16_53_03"


#dat  = "Aug182020_16_44_18"

#"20210526_1416-25"
partIDs  = ["20210526_1318-12",
            "20210526_1358-27", "20210526_1503-39",            
            "20210609_1321-35", "20210609_1747-07",
            "20210609_1248-16", "20210609_1230-28", 
            "20200108_1642-20", "20200109_1504-32",
            "20200812_1252-50", "20200812_1331-06",
            "20200818_1546-13", "20200818_1603-42",
            "20200818_1624-01", "20200818_1644-09"]

partIDs  = ["20210609_1321-35", "20210609_1747-07",
            "20210609_1248-16",
            "20200109_1504-32",
            "20200818_1603-42",]


partIDs  = ["20200109_1504-32"]
ths      = [_N.array([0, 1, 2]), _N.array([0, 1, 2]),
            _N.array([0]), _N.array([0, 1, 2]),
            _N.array([0, 1, 2]), _N.array([0, 1, 2]),
            _N.array([1, 2]), _N.array([0, 1, 2]),
            _N.array([0, 1, 2]), _N.array([0, 1, 2]),
            _N.array([0, 1, 2]), _N.array([0]),
            _N.array([0, 1, 2]), _N.array([0, 1, 2])]            

#partIDs = ["20200108_1642-20", "20200812_1331-06"]

#partIDs  = ["20200818_1546-13", "20200818_1644-09", "20210609_1230-28", "20210609_1321-35", "20210609_1248-16", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200109_1504-32"]

#partIDs  = ["20200818_1546-13", "20200818_1644-09", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", ]

#dat     = "Jan092020_15_05_39"#"Apr312020_16_53_03"
#dat     = "Aug182020_16_02_49"
#dat  = "Aug122020_13_30_23"
#dat   = "Aug182020_15_45_27"
#dat  = "Aug122020_12_52_44"
#dat      = "Jan082020_17_03_48"

#dat   = "Aug182020_16_25_28"
#dat  = "May142020_23_16_34"   #  35 seconds    #  15:04:32
#dat  = "May142020_23_31_04"   #  35 seconds    #  15:04:32

MULT   = 1   #  kernel width

fnt_tck = 11
fnt_lbl = 13

armv_ver = 1
gcoh_ver =2

manual_cluster = False

Fs=300
win, slideby      = _ppv.get_win_slideby(gcoh_ver)

t_offset = 0  #  ms offset behv_sig_ts
stop_early   = 0#180
ev_n   = 0

show_shuffled = False
lblsz = 18
tksz  = 15
process_keyval_args(globals(), sys.argv[1:])
#######################################################

frng = [32, 48]
#frng = [7, 15]
#frng = [35, 45]

fp_O = open("times_O_%(1)d_%(2)d.txt" % {"1" : frng[0], "2" : frng[1]}, "w")
fp_S = open("times_S_%(1)d_%(2)d.txt" % {"1" : frng[0], "2" : frng[1]}, "w")
fp_m_gi_ts = open("times_gmint_v_ACts_%(1)d_%(2)d.txt" % {"1" : frng[0], "2" : frng[1]}, "w")

for partID in partIDs:
    rpsm_key = rpsms.rpsm_partID_as_key[partID]

    sections = 4

    pikdir     = datconf.getResultFN(datconf._RPS, "%(dir)s/v%(av)d%(gv)d" % {"dir" : rpsm_key, "av" : armv_ver, "gv" : gcoh_ver})
    label          = 53
    outdir         = pikdir#"%(pd)s/%(lb)d_%(st)s_%(sec)d_%(toff)d" % {"pd" : pikdir, "lb" : label, "st" : shfl_type_str[shfl_type], "toff" : t_offset, "sec" : sections}

    print("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d_%(lb)d.dmp" % {"rk" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir, "lb" : label})
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
    #use_behv     = _N.array([_WTL, _HUMRPS, _AIRPS])
    #use_behv     = _N.array([_ME_WTL, _ME_RPS])
    #use_behv     = _N.array([_ME_RPS])
    _behv_sigs   = _N.sum(_behv_sigs_all[use_behv], axis=0)

    #_behv_sigs   = _behv_sigs_all[0]

    _bigchg_behv_sigs    = _behv_sigs[0:_behv_sigs.shape[0]-stop_early]
    hlfL = _bigchg_behv_sigs.shape[0]//2

    real_evs = lm["EIGVS"]   #  shape different
    nChs     = real_evs.shape[3]
    print("L_gcoh   %d" % L_gcoh)

    #TRs      = _N.array([1, 1, 3, 10, 30, 40, 50, 60])  # more tries for higher K
    TRs      = _N.array([1, 1, 10, 20, 30, 40, 50, 60])  # more tries for higher K

    fs_gcoh = lm["fs"]
    fs_spec = lm["fs_spectrograms"]

    #  smoothing the 1vR autocorrelation.  Because 
    hnd_dat = lm["hnd_dat"]
    mn_mvtm = _N.mean(_N.diff(hnd_dat[:, 3])) / 1000  #  move duration (in seconds)
    num_samples_smooth = 300/slideby * mn_mvtm * MULT  #  how many bins of 1vR is that?

    gkInt = int(_N.round(num_samples_smooth))
    gk = gauKer(gkInt)    #  not interested in things of single-move timescales
    gk /= _N.sum(gk)


    t_btwn_rps = _N.mean(_N.diff(hnd_dat[:, 3]))
    clrs  = ["black", "orange", "blue", "green", "red", "lightblue", "grey", "pink", "yellow"]

    fi = 0

    pl_num      = -1

    lags_sec=50
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

    nStates, _rmpd_lab = find_or_retrieve_GMM_labels(datconf._RPS, partID, "%(gf)s_gcoh%(evn)d_%(w)d_%(s)d_v%(av)d%(gv)d" % {"gf" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, real_evs[ev_n], iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=try_Ks, TRs=TRs, manual_cluster=manual_cluster, ignore_stored=ignore_stored, do_pca=True, min_var_expld=0.95)

    rmpd_lab  = None

    all_behv = []
    dNGS_timescales = _N.empty(sections)
    GCOH_timescales = _N.empty((nStates, sections))

    for flip in [False, True]:
         bigchg_behv_sigs    = _bigchg_behv_sigs[::-1] if flip else _bigchg_behv_sigs
         all_behv.append(_N.array(bigchg_behv_sigs))

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

              dNGS_timescales[hlf] = find_osc_timescale(dNGS_ac, lags)
              toobig = _N.where(dNGS_acs[hlf] > 2.8)[0]
              dNGS_acs[hlf, toobig] = 2.8   #  cut it off


         for ns in range(nStates):
              rmpd_lab = _rmpd_lab

              stateInds = _N.where(rmpd_lab == ns)[0]

              stateBin[:] = 0
              stateBin[stateInds] = 1   #  neural signal
              fstateBin = _N.convolve(stateBin, gk, mode="same")

              for hlf in range(sections):
                   i0 = be_inds[hlf]
                   i1 = be_inds[hlf+1]
                   in_this_sec = _N.where(stateBin[i0:i1] == 1)[0]

                   print("%(0)d  %(1)d     %(sb)d" % {"0" : i0, "1" : i1, "sb" : len(in_this_sec)})

                   if (_N.mean(stateBin[i0:i1]) == 0):
                        print("state %(ns)d not observed in hlf %(h)d" % {"ns" : ns, "h" : hlf})
                        fac01s[ns, hlf] = 0
                   else:
                        #ac01 = _eu.autocorrelate_whatsthisfor(fstateBin[i0:i1], lags, pieces=1)
                        x, ac01 = _eu.autocorrelate(fstateBin[i0:i1], lags)
                        fac01 = ac01
                        #fac01 = _N.convolve(gk, ac01, mode="same")
                        fac01_AMP = _N.std(fac01[0:lags-5])
                        if not fac01_AMP == 0:
                             fac01_mn  = _N.mean(fac01[0:lags-5])

                             fac01s[ns, hlf] = (fac01 - fac01_mn)/fac01_AMP
                        else:
                             print("not enough to calculate AC")
                             fac01s[ns, hlf] = 0


         # diffs = _N.zeros((nStates, sections, sections))
         # for ns in range(nStates):
         #      for hlf1 in range(sections):
         #           for hlf2 in range(sections):
         #                _plt.plot(time_lags, (fac01 - fac01_mn)/fac01_AMP, color="grey")
         #                _plt.plot(time_lags, dNGS_acs[hlf], color="blue")
         #                _plt.ylim(-2, 3.5)

         #                y = fac01s[ns, hlf1, 0:lags-5] - dNGS_acs[hlf2, 0:lags-5]

         #                diffs[ns, hlf1, hlf2] = _N.sqrt(_N.dot(y, y))


                        #  for state 1



         fig = _plt.figure(figsize=(14, 12.))

         for ns in range(nStates):
              rmpd_lab = _rmpd_lab

              stateInds = _N.where(rmpd_lab == ns)[0]

              stateBin[:] = 0
              stateBin[stateInds] = 1   #  neural signal

              for hlf in range(sections):
                   i0 = be_inds[hlf]
                   i1 = be_inds[hlf+1]
                   i102 = (i1-i0)/2
                   #  we want to weight observations higher that have states are not too sparse or too dense.  
                   #print("i102:  %(102)d   %(st)d" % {"102" : i102, "st" : len(_N.where(rmpd_lab[i0:i1] == ns)[0])})
                   wgt_states[ns, hlf] = (i102 - _N.abs(i102 - len(_N.where(rmpd_lab[i0:i1] == ns)[0]))) / (i1-i0)

                   if flip == False:
                       GCOH_timescales[ns, hlf] = find_osc_timescale(fac01s[ns, hlf], lags)  #  not yet cut off

                   ax = fig.add_subplot(sections, nStates, hlf*nStates+ns+1)
                   #ax.set_facecolor("#999999")
                   toobig = _N.where(fac01s[ns, hlf] > 2.8)[0]
                   fac01s[ns, hlf, toobig] = 2.8
                   _plt.plot(time_lags, fac01s[ns, hlf], color="black", lw=3)

                   #_plt.plot(time_lags, dNGS_acs[hlf], color="#6666FF", lw=3)
                   _plt.plot(time_lags, dNGS_acs[hlf], color="blue", lw=4.5)
                   pc, pv = _ss.pearsonr(fac01s[ns, hlf][0:lags-10], dNGS_acs[hlf][0:lags-10])


                   _plt.ylim(-2.8, 3.7)

                   tLow  = i0
                   tHigh = i1


                   ### timescale of 1 RPS game (compare to AC)
                   _plt.plot([0, 0], [3.05, 3.65], color="blue", lw=1)
                   _plt.plot([t_btwn_rps/1000, t_btwn_rps/1000], [3.05, 3.65], color="blue", lw=1)               
                   _plt.plot([0, t_btwn_rps/1000], [3.35, 3.35], color="blue", lw=1)

                   ### scatter plot of 1-vs-rest
                   xs = _N.linspace(0, 1, tHigh - tLow) * 2*lags_sec - lags_sec  #  for scatter plot of coherence pattern time series
                   shp_on = _N.where(stateBin[i0:i1] == 1)[0]
                   _plt.scatter(xs[shp_on], _N.ones(shp_on.shape[0])*-2.6, color="black", s=2)
                   #(tHigh-tLow)*(slideby/Fs)  = seconds  (duration 1-vs-rest tmsrs)
                   tACwin = (lags_sec / ((tHigh-tLow)*(slideby/Fs))) * (2*lags_sec)

                   _plt.plot([-lags_sec, tACwin-lags_sec], [-2.4, -2.4], color="black")
                   _plt.plot([-lags_sec, -lags_sec], [-2.55, -2.25], color="black")
                   _plt.plot([tACwin-lags_sec, tACwin-lags_sec], [-2.55, -2.25], color="black")


                   _plt.axvline(x=-50, ls=":", color="grey", lw=2)
                   _plt.axvline(x=-25, ls=":", color="grey", lw=2)
                   _plt.axvline(x=0, ls=":", color="grey", lw=2)
                   _plt.axvline(x=25, ls=":", color="grey", lw=2)
                   _plt.axvline(x=50, ls=":", color="grey", lw=2)
                   _plt.axhline(y=0, ls=":", color="grey", lw=2)

                   _plt.yticks([])
                   #if hlf < sections -1:
                   #     _plt.xticks([])
                   #else:
                   _plt.xticks(list(range(-50, 51, 25)), fontsize=tksz)
                   _plt.xlim(-51, 51)
                   _plt.xlabel("time lag (s)", fontsize=lblsz)
                   # if hlf == 0:
                   #      _plt.title("pat %(n)d  pc %(pc).3f" % {"n" : ns, "pc" : pc}, fontsize=16)
                   # else:
                   #      _plt.title("pc %(pc).3f" % {"pc" : pc}, fontsize=16)

                   if ns == 0:
                        _plt.ylabel("epoch %d" % hlf)
              _plt.suptitle("%(dat)s   label=%(lab)d  hz: %(hz)s   flip=%(flp)s" % {"dat" : partID, "flp" : str(flip), "lab" : label, "hz" : str(frng)}, fontsize=18)
              fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.92, hspace=0.55)

              _plt.savefig("%(od)s/f_acorr_behv_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d%(flp)s" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "lb" : label, "flp" : ("_flip" if flip else "")}, transparent=False)


    flat_GCOH_ts = GCOH_timescales.flatten()
    flat_dNGS_ts = dNGS_timescales.flatten()

    prs  = []
    for h in range(sections):
        if dNGS_timescales[h] > 0:
            stInds = _N.where(GCOH_timescales[:, h] > 5)[0]
            if len(stInds) >= 1:
                mn = _N.sum((GCOH_timescales[stInds, h] * wgt_states[stInds, h]) / _N.sum(wgt_states[stInds, h]))
                #prs.append([_N.mean(GCOH_timescales[stInds, h]), dNGS_timescales[h]])
                prs.append([mn, dNGS_timescales[h]])
    for i in range(len(prs)):
        fp_O.write("%(g).1f  %(C).1f\n" % {"g" : prs[i][0], "C" : prs[i][1]})
    print(",,,,")
    for i in range(len(prs)):
        fp_S.write("%(g).1f  %(C).1f\n" % {"g" : prs[i][0], "C" : prs[(i+1)%len(prs)][1]})
    fp_O.write("###########################\n")
    fp_S.write("###########################\n")    


    ths = _N.where(flat_GCOH_ts > 0)[0]
    ts_GCOH = _N.mean(flat_GCOH_ts[ths])
    ths = _N.where(flat_dNGS_ts > 0)[0]
    ts_dNGS = _N.mean(flat_dNGS_ts[ths])

    #print("mean movetime   GC and CR")
    fp_m_gi_ts.write("%(mmt).1f   %(tsGC).1f    %(tsCR).1f\n" % {"tsGC" : (ts_GCOH*(slideby/300)), "tsCR" : (ts_dNGS*(slideby/300)), "mmt" : mn_mvtm})

fp_O.close()
fp_S.close()
fp_m_gi_ts.close()
