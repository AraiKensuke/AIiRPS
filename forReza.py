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
partIDs  = ["20210526_1416-25", "20210609_1747-07", "20210609_1230-28", "20210609_1321-35", "20210609_1248-16", "20210526_1358-27", "20200108_1642-20", "20200109_1504-32", "20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]
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

MULT   = 1.   #  kernel width
lags_sec=50

_fp = open("times_O_%(1)d_%(2)d_%(lags)d_%(mult).1f.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : MULT}, "w")
_fp_F = open("times_O_%(1)d_%(2)d_%(lags)d_%(mult).1f_flip.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : MULT}, "w")
_fp_m_gi_ts = open("times_gmint_v_ACts_%(1)d_%(2)d_%(lags)d_%(mult).1f.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : MULT}, "w")
_fp_m_gi_ts_F = open("times_gmint_v_ACts_%(1)d_%(2)d_%(lags)d_%(mult).1f_flip.txt" % {"1" : frng[0], "2" : frng[1], "lags" : lags_sec, "mult" : MULT}, "w")

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

    nStates, _rmpd_lab = find_or_retrieve_GMM_labels(datconf._RPS, partID, "%(gf)s_gcoh%(evn)d_%(w)d_%(s)d_v%(av)d%(gv)d" % {"gf" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, real_evs[ev_n], iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=try_Ks, TRs=TRs, manual_cluster=manual_cluster, ignore_stored=ignore_stored, do_pca=True, min_var_expld=0.95)

    nObsv_of_pattern_in_sect = _N.zeros((sections, nStates), dtype=_N.int)
    use_this                 = _N.zeros((sections, nStates), dtype=_N.bool)
    rmpd_lab  = None

    fig = _plt.figure(figsize=(14, 13))
    
    for flip in [False, True]:
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
              #fstateBin = stateBin

              # gk4 = gauKer(4)
              # gk8 = gauKer(8)
              # gk4 /= _N.sum(gk4)
              # gk8 /= _N.sum(gk8)
              # fstateBin4 = _N.convolve(stateBin, gk4, mode="same")
              # fstateBin8 = _N.convolve(stateBin, gk8, mode="same")
              

              for hlf in range(sections):
                   i0 = be_inds[hlf]
                   i1 = be_inds[hlf+1]
                   in_this_sec = _N.where(stateBin[i0:i1] == 1)[0]
                   nObsv_of_pattern_in_sect[hlf, ns] = len(in_this_sec)
                                 

                   #print("%(0)d  %(1)d     %(sb)d" % {"0" : i0, "1" : i1, "sb" : len(in_this_sec)})

                   if (_N.mean(stateBin[i0:i1]) == 0):
                        print("state %(ns)d not observed in hlf %(h)d" % {"ns" : ns, "h" : hlf})
                        fac01s[ns, hlf] = 0
                   else:
                        #ac01 = _eu.autocorrelate_whatsthisfor(fstateBin[i0:i1], lags, pieces=1)
                        #fig = _plt.figure()
                        #_plt.plot(fstateBin4[i0:i1])
                        #_plt.plot(fstateBin8[i0:i1]+1)                        
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
                   state_in_this_hlf  = _N.where(stateBin[i0:i1] == 1)[0]

                   #if flip == False:
                   GCOH_timescales[ns, hlf] = find_osc_timescale(fac01s[ns, hlf, lags-5:], lags)  #  not yet cut off

                   ax = fig.add_subplot(sections, nStates, hlf*nStates+ns+1)
                   if len(state_in_this_hlf) > 0:
                       sqz =   len(state_in_this_hlf) / (state_in_this_hlf[-1] - state_in_this_hlf[0])
                   else:
                       sqz = 1

                   enough_obs = nObsv_of_pattern_in_sect[hlf, ns] / (i1-i0) > 0.1   # am I seeing this pattern enough to think its AC is meaningful?
                   too_many_obs = nObsv_of_pattern_in_sect[hlf, ns] / (i1-i0) > 0.95   # am I seeing this pattern enough to think its AC is meaningful?

                   use_this[hlf, ns] = (enough_obs and (sqz < 0.9)) and (not too_many_obs)
                   
                   if use_this[hlf, ns]:
                       ax.set_facecolor("#DDDDDD")
                       toobig = _N.where(fac01s[ns, hlf] > 2.8)[0]
                       fac01s[ns, hlf, toobig] = 2.8

                       offset = 3 if flip else 0
                       #_plt.plot(time_lags, dNGS_acs[hlf], color="#6666FF", lw=3)
                       if flip == False:
                           _plt.plot(time_lags, dNGS_acs[hlf]+offset, color="red", lw=2)
                       else:
                           _plt.plot(time_lags, dNGS_acs[hlf]+offset, color="blue", lw=2)
                       _plt.plot(time_lags, fac01s[ns, hlf]+offset, color="black", lw=3)                       
                       pc, pv = _ss.pearsonr(fac01s[ns, hlf][0:lags-20], dNGS_acs[hlf][0:lags-20])


                   _plt.ylim(-2.8, 6.2)

                   tLow  = i0
                   tHigh = i1

                   ### timescale of 1 RPS game (compare to AC)
                   _plt.plot([0, 0], [5.8, 6.1], color="blue", lw=1)
                   _plt.plot([t_btwn_rps/1000, t_btwn_rps/1000], [5.8, 6.1], color="blue", lw=1)               
                   _plt.plot([0, t_btwn_rps/1000], [5.95, 5.95], color="blue", lw=1)
                   
                   ### scatter plot of 1-vs-rest
                   xs = _N.linspace(0, 1, tHigh - tLow) * 2*lags_sec - lags_sec  #  for scatter plot of coherence pattern time series
                   shp_on = _N.where(stateBin[i0:i1] == 1)[0]
                   _plt.scatter(xs[shp_on], _N.ones(shp_on.shape[0])*-2.6, color="black", s=2)
                   #(tHigh-tLow)*(slideby/Fs)  = seconds  (duration 1-vs-rest tmsrs)
                   tACwin = (lags_sec / ((tHigh-tLow)*(slideby/Fs))) * (2*lags_sec)
                   
                   _plt.plot([-lags_sec, tACwin-lags_sec], [-2.4, -2.4], color="black")
                   _plt.plot([-lags_sec, -lags_sec], [-2.55, -2.25], color="black")
                   _plt.plot([tACwin-lags_sec, tACwin-lags_sec], [-2.55, -2.25], color="black")
                   
                   
                   _plt.axvline(x=-30, ls=":", color="grey", lw=1)
                   _plt.axvline(x=-15, ls=":", color="grey", lw=1)
                   _plt.axvline(x=0, ls=":", color="grey", lw=1)
                   _plt.axvline(x=15, ls=":", color="grey", lw=1)
                   _plt.axvline(x=30, ls=":", color="grey", lw=1)
                   _plt.axhline(y=0, ls=":", color="grey", lw=1)
                    
                   _plt.yticks([])
                   if hlf < sections -1:
                        _plt.xticks([])
                   else:
                        _plt.xticks(list(range(-30, 31, 15)))
                        _plt.xlabel("time lag (s)")
                   #if hlf == 0:
                   #     _plt.title("pat %(n)d  pc %(pc).3f" % {"n" : ns, "pc" : pc}, fontsize=16)
                   #else:
                   #     _plt.title("pc %(pc).3f" % {"pc" : pc}, fontsize=16)

                   if ns == 0:
                        _plt.ylabel("epoch %d" % hlf)
              _plt.suptitle("%(dat)s   label=%(lab)d  hz: %(hz)s   flip=%(flp)s" % {"dat" : partID, "flp" : str(flip), "lab" : label, "hz" : str(frng)}, fontsize=18)
         flat_GCOH_ts = GCOH_timescales.flatten()
         flat_dNGS_ts = dNGS_timescales.flatten()

         prs  = []
         for h in range(sections):
             if dNGS_timescales[h] > 0:
                 mns = []
                 for ns in range(nStates):
                     if use_this[h, ns]:
                         if GCOH_timescales[ns, h] > 5:
                             mns.append(GCOH_timescales[ns, h])
                 if len(mns) > 0:
                     #print(mns)
                     prs.append([_N.mean(mns), dNGS_timescales[h]])
         for i in range(len(prs)):
             fp.write("%(g).1f  %(C).1f\n" % {"g" : prs[i][0], "C" : prs[i][1]})
         fp.write("###########################\n")

         thsGCOH = _N.where(flat_GCOH_ts > 0)[0]
         ts_GCOH = _N.mean(flat_GCOH_ts[thsGCOH])
         thsdNGS = _N.where(flat_dNGS_ts > 0)[0]
         ts_dNGS = _N.mean(flat_dNGS_ts[thsdNGS])
         if len(thsGCOH) > 0:
             fp_m_gi_ts.write("%(mmt).1f   %(tsGC).1f    %(tsCR).1f\n" % {"tsGC" : (ts_GCOH*(slideby/300)), "tsCR" : (ts_dNGS*(slideby/300)), "mmt" : mn_mvtm})


    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.92)

    _plt.savefig("%(od)s/f_acorr_behv_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d.png" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "lb" : label}, transparent=False)
    _plt.savefig("f_acorr_behv_out_%(partID)s_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d.png" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "lb" : label, "partID" : partID}, transparent=False)
    _plt.close()
    #os.system("ln -sf %(od)s/f_acorr_behv_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d.png  %(pid)s_acorr.png" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "lb" : label, "pid" : partID})

_fp.close()
_fp_m_gi_ts.close()
_fp_F.close()
_fp_m_gi_ts_F.close()
