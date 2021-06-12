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

from AIiRPS.utils.dir_util import getResultFN
import GCoh.datconfig as datconf

__1st__ = 0
__2nd__ = 1
__ALL__ = 2

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1
covs    = _N.array(["WTL", "RPS"])
shfl_type_str   = _N.array(["keep_cont", "no_keep_cont"])

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

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

dat     = "Jan092020_15_05_39"#"Apr312020_16_53_03"
#dat     = "Aug182020_16_02_49"
dat  = "Aug122020_13_30_23"
#dat   = "Aug182020_15_45_27"
#dat  = "Aug122020_12_52_44"
#dat      = "Jan082020_17_03_48"

#dat   = "Aug182020_16_25_28"
#dat  = "May142020_23_16_34"   #  35 seconds    #  15:04:32
#dat  = "May142020_23_31_04"   #  35 seconds    #  15:04:32

fnt_tck = 15
fnt_lbl = 17

rpsm_key = rpsms.rpsm_eeg_as_key[dat]
armv_ver = 1
gcoh_ver =3

manual_cluster = False

Fs=300
win, slideby, dpss_bw      = _ppv.get_win_slideby(gcoh_ver)

t_offset = 0  #  ms offset behv_sig_ts
stop_early   = 0#180
ev_n   = 0

show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

gk = gauKer(8)    #  probably better to keep this small kernel
gk /= _N.sum(gk)
sections = 7

flip       = True
pikdir     = datconf.getResultFN(datconf._RPS, "%(dir)s/v%(av)d%(gv)d" % {"dir" : dat, "av" : armv_ver, "gv" : gcoh_ver})
label          = 8
outdir         = pikdir#"%(pd)s/%(lb)d_%(st)s_%(sec)d_%(toff)d" % {"pd" : pikdir, "lb" : label, "st" : shfl_type_str[shfl_type], "toff" : t_offset, "sec" : sections}

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("stop_early %d" % stop_early)
print("t_offset %d" % t_offset)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

print("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d_%(lb)d.dmp" % {"rk" : rpsm_key, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir, "lb" : label})
lm       = depickle("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d_%(lb)d.dmp" % {"rk" : rpsm_key, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir, "lb" : label})

if not os.access(outdir, os.F_OK):
     os.mkdir(outdir)

win_gcoh      = lm["win_gcoh"]
slide_gcoh    = lm["slide_gcoh"]
win_spec      = lm["win_spec"]
slide_spec    = lm["slide_spec"]
eeg_date      = lm["gcoh_fn"]
ts_spectrograms = lm["ts_spectrograms"]
ts_gcoh         = lm["ts_gcoh"]#[:-1]
print("!!!!!!!!")
print(ts_gcoh)
print(ts_gcoh.shape)
print("!!!!!!!!")
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
_behv_sigs_all   = _N.array(lm["fbehv"])

use_behv     = _N.array([_ME_WTL])
#use_behv     = _N.array([_ME_WTL, _ME_RPS])
#use_behv     = _N.array([_ME_RPS])
_behv_sigs   = _N.sum(_behv_sigs_all[use_behv], axis=0)

#_behv_sigs   = _behv_sigs_all[0]

_bigchg_behv_sigs    = _behv_sigs[0:_behv_sigs.shape[0]-stop_early]
hlfL = _bigchg_behv_sigs.shape[0]//2

real_evs = lm["EIGVS"]   #  shape different
nChs     = real_evs.shape[3]
print("L_gcoh   %d" % L_gcoh)
print(real_evs.shape)

#TRs      = _N.array([1, 1, 3, 10, 30, 40, 50, 60])  # more tries for higher K
TRs      = _N.array([1, 1, 10, 20, 30, 40, 50, 60])  # more tries for higher K

fs_gcoh = lm["fs"]
fs_spec = lm["fs_spectrograms"]
frng = [32, 48]

hnd_dat = lm["hnd_dat"]
t_btwn_rps = _N.mean(_N.diff(hnd_dat[:, 3]))
clrs  = ["black", "orange", "blue", "green", "red", "lightblue", "grey", "pink", "yellow"]

fi = 0

pl_num      = -1

lags_sec=30
slideby_sec = slideby/300
xticksD = [-20, -10, 0, 10, 20]
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

nStates, _rmpd_lab = find_or_retrieve_GMM_labels(datconf._RPS, dat, "%(gf)s_gcoh%(evn)d_%(w)d_%(s)d_v%(av)d%(gv)d" % {"gf" : dat, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, real_evs[ev_n], iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=try_Ks, TRs=TRs, manual_cluster=manual_cluster, ignore_stored=ignore_stored, do_pca=True, min_var_expld=0.95)

rmpd_lab  = None

all_behv = []
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
     print("!!!!!!!!!!!!!!!!!!!!")
     print(be_inds)

     time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)

     dNGS_acs = _N.empty((sections, 2*lags+1))
     fac01s   = _N.empty((nStates, sections, 2*lags+1))
     for hlf in range(sections):
          i0 = be_inds[hlf]
          i1 = be_inds[hlf+1]
          x, dNGS_ac = _eu.autocorrelate(cwtl_avg_behv_sig_interp[i0:i1] - _N.mean(cwtl_avg_behv_sig_interp[i0:i1]), lags)
          acdNGS_AMP = _N.std(dNGS_ac[0:lags-5])
          acdNGS_mn  = _N.mean(dNGS_ac[0:lags-5])

          dNGS_acs[hlf] = (dNGS_ac - acdNGS_mn) / acdNGS_AMP
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

               if (_N.mean(stateBin[i0:i1]) == 0):
                    print("ZERO MEAN STATE")
                    fac01s[ns, hlf] = 0
               else:
                    ac01 = _eu.autocorrelate_whatsthisfor(stateBin[i0:i1], lags, pieces=1)
                    fac01 = ac01#_N.convolve(gk, ac01, mode="same")
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
     fig = _plt.figure(figsize=(14, 13))

     for ns in range(nStates):
          rmpd_lab = _rmpd_lab

          stateInds = _N.where(rmpd_lab == ns)[0]

          stateBin[:] = 0
          stateBin[stateInds] = 1   #  neural signal

          for hlf in range(sections):
               i0 = be_inds[hlf]
               i1 = be_inds[hlf+1]

               ax = fig.add_subplot(sections, nStates, hlf*nStates+ns+1)
               ax.set_facecolor("#999999")
               toobig = _N.where(fac01s[ns, hlf] > 2.8)[0]
               fac01s[ns, hlf, toobig] = 2.8
               _plt.plot(time_lags, fac01s[ns, hlf], color="black")
               #_plt.plot(time_lags, dNGS_acs[hlf], color="#6666FF", lw=3)
               _plt.plot(time_lags, dNGS_acs[hlf], color="orange", lw=2.5)
               _plt.ylim(-2.8, 3.5)

               tLow  = i0
               tHigh = i1


               ### timescale of 1 RPS game (compare to AC)
               _plt.plot([0, 0], [3.1, 3.4], color="blue", lw=1)
               _plt.plot([t_btwn_rps/1000, t_btwn_rps/1000], [3.1, 3.4], color="blue", lw=1)               
               _plt.plot([0, t_btwn_rps/1000], [3.25, 3.25], color="blue", lw=1)

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
               if hlf == 0:
                    _plt.title("pattern %d" % ns)
               if ns == 0:
                    _plt.ylabel("epoch %d" % hlf)
          _plt.suptitle("%(dat)s   label=%(lab)d  flip=%(flp)s" % {"dat" : dat, "flp" : str(flip), "lab" : label})
          fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.92)

          
          _plt.savefig("%(od)s/f_acorr_behv_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d%(flp)s" % {"1" : fL, "2" : fH, "dat" : dat, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "lb" : label, "flp" : ("_flip" if flip else "")}, transparent=False)
