#!/usr/bin/python

import numpy as _N
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import AIiRPS.utils.read_taisen as _rd
from filter import gauKer
from scipy.signal import savgol_filter
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, mtfftc
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


__1st__ = 0
__2nd__ = 1
__ALL__ = 2

_ME_WTL = 0
_ME_RPS = 1
covs    = _N.array(["WTL", "RPS"])

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
#dat  = "Aug122020_13_30_23"
dat   = "Aug182020_15_45_27"
#dat  = "Aug122020_12_52_44"
dat      = "Jan082020_17_03_48"

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
win, slideby      = _ppv.get_win_slideby(gcoh_ver)

t_offset = 0  #  ms offset behv_sig_ts
stop_early   = 0#180
ev_n   = 0

gk_std = 1
gk = gauKer(gk_std)
gk /= _N.sum(gk)
show_shuffled = False
rvrs   = False

process_keyval_args(globals(), sys.argv[1:])
######################################################3


srvrs = "_revrsd" if rvrs else ""
sshf        = "_sh" if show_shuffled else ""
rpsmdir     = getResultFN(dat)
pikdir     = getResultFN("%(dir)s/v%(av)d%(gv)d" % {"dir" : dat, "av" : armv_ver, "gv" : gcoh_ver})
outdir     = pikdir

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("stop_early %d" % stop_early)
print("t_offset %d" % t_offset)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

print("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d.dmp" % {"rk" : rpsm_key, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir})
lm       = depickle("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d.dmp" % {"rk" : rpsm_key, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : pikdir})


s_offset = "off%d_" % t_offset if t_offset != 0 else ""
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
if rvrs:
     bigchg_behv_sigs = _N.array(_behv_sigs[::2].tolist() + _behv_sigs[1::2].tolist())[::-1]
else:
     bigchg_behv_sigs    = _bigchg_behv_sigs
#bigchg_behv_sigs    = _bigchg_behv_sigs[::-1] if rvrs else _bigchg_behv_sigs
#bigchg_behv_sig_ts = lm["behv_ts"][0:_behv_sigs.shape[0]-stop_early] - t_offset
bigchg_behv_sig_ts = lm["behv_ts"][0:_behv_sigs.shape[0]-stop_early] - t_offset

#hnd_latst_offset = _N.where(higchg_behv_sig_ts[0] == hnd_dat[:, 3])[0][0]

real_evs = lm["EIGVS"]   #  shape different
nChs     = real_evs.shape[3]
print("L_gcoh   %d" % L_gcoh)
print(real_evs.shape)

#TRs      = _N.array([1, 1, 3, 10, 30, 40, 50, 60])  # more tries for higher K
TRs      = _N.array([1, 1, 10, 20, 30, 40, 50, 60])  # more tries for higher K

fs_gcoh = lm["fs"]
fs_spec = lm["fs_spectrograms"]
#
#frngs = [[12, 18], [20, 25], [28, 35], [35, 42]]
#frngs = [[12, 18], [20, 25], [28, 35], [35, 42], [38, 45], ]
#frngs = [[25, 35]]
#frngs = [[40, 50]]
#frngs = [[38, 45]]
#frngs = [[12, 18]]
frngs = [[35, 40]]
#frngs = [[20, 30]]
#frngs  = [[15, 22]]

frngs = [[35, 47]]
#frngs = [[38, 45]]
#frngs  = [[35, 42]]
#frngs =[[30, 35]]
#frngs = [[20, 35]]
#frngs = [[30, 38]]
#frngs = [[34, 41], [33, 40], [35, 42], [36, 43]]
#frngs = [[18, 25]]
#frngs = [[33, 40], [34, 41], [35, 42], [36, 43], [37, 44]]

#frngs = [[8, 12], [12, 18]]
#frngs = [[12, 18], [20, 25], [28, 35], [35, 42]]


clrs  = ["black", "orange", "blue", "green", "red", "lightblue", "grey", "pink", "yellow"]

results_bundle = {}

fi = 0

pl_num      = -1

fp = open("%(od)s/corr_out" % {"od" : outdir}, "w")

summary = {}

lags_sec=12
slideby_sec = slideby/300
xticksD = [-20, -10, 0, 10, 20]
lags = int(lags_sec / slideby_sec)+1  #  (slideby/300)*lags
SHFLS= 50
local_shf_win = 60   #  number gcoh samples  // L_gcoh//local_shf_win ~ 23 seconds

sections = 7
xcs  = _N.empty((sections, SHFLS+1, 2*lags+1))
shps = _N.empty((sections, SHFLS+1), dtype=_N.int)
number_state_obsvd = _N.zeros(sections, dtype=_N.int)

cwtl = 0

all_localmaxs = []
all_localmins = []
all_localthemaxs = []
all_localthemins = []
all_xcs       = []
all_xcs_r       = []
all_acs       = []
all_shps      = []


for frng in frngs:
     summary_for_range = {}
     fi += 1
     fL = frng[0]#
     fH = frng[1]

     fp.write("*************** doing %(fL)d  %(fH)d\n" % {"fL" : fL, "fH" : fH})

     irngs = _N.where((fs_spec > fL) & (fs_spec < fH))[0]
     sL    = irngs[0]
     sH    = irngs[-1]    

     irngs = _N.where((fs_gcoh > fL) & (fs_gcoh < fH))[0]
     iL    = irngs[0]
     iH    = irngs[-1]    

     minK    = 1
     maxK    = 8
     try_Ks  = _N.arange(minK, maxK+1)
     #TRs      = _N.array([1, 1, 3, 5, 10, 15, 20, 25, 25])  # more tries for higher K
     TRs      = _N.array([1, 4, 10, 20, 30, 40, 50, 60, 70])  # more tries for higher K


     nStates, _rmpd_lab = find_or_retrieve_GMM_labels(dat, "%(gf)s_gcoh%(evn)d_%(w)d_%(s)d_v%(av)d%(gv)d" % {"gf" : dat, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, real_evs[ev_n], iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=try_Ks, TRs=TRs, manual_cluster=manual_cluster, ignore_stored=ignore_stored, do_pca=True, min_var_expld=0.95)

     stateBin          = _N.zeros(L_gcoh, dtype=_N.int)

     hlfOverlap = 1#int((win/slideby)*0.5)
     sss = _N.empty(SHFLS+1)

     ###  BIGCHG
     print("!!!!")
     print(ts_gcoh.shape)
     #cwtl_avg_behv_sig_interp = _N.empty(ts_gcoh.shape[0])
     avg_behv_sig = bigchg_behv_sigs  # 12/5/2020
     #avg_behv_sig = bigchg_behv_sigs[1:]
     print("----")
     print(bigchg_behv_sig_ts.shape)
     print(avg_behv_sig.shape)

     #cwtl_avg_behv_sig_interp = _N.interp(ts_gcoh, bigchg_behv_sig_ts/1000., avg_behv_sig)
     rps_times = _N.zeros(ts_gcoh.shape[0])
     rps_times[_N.array((bigchg_behv_sig_ts/1000) / slideby_sec, dtype=_N.int)] = 1

     ts_gcoh, bigchg_behv_sig_ts/1000.

     # beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
     # endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]
     # endIndM         = beginInd + (endInd - beginInd)//2

     beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
     endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]

     be_inds = _N.array(_N.linspace(beginInd, endInd, sections+1), dtype=_N.int)
     print("!!!!!!!!!!!!!!!!!!!!")
     print(be_inds)


     time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)
     T = time_lags[-1]

     fig = _plt.figure(figsize=(14, 10))
     _plt.suptitle("%(dat)s   ev %(evn)d  frng %(fr)s  %(rvr)s" % {"fr" : str(frng), "dat" : dat, "evn" : ev_n, "rvr" : srvrs})

     for ns in range(nStates):
          rmpd_lab = _rmpd_lab #shift_correlated_shuffle(_rmpd_lab, low=2, high=4, local_shuffle=True, local_shuffle_pcs=(4*sections)) if shf > 0 else _rmpd_lab
          stateInds = _N.where(rmpd_lab == ns)[0]
          
          stateBin[:] = 0
          stateBin[stateInds] = 1   #  neural signal

          k_stateBin = _N.convolve(stateBin, gk, mode="same")
          k_rps_ts = _N.convolve(rps_times, gk, mode="same")


          for hlf in range(sections):
               i0 = be_inds[hlf]
               i1 = be_inds[hlf+1]
               ac1 = _eu.autocorrelate(k_stateBin[i0:i1] - _N.mean(k_stateBin[i0:i1]), lags)
               ac2 = _eu.autocorrelate(k_rps_ts[i0:i1] - _N.mean(k_rps_ts[i0:i1]), lags)
               ac1[1][lags] = 0
               ac2[1][lags] = 0
               std1 = _N.std(ac1[1][lags+3:])
               std2 = _N.std(ac2[1][lags+2:])
               fig.add_subplot(nStates, sections, sections*ns+hlf+1)
               if ns == 0:
                    _plt.title("section %d" % (hlf+1))
               if hlf == 0:
                    _plt.ylabel("pattern %d" % ns)

               #_plt.plot(time_lags[0:lags-1], (ac1[1][0:lags-1] - _N.mean(ac1[1]))*4 + 0.2, lags, color="black")
               _plt.plot(time_lags[lags+3:], (ac1[1][lags+3:] - _N.mean(ac1[1]))*(0.4/std1), lags, color="black")
               #_plt.plot(time_lags[0:lags-1], ac2[1][0:lags-1] - _N.mean(ac2[1]), lags, color="grey")
               _plt.plot(time_lags[lags+2:], (ac2[1][lags+2:] - _N.mean(ac2[1]))*0.4/std2, lags, color="grey")
               _plt.xticks(range(0, 9))
               _plt.ylim(-1, 2.1)
               _plt.grid(ls=":")
               
               sts    = _N.where(stateBin[i0:i1] == 1)[0]
               rps_ts = _N.where(rps_times[i0:i1] == 1)[0]
               _plt.scatter((sts / (i1-i0))*T, 1.8*_N.ones(len(sts)), s=1, color="red")
               _plt.scatter((rps_ts / (i1-i0))*T, 2.04*_N.ones(len(rps_ts)), s=1, color="blue")


     _plt.savefig("%(od)s/acorr_out_%(sec)d_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d%(sr)s" % {"1" : fL, "2" : fH, "dat" : dat, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "sec" : sections, "od" : outdir, "evn" : ev_n, "sr" : srvrs}, transparent=True)
