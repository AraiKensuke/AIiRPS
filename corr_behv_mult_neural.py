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


#partID  = "20210609_1747-07"
#partID  = "20210609_1230-28"
#partID  = "20210609_1248-16"
#partID = "20210609_1321-35"
partID = "20200109_1504-32"


#dat     = "Aug182020_16_02_49"
#dat  = "Aug122020_13_30_23"
#dat   = "Aug182020_15_45_27"
#dat  = "Aug122020_12_52_44"
#dat      = "Jan082020_17_03_48"

#dat   = "Aug182020_16_25_28"
#dat  = "May142020_23_16_34"   #  35 seconds    #  15:04:32
#dat  = "May142020_23_31_04"   #  35 seconds    #  15:04:32

fnt_tck = 15
fnt_lbl = 17

rpsm_key = rpsms.rpsm_partID_as_key[partID]
armv_ver = 1
gcoh_ver =2

manual_cluster = False

Fs=300
dpss_bw = 7
win, slideby      = _ppv.get_win_slideby(gcoh_ver)

t_offset = 0  #  ms offset behv_sig_ts
stop_early   = 0#180
ev_n   = 0

show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

sections = 4
SHFLS= 50
shfl_type=_SHFL_KEEP_CONT

pikdir     = datconf.getResultFN(datconf._RPS, "%(dir)s/v%(av)d%(gv)d" % {"dir" : rpsm_key, "av" : armv_ver, "gv" : gcoh_ver})

print(pikdir)


label          = 71
outdir         = "%(pd)s/%(lb)d_%(st)s_%(sec)d_%(toff)d" % {"pd" : pikdir, "lb" : label, "st" : shfl_type_str[shfl_type], "toff" : t_offset, "sec" : sections}

print(outdir)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("stop_early %d" % stop_early)
print("t_offset %d" % t_offset)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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

use_behv     = _N.array([_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS])
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

mn_mvtm = _N.mean(_N.diff(hnd_dat[:, 3])) / 1000  #  move duration (in seconds)
num_samples_smooth = (300/slideby * mn_mvtm)  #  how many bins of 1vR is that?

gkInt = int(_N.round(num_samples_smooth))
gk = gauKer(gkInt)    #  not interested in things of single-move timescales
gk /= _N.sum(gk)

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
TRs      = _N.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36])

nStates, _rmpd_lab = find_or_retrieve_GMM_labels(datconf._RPS, partID, "%(gf)s_gcoh%(evn)d_%(w)d_%(s)d_v%(av)d%(gv)d" % {"gf" : partID, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, real_evs[ev_n], iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=try_Ks, TRs=TRs, manual_cluster=manual_cluster, ignore_stored=ignore_stored, do_pca=True, min_var_expld=0.95)

rmpd_lab  = None

gk = gauKer(2)
gk /= _N.sum(gk)

yLo = -0.6
yHi =  0.6
all_behv = []
for dat_mod in [[False, False], [True, False], [False, True], [True, True]]:
#for dat_mod in [[True, False], [False, True], [True, True]]:
     rvrs = dat_mod[0]
     hlfs_intrchg = dat_mod[1]
     srvrs = ""#

     if rvrs and hlfs_intrchg:
          srvrs = "_hi_revrsd"
     elif rvrs and not hlfs_intrchg:
          srvrs = "_revrsd"
     elif not rvrs and hlfs_intrchg:
          srvrs = "_hi"

     if rvrs and not hlfs_intrchg:
          bigchg_behv_sigs = _N.array(_bigchg_behv_sigs[::-1])
          #bigchg_behv_sigs = _N.array(_bigchg_behv_sigs)
          #_N.random.shuffle(bigchg_behv_sigs)
     elif not rvrs and hlfs_intrchg:
          bigchg_behv_sigs = _N.array(_bigchg_behv_sigs[hlfL:].tolist() + _bigchg_behv_sigs[0:hlfL].tolist())
          #bigchg_behv_sigs = _N.array(_bigchg_behv_sigs)
          #_N.random.shuffle(bigchg_behv_sigs)
     elif rvrs and hlfs_intrchg:
          bigchg_behv_sigs = _N.array(_N.array(_bigchg_behv_sigs[hlfL:].tolist() + _bigchg_behv_sigs[0:hlfL].tolist())[::-1])
          #bigchg_behv_sigs = _N.array(_bigchg_behv_sigs)
          #_N.random.shuffle(bigchg_behv_sigs)
     else:
          bigchg_behv_sigs    = _bigchg_behv_sigs

     #bigchg_behv_sigs = _N.convolve(bigchg_behv_sigs, gk, mode="same")
     all_behv.append(_N.array(bigchg_behv_sigs))
     
     #bigchg_behv_sigs    = _bigchg_behv_sigs[::-1] if rvrs else _bigchg_behv_sigs
     #bigchg_behv_sig_ts = lm["behv_ts"][0:_behv_sigs.shape[0]-stop_early] - t_offset
     bigchg_behv_sig_ts = lm["behv_ts"][0:_behv_sigs.shape[0]-stop_early] - t_offset

     all_acs       = []
     all_shps      = []

     

     xcs  = _N.empty((sections, SHFLS+1, 2*lags+1))
     shps = _N.empty((sections, SHFLS+1), dtype=_N.int)
     number_state_obsvd = _N.zeros(sections, dtype=_N.int)

     stateBin          = _N.zeros(L_gcoh, dtype=_N.int)
     _stateBin          = _N.zeros(L_gcoh, dtype=_N.int)

     all_xcs_r  = _N.empty((nStates, sections, SHFLS+1, 2*lags+1))

     ###  BIGCHG
     print("!!!!")
     print(ts_gcoh.shape)
     #cwtl_avg_behv_sig_interp = _N.empty(ts_gcoh.shape[0])
     avg_behv_sig = bigchg_behv_sigs  # 12/5/2020
     #avg_behv_sig = bigchg_behv_sigs[1:]
     print("----")
     print(bigchg_behv_sig_ts.shape)
     print(avg_behv_sig.shape)

     cwtl_avg_behv_sig_interp = _N.interp(ts_gcoh, bigchg_behv_sig_ts/1000., avg_behv_sig)

     # beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
     # endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]
     # endIndM         = beginInd + (endInd - beginInd)//2

     beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
     endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]

     be_inds = _N.array(_N.linspace(beginInd, endInd, sections+1), dtype=_N.int)
     print("!!!!!!!!!!!!!!!!!!!!")
     print(be_inds)

     fig  = _plt.figure(figsize=(3.*nStates, sections*2.5))
     pc, pv = _ss.pearsonr(bigchg_behv_sigs, _bigchg_behv_sigs)
     
     _plt.suptitle("%(dat)s   ev %(evn)d  frng %(fr)s  %(rvr)s  %(st)s   pc  %(pc).4f    pv  %(pv).4f" % {"fr" : str(frng), "dat" : partID, "evn" : ev_n, "rvr" : srvrs, "st" : shfl_type_str[shfl_type], "pc" : pc, "pv" : pv})

     time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)
     for ns in range(nStates):
          for shf in range(SHFLS+1):
               if shfl_type == _SHFL_NO_KEEP_CONT:
                    rmpd_lab = shift_correlated_shuffle(_rmpd_lab, low=int(win/slideby-1), high=int(win/slideby+1), local_shuffle=True, local_shuffle_pcs=(2*sections)) if shf > 0 else _rmpd_lab
               elif shfl_type == _SHFL_KEEP_CONT:
                    rmpd_lab = shuffle_discrete_contiguous_regions(_rmpd_lab, local_shuffle=True, local_shuffle_pcs=sections) if shf > 0 else _rmpd_lab

               stateInds = _N.where(rmpd_lab == ns)[0]

               _stateBin[:] = 0
               _stateBin[stateInds] = 1   #  neural signal
               stateBin = _N.convolve(_stateBin, gk, mode="same")

               for hlf in range(sections):
                    i0 = be_inds[hlf]
                    i1 = be_inds[hlf+1]
                    if shf == 0:
                         number_state_obsvd[hlf] = _N.sum(stateBin[i0:i1])

                    if (_N.mean(stateBin[i0:i1]) == 0):
                         print("ZERO MEAN STATE")
                         xc = _N.zeros(2*lags+1)
                    else:
                         xc = _eu.crosscorrelate(stateBin[i0:i1] - _N.mean(stateBin[i0:i1]), cwtl_avg_behv_sig_interp[i0:i1] - _N.mean(cwtl_avg_behv_sig_interp[i0:i1]), lags)
                    dxc = _N.diff(xc)
                    pks = _N.where((dxc[0:-1] > 0) & (dxc[1:] < 0))[0]
                    trhs= _N.where((dxc[0:-1] < 0) & (dxc[1:] > 0))[0]
                    shps[hlf, shf] = len(pks) + len(trhs)

                    xcs[hlf, shf] = xc
               #print(xcs[0, 0])


          rmpd_lab = _rmpd_lab
          stateInds = _N.where(rmpd_lab == ns)[0]
          stateBin[:] = 0
          stateBin[stateInds] = 1   #  neural signal

          for hlf in range(sections):
               i0 = be_inds[hlf]
               i1 = be_inds[hlf+1]
               tLow  = i0
               tHigh = i1
               xs = _N.linspace(0, 1, tHigh - tLow) * 2*lags_sec - lags_sec  #  for scatter plot of coherence pattern time series


               dy = _N.diff(xcs[hlf, 0])
               # if _N.sum(_N.isnan(xcs[hlf, 0]))== 0:
               #      locmaxs = _N.where((dy[0:-1] > 0) & (dy [1:] < 0))[0]
               #      themax  = _N.where(xcs[hlf, 0] == _N.max(xcs[hlf, 0, locmaxs]))[0]
               #      locmins = _N.where((dy[0:-1] < 0) & (dy [1:] > 0))[0]


                    # print(len(locmins))
                    # themin  = _N.where(xcs[hlf, 0] == _N.min(xcs[hlf, 0, locmins]))[0]
                    # all_localmaxs.extend(locmaxs)
                    # all_localmins.extend(locmins)
                    # all_localthemaxs.append(themax)
                    # all_localthemins.append(themin)

               fig.add_subplot(sections, nStates, hlf*nStates+ns+1)

               #_plt.plot(time_lags, xcs[hlf, 0], marker=".", ms=2, color=("red" if (hlf == 2) else "black"), lw=1)
               #_plt.plot(time_lags, xcs[hlf, 0], marker=".", ms=2, color="black", lw=1)
               _plt.plot(time_lags, xcs[hlf, 0], marker=".", ms=2, color="black", lw=1)
               shp_on = _N.where(stateBin[i0:i1] == 1)[0]
               _plt.scatter(xs[shp_on], _N.ones(shp_on.shape[0])*(yLo + (yHi-yLo)*0.1), color="red", s=4)

               srtd = _N.sort(xcs[hlf, 1:], axis=0)
               _plt.fill_between(time_lags, srtd[int(SHFLS*0.025)], srtd[int(SHFLS*0.975)], alpha=0.3, color="blue")
               _plt.ylim(yLo, yHi)

               _plt.axhline(y=0, ls=":")
               #_plt.axvline(x=0, ls=":")

               if hlf == sections - 1:
                    _plt.xticks(ticks=xticksD, fontsize=fnt_tck)
               else:
                    _plt.xticks(ticks=xticksD, labels=([""] * len(xticksD)), fontsize=fnt_tck)
               _plt.xlim(-lags_sec, lags_sec)
               _plt.yticks(fontsize=fnt_tck)
               _plt.axvline(x=-20, ls=":", color="grey", lw=1)
               _plt.axvline(x=-10, ls=":", color="grey", lw=1)
               _plt.axvline(x=0, ls="--", color="#111111", lw=2.5)
               _plt.axvline(x=10, ls=":", color="grey", lw=1)
               _plt.axvline(x=20, ls=":", color="grey", lw=1)

               morePks = _N.where(shps[hlf, 0] >= shps[hlf, 1:])[0]
               if hlf == sections-1:
                    _plt.xlabel("lag (s)", fontsize=fnt_lbl)
               if ns == 0:
                    _plt.ylabel("epc %d" % hlf, fontsize=fnt_lbl)
               _plt.title(len(shp_on))
          #pc, pv = _ss.pearsonr(xcs[0, 0, lags//2:2*lags-1-lags//2], xcs[1, 0, lags//2:2*lags-1-lags//2])
          #print("state %(ns)d  pc %(pc).3f" % {"ns" : ns, "pc" : pc})
          all_xcs_r[ns] = _N.array(xcs)

          #all_shps.append(_N.array(shps))
          all_shps.append(_N.array(number_state_obsvd))

     #scov = ""
     #for ub in range(len(use_behv)-1):
     #     scov += covs[use_behv[ub]] + ","
     #scov += covs[use_behv[-1]]
     scov = "ALL"
     fig.subplots_adjust(wspace=0.4, hspace=0.3, left=0.08, bottom=0.12)

     _plt.savefig("%(od)s/xcorr_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d_%(ub)s_%(sr)s" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "sr" : srvrs, "ub" : scov, "lb" : label}, transparent=False)

     pickle_put = {}
     pickle_put["all_xcs_r"] = all_xcs_r
     pickle_put["all_shps"] = all_shps
     pickle_put["time_lags"] = time_lags
     pickle_put["shfl_type"] = shfl_type_str
     pickle_put["t_offset"] = t_offset
     pickle_put["SHFLS"] = SHFLS

     print("*************************xcorr_out")
     print("%(od)s/xcorr_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d%(sr)s.dmp" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "sr" : srvrs})
     dmp = open("%(od)s/xcorr_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d%(sr)s.dmp" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "sr" : srvrs, "lb" : label}, "wb")
     pickle.dump(pickle_put, dmp, -1)
     dmp.close()

     # alls = _N.array(all_shps)

     # for ns in range(nStates):
     #      print("state %d" % ns)
     #      for hlf in range(3):
     #           morePks = _N.where(alls[ns, hlf, 0] >= alls[ns, hlf, 1:])[0]
     #           print(morePks)
#_N.savetxt("themaxs%s" % dat, _N.array(all_localthemaxs)[:, 0])
#_N.savetxt("themins%s" % dat, _N.array(all_localthemins)[:, 0])

# savedir = "Results/%(dsf)s/v%(av)d%(gv)d" % {"dsf" : dat, "av" : armv_ver, "gv" : gcoh_ver}
# dmp = open("%(sd)s/%(rk)s_%(w)d_%(s)d_corr_output_v%(av)d%(gv)d.dmp" % {"rk" : rpsm_key, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "sd" : savedir}, "wb")

# pickle.dump(summary, dmp, -1)
# dmp.close()



# SHFLS=0
# stateBin          = _N.zeros(L_gcoh, dtype=_N.int)

# # beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
# # endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]
# # endIndM         = beginInd + (endInd - beginInd)//2

# beginInd        = _N.where(ts_gcoh < bigchg_behv_sig_ts[0]/1000)[0][-1]
# endInd          = _N.where(ts_gcoh > bigchg_behv_sig_ts[-1]/1000)[0][0]

# be_inds = _N.array(_N.linspace(beginInd, endInd, sections+1), dtype=_N.int)
# print("!!!!!!!!!!!!!!!!!!!!")
# print(be_inds)

# fig  = _plt.figure(figsize=(3.*nStates, sections*2.5))
# _plt.suptitle("%(dat)s   ev %(evn)d  frng %(fr)s  %(rvr)s" % {"fr" : str(frng), "dat" : partID, "evn" : ev_n, "rvr" : srvrs})

# time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)
# for ns in range(nStates):
#      ac_this_state = []
#      rmpd_lab = _rmpd_lab
#      stateInds = _N.where(rmpd_lab == ns)[0]

#      stateBin[:] = 0
#      stateBin[stateInds] = 1   #  neural signal

#      for hlf in range(sections):
#           fig.add_subplot(sections, nStates, hlf*nStates+ns+1)
#           i0 = be_inds[hlf]
#           i1 = be_inds[hlf+1]

#           ac = _eu.autocorrelate_whatsthisfor(stateBin[i0:i1], lags, pieces=1)
#           ac_this_state.append(_N.array(ac))
#           _plt.plot(time_lags, ac, color="black", lw=1)
#           _plt.ylim(-0.2, 0.4)
#           _plt.xlim(-lags_sec, lags_sec)
#           shp_on = _N.where(stateBin[i0:i1] == 1)[0]
#           _plt.scatter(xs[shp_on], _N.ones(shp_on.shape[0])*-0.18, color="red", s=4)
#           if hlf == sections - 1:
#                _plt.xticks(ticks=xticksD, fontsize=fnt_tck)
#           else:
#                _plt.xticks(ticks=xticksD, labels=([""] * len(xticksD)), fontsize=fnt_tck)

#           _plt.axvline(x=-20, ls=":", color="grey", lw=1)
#           _plt.axvline(x=-10, ls=":", color="grey", lw=1)
#           _plt.axvline(x=0, ls=":", color="grey", lw=1)
#           _plt.axvline(x=10, ls=":", color="grey", lw=1)
#           _plt.axvline(x=20, ls=":", color="grey", lw=1)
#           _plt.axhline(y=0, ls=":", color="grey", lw=1)
#      all_acs.append(ac_this_state)

# fig.subplots_adjust(wspace=0.4, hspace=0.3, left=0.08, bottom=0.12)
# _plt.savefig("%(od)s/acorr_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d%(sr)s" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "sr" : srvrs, "lb" : label}, transparent=True)

# pickle_put = {}
# pickle_put["all_acs"] = all_acs
# pickle_put["time_lags"] = time_lags

# dmp = open("%(od)s/acorr_out_%(evn)d_%(w)d_%(sl)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d%(sr)s.dmp" % {"1" : fL, "2" : fH, "dat" : partID, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n, "sr" : srvrs, "lb" : label}, "wb")
# pickle.dump(pickle_put, dmp, -1)
# dmp.close()


