import numpy as _N
import matplotlib.pyplot as _plt
import rpsms
import GCoh.eeg_util as _eu
import GCoh.preprocess_ver as _ppv
import scipy.stats as _ss
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, mtfftc
import GCoh.datconfig as datconf

from filter import gauKer
import GCoh.skull_plot as _sp
from AIiRPS.utils.dir_util import getResultFN
import os

lbsz=18
tksz=16
def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

def patIDs_for_epoch(keys, epoch):
     patIDs = []
     for k in keys:
          epc, pId = k.split(",")
          if epc == str(epoch):
               patIDs.append(int(pId))
     return _N.array(patIDs)

#{"xc" : [[1, 1, -1], [1, 2, 1], [2, 2, -1], [2, 4, 1], [3, 2, 1], [4, 4, 1], [5, 4, 1]],
#dats = {"Aug122020_12_52_44"}

#  autocorrelograms of dNGS and selected BNPTn's
#  cross correlograms dNGS and BNPTn's
#  show similar time scales

clrs     = ["green", "blue", "red", "orange", "grey"]
#key_dats = dats.keys()
#key_dats = ["Aug182020_16_44_18"]
#key_dats   = ["Aug182020_15_45_27"]
#key_dats=["Aug182020_16_25_28"]
#key_dats = ["Aug182020_16_25_28"]#, "Aug182020_16_44_18"]
key_dats = ["Jan092020_15_05_39"]
#key_dats = ["Aug122020_12_52_44"]
#key_dats = ["Jan082020_17_03_48"]
#key_dats  = ["Aug122020_13_30_23"]
#key_dats = ["Jan082020_17_03_48", "Jan092020_15_05_39", "Aug122020_12_52_44", "Aug122020_13_30_23", "Aug182020_15_45_27"]  # Ken
#key_dats = ["Aug182020_16_25_28", "Aug182020_16_44_18"]  # Ali 


armv_ver = 4
gcoh_ver =3
label    = 5

win, slideby      = _ppv.get_win_slideby(gcoh_ver)

ev_n = 0

#frg      = "35-47"
frg      = "32-48"
#frg      = "38-50"
#frg      = "30-45"
#frg      = "18-25"
#frg      = "15-25"
#frg      = "25-35"

Fs      = 300
lags_sec=30
SHFLS   = 100
slideby_sec = slideby/Fs
lags = int(lags_sec / slideby_sec)+1  #  (slideby/300)*lags
xticksD = [-20, -10, 0, 10, 20]
time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)
xticksD = [-20, -10, 0, 10, 20]
midp = lags

print("lags   %d" % lags)
iexpt = -1

#time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)

gk = gauKer(2)

#all_x= _N.arange(-141, 142)

ch_w_CM, rm_chs, list_ch_names, ch_types = datconf.getConfig(datconf._RPS)
arr_ch_names = _N.array(list_ch_names)

for dat_mod in [[False, False], [True, False], [False, True], [True, True]]:
     rvrs   = dat_mod[0]
     hlfs_intrchg=dat_mod[1]

     srvrs = ""
     if rvrs and hlfs_intrchg:
          srvrs = "_hi_revrsd"
     elif rvrs and not hlfs_intrchg:
          srvrs = "_revrsd"
     elif not rvrs and hlfs_intrchg:
          srvrs = "_hi"

     for key in key_dats:
         iexpt += 1
         f12 = frg.split("-")
         f1 = f12[0]
         f2 = f12[1]
         allWFs   = []

         vs = "%(a)d%(g)d" % {"a" : armv_ver, "g" : gcoh_ver}
         #if key == "Jan092020_15_05_39":

         pikdir     = getResultFN("%(dir)s/v%(vs)s" % {"dir" : key, "vs" : vs})
         outdir     = getResultFN("%(dir)s/v%(vs)s/%(lb)d" % {"dir" : key, "vs" : vs, "lb" : label})

         lmBhv  = depickle("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(vs)s_%(lb)d.dmp" % {"rk" : rpsms.rpsm_eeg_as_key[key], "w" : win, "s" : slideby, "vs" : vs, "od" : pikdir, "lb" : label})
         lmXCr = depickle("%(pkd)s/xcorr_out_0_%(w)d_%(sb)d_%(f1)s_%(f2)s_v%(vs)s_%(lb)d%(rv)s.dmp" % {"pkd" : outdir, "vs" : vs, "f1" : f1, "f2" : f2, "w" : win, "sb" : slideby, "rv" : srvrs, "lb" : label})

         lmEEG         = depickle(datconf.getDataFN(datconf._RPS, "%(dsf)s_artfctrmvd/v%(av)d/%(dsf)s_gcoh_%(wn)d_%(sld)d_v%(av)d%(gv)d.dmp" % {"dsf" : key, "av" : armv_ver, "gv" : gcoh_ver, "wn" : win, "sld" : slideby}))

         fs_gcoh = lmBhv["fs"]
         fL = int(f1)
         fH = int(f2)

         irngs = _N.where((fs_gcoh > fL) & (fs_gcoh < fH))[0]
         iL    = irngs[0]
         iH    = irngs[-1]    

         real_evs = lmBhv["EIGVS"]
         nStates, rmpd_lab = find_or_retrieve_GMM_labels(datconf._RPS, key, "%(gf)s_gcoh%(evn)d_%(w)d_%(s)d_v%(av)d%(gv)d" % {"gf" : key, "w" : win, "s" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, real_evs[ev_n], iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=None, TRs=None, manual_cluster=False, ignore_stored=False)


         #fig = _plt.figure(figsize=(12, 12))
         fig = _plt.figure(figsize=(13, 6))

         ir = -1
         all_xcs_r = lmXCr["all_xcs_r"]

         mn_maxs = _N.empty(1+SHFLS)
         mn_mins = _N.empty(1+SHFLS)
         min_max_corr = _N.empty(1+SHFLS)

         pk_win = int(_N.ceil(2. / (slideby/Fs)))
         for rndmz in range(1+SHFLS):
              ir += 1
              print("rand %d"   % ir)
              max_near_midp = {}
              min_near_midp = {}

              xcs = all_xcs_r[:, :, rndmz]
                   #xcs = lmXCr["all_xcs_r%d" % ir]
              #xcs = lmXCr["all_xcs_r"] if rndmz else lmXCr["all_xcs"]
              all_shps = _N.array(lmXCr["all_shps"])
              all_shps = all_shps/_N.sum(all_shps)

              #add_this = []
              ipl = 0

              impa = -1
              for ixc in range(len(xcs)):
                   for icol in range(6):
                        if not _N.isnan(xcs[ixc][icol][0]):
                             fxcs = _N.convolve(xcs[ixc][icol], gk, mode="same")
                             fxcs -= _N.mean(fxcs)
                             srtd = _N.sort(fxcs)  #  to find highest point of xc

                             ipl += 1
                             #add_this.append(_N.abs(fxcs) * all_shps[ixc, icol])

                             dy = _N.diff(fxcs)
                             dy_a1 = dy[0:-1]
                             dy_a2 = dy[1:]

                             dy1 = dy[midp-pk_win:midp+pk_win]
                             dy2 = dy[midp-(pk_win-1):midp+pk_win+1]
                             minimas = _N.where((dy1 < 0) & (dy2 >= 0))[0]
                             maximas = _N.where((dy1 > 0) & (dy2 <= 0))[0]
                             all_minimas = _N.where((dy_a1 < 0) & (dy_a2 >= 0))[0]
                             all_maximas = _N.where((dy_a1 > 0) & (dy_a2 <= 0))[0]

                             #hiXC = srtd[int((2*midp+1)*0.85)]
                             l_max = len(all_maximas)
                             l_min = len(all_minimas)
                             if l_max > 2:  #  
                                  srtdMxm = _N.sort(fxcs[all_maximas])
                                  hiXC    = srtdMxm[l_max-2]-0.00001
                             else:
                                  hiXC = srtd[int((2*midp+1)*0.85)]
                             if l_min > 2:
                                  #srtdMxm = _N.sort(fxcs[all_maximas])
                                  srtdMin = _N.sort(fxcs[all_minimas])
                                  loXC    = srtdMin[-2]+0.00001
                             else:
                                  loXC = srtd[int((2*midp+1)*0.15)]

                             #loXC = srtd[int((2*midp+1)*0.15)]

                             if (len(minimas) > 0):
                                  for i in range(len(minimas)):
                                       im  = minimas[i] + midp-pk_win

                                       if fxcs[im] < loXC:
                                            impa += 1
                                            #min_near_midp.append(xcs[ixc][icol])
                                            min_near_midp["%(w)d,%(p)d" % {"p" : ixc, "w" : icol}] = xcs[ixc][icol]
                                            #print("min %(impa)d  %(v).4f" % {"impa" : impa, "v" : fxcs[im]})
                             if (len(maximas) > 0):            
                                  for i in range(len(maximas)):   
                                       im  = maximas[i] + midp-pk_win
                                       if fxcs[im] > hiXC:
                                            impa += 1
                                            max_near_midp["%(w)d,%(p)d" % {"p" : ixc, "w" : icol}] = xcs[ixc][icol]
                                            #max_near_midp.append(xcs[ixc][icol])
                                            #print("max %(impa)d  %(v).4f" % {"impa" : impa, "v" : fxcs[im]})

                             # if len(maxmin) == 1:
                             #      print("%(xc)d %(ic)d" % {"xc" : ixc, "ic" : icol})

              """
              mn_xc = _N.mean(_N.array(add_this), axis=0)
              #fig.add_subplot(8, 7, 1)
              fig.add_subplot(1, 1, 1)
              _plt.title("AVERAGE")
              _plt.plot(time_lags, mn_xc, lw=2, color="black")
              _plt.axvline(x=0, lw=1, color="red", ls=":")
              _plt.xticks([-30, -20, -10, 0, 10, 20, 30])
              _plt.grid()
              min_xc = _N.min(mn_xc)
              max_xc = _N.max(mn_xc)
              AMP    = max_xc - min_xc
              _plt.ylim(min_xc - 0.1*AMP, max_xc + 0.1*AMP)
              _plt.suptitle(key)
              fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, wspace=0.2, hspace=0.3)
              _plt.savefig("%(pd)s/cmp_xcorrs_%(ky)s_%(fr)s" % {"pd" : outdir, "fr" : frg, "ky" : key})
              """
              maxs = _N.array(list(max_near_midp.values()))
              mins = _N.array(list(min_near_midp.values()))

              if (maxs.shape[0] > 0) and (mins.shape[0] > 0):
                   mn_max = _N.mean(maxs, axis=0)
                   mn_min = _N.mean(mins, axis=0)

                   wd = pk_win*3   #  for correlation
                   mn_min_outer = _N.ones(2*midp+1 - 2*wd) * -1
                   mn_max_outer = _N.ones(2*midp+1 - 2*wd) * -1

                   mn_min_outer[0:midp-wd] = mn_min[0:midp-wd]
                   mn_min_outer[midp-wd:] = mn_min[midp+wd:]
                   mn_max_outer[0:midp-wd] = mn_max[0:midp-wd]
                   mn_max_outer[midp-wd:] = mn_max[midp+wd:]

                   pc, pv = _ss.pearsonr(mn_min_outer, mn_max_outer)

                   maxy = _N.max(mn_max)
                   miny = _N.min(mn_min)
                   mn_maxs[rndmz] = maxy
                   mn_mins[rndmz] = miny
                   min_max_corr[rndmz] = pc
              else:
                   print("0 FOUND for min_max, rand %d" % ir)
                   mn_maxs[rndmz] = 0
                   mn_mins[rndmz] = 0
                   min_max_corr[rndmz] = 0
                   pc, pv = 0, 1
                   maxy = 0
                   miny = 0


              if ir < 8:
                   ax = fig.add_subplot(2, 8, ir+1)
                   if rndmz > 0:
                        ax.set_facecolor("#EEEEEE")
                   _plt.title("CC=%.3f" % pc)

                   if rndmz == 0:
                        maxy0 = maxy
                        miny0 = miny
                   for nx in range(maxs.shape[0]):
                        _plt.plot(time_lags, maxs[nx], color="orange")
                   _plt.plot(time_lags, mn_min, color="black", lw=3)
                   _plt.plot(time_lags, mn_max, color="blue", lw=3)
                   _plt.axvline(x=0, ls="--", color="orange")    
                   _plt.axhline(y=0, ls="--", color="orange")
                   _plt.axhline(y=maxy0, ls="--", color="grey")
                   _plt.axhline(y=miny0, ls="--", color="grey")

                   if rndmz > 0:
                        _plt.axhline(y=maxy, ls=":", color="grey")
                        _plt.axhline(y=miny, ls=":", color="grey")
                   _plt.xticks(xticksD)
                   _plt.grid()
                   _plt.ylim(-0.2, 0.2)
                   _plt.yticks([-0.2, 0, 0.2])
                   ########################
                   ax = fig.add_subplot(2, 8, ir+9)
                   if rndmz > 0:
                        ax.set_facecolor("#DDDDDD")
                   for nx in range(mins.shape[0]):
                        _plt.plot(time_lags, mins[nx], color="orange")
                   _plt.plot(time_lags, mn_min, color="black", lw=3)
                   _plt.plot(time_lags, mn_max, color="blue", lw=3)
                   _plt.axvline(x=0, ls="--", color="orange")    
                   _plt.axhline(y=0, ls="--", color="orange")
                   _plt.axhline(y=maxy0, ls="--", color="grey")
                   _plt.axhline(y=miny0, ls="--", color="grey")

                   if rndmz > 0:
                        _plt.axhline(y=maxy, ls=":", color="grey")
                        _plt.axhline(y=miny, ls=":", color="grey")


                   _plt.xticks(xticksD)
                   _plt.ylim(-0.2, 0.2)
                   _plt.yticks([-0.2, 0, 0.2])
                   _plt.grid()

         _plt.suptitle("%(ky)s %(rv)s" % {"ky" : key, "rv" : srvrs})
         fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.9)
         _plt.savefig("%(pd)s/cmp_xcorrs_show_%(ky)s_%(fr)s" % {"pd" : outdir, "fr" : frg, "ky" : key})

         max_keys = max_near_midp.keys()
         min_keys = min_near_midp.keys()

         maxdir = "%(pd)s/skullplots_max_%(w)d_%(sb)d_%(f1)s_%(f2)s_v%(vs)s" % {"pd" : outdir, "k" : key, "vs" : vs, "f1" : f1, "f2" : f2, "w" : win, "sb" : slideby}
         mindir = "%(pd)s/skullplots_min_%(w)d_%(sb)d_%(f1)s_%(f2)s_v%(vs)s" % {"pd" : outdir, "k" : key, "vs" : vs, "f1" : f1, "f2" : f2, "w" : win, "sb" : slideby}
         if not os.access(maxdir, os.F_OK):
              os.mkdir(maxdir)
         if not os.access(mindir, os.F_OK):
              os.mkdir(mindir)

         print("max___")
         real_evs = lmBhv["EIGVS"]
         chs_picks = lmEEG["chs_picks"]
         #all_vecs = _N.empty((len(chs_picks), nStates))
         all_vecs = _N.zeros((6, len(chs_picks), nStates))

         for epc in range(6):
              all_vecs[:, :] = 0
              patIDs = patIDs_for_epoch(max_keys, epc)         
              for pI in range(len(patIDs)):
                   these_pats = _N.where(rmpd_lab == patIDs[pI])[0]              

                   all_vecs[epc, :, patIDs[pI]] = _N.mean(_N.mean(real_evs[0, these_pats, iL:iH], axis=1), axis=0)


              if len(patIDs) > 0:
                   _sp.do_skull_plot_all_EVs(all_vecs[epc], _N.arange(nStates), arr_ch_names[chs_picks], "%(od)s/%(dat)s_%(w)d_%(sl)d_skull_coh_pattern_%(epc)d_%(evn)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d%(rs)s" % {"1" : fL, "2" : fH, "dat" : key, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : maxdir, "evn" : ev_n, "epc" : epc, "lb" : label, "rs" : srvrs}, key, fL, fH, extra_suptitle=("max epc %d" % epc), close=True)
         #_sp.do_skull_plot_all_EVs_timeseries(all_vecs, _N.arange(nStates), arr_ch_names[chs_picks], "%(od)s/%(dat)s_%(w)d_%(sl)d_skull_coh_pattern_all_%(evn)d_%(1)d_%(2)d_v%(av)d%(gv)d" % {"1" : fL, "2" : fH, "dat" : key, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : maxdir, "evn" : ev_n, "epc" : epc}, key, fL, fH, extra_suptitle=("max epc %d" % epc))
         print("min___")


         for epc in range(6):
              all_vecs[:, :] = 0
              patIDs = patIDs_for_epoch(min_keys, epc)         
              print("epc %d" % epc)
              print(patIDs)


              for pI in range(len(patIDs)):
                   these_pats = _N.where(rmpd_lab == patIDs[pI])[0]
                   all_vecs[epc, :, patIDs[pI]] = _N.mean(_N.mean(real_evs[0, these_pats, iL:iH], axis=1), axis=0)


              if len(patIDs) > 0:
                    _sp.do_skull_plot_all_EVs(all_vecs[epc], _N.arange(nStates), arr_ch_names[chs_picks], "%(od)s/%(dat)s_%(w)d_%(sl)d_skull_coh_pattern_%(epc)d_%(evn)d_%(1)d_%(2)d_v%(av)d%(gv)d_%(lb)d%(rs)s" % {"1" : fL, "2" : fH, "dat" : key, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : mindir, "evn" : ev_n, "epc" : epc, "rs" : srvrs, "lb" : label}, key, fL, fH, extra_suptitle=("min epc %d" % epc), close=True)


         fig  = _plt.figure(figsize=(6, 6))
         _plt.suptitle("%(ky)s %(rv)s" % {"ky" : key, "rv" : srvrs})
         fig.add_subplot(2, 2, 1)
         _plt.title("maxes %(o)d/%(s)d" % {"o" : len(_N.where(mn_maxs[1:] > mn_maxs[0])[0]), "s" : SHFLS})
         _plt.hist(mn_maxs[1:], bins=_N.linspace(0, 0.2, 51), color="black")
         _plt.axvline(x=mn_maxs[0], color="red", ls=":", lw=2)
         fig.add_subplot(2, 2, 2)
         _plt.title("mins %(o)d/%(s)d" % {"o" : len(_N.where(mn_mins[1:] < mn_mins[0])[0]), "s" : SHFLS})
         _plt.hist(mn_mins[1:], bins=_N.linspace(-0.2, 0, 51), color="black")
         _plt.axvline(x=mn_mins[0], color="red", ls=":", lw=2)
         fig.add_subplot(2, 2, 3)
         dm = mn_maxs - mn_mins
         _plt.title("max-min %(o)d/%(s)d" % {"o" : len(_N.where(dm[1:] > dm[0])[0]), "s" : SHFLS})
         _plt.hist(dm[1:], bins=_N.linspace(-0.2, 0.2, 51), color="black")
         _plt.axvline(x=dm[0], color="red", ls=":", lw=2)
         fig.add_subplot(2, 2, 4)
         _plt.title("corr min,max %(o)d/%(s)d" % {"o" : len(_N.where(min_max_corr[1:] > min_max_corr[0])[0]), "s" : SHFLS})
         _plt.hist(min_max_corr[1:], bins=_N.linspace(-1, 1, 51), color="black")
         _plt.axvline(x=min_max_corr[0], color="red", ls=":", lw=2)

         _plt.savefig("%(pd)s/cmp_xcorrs_%(ky)s_%(fr)s_%(lb)d%(rs)s" % {"pd" : outdir, "fr" : frg, "ky" : key, "rs" : srvrs, "lb" : label})
