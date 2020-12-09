import numpy as _N
import matplotlib.pyplot as _plt
import rpsms
import GCoh.eeg_util as _eu
from filter import gauKer

from AIiRPS.utils.dir_util import getResultFN

lbsz=18
tksz=16
def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

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
#key_dats = ["Aug182020_16_25_28", "Aug182020_16_44_18"]
#key_dats = ["Jan092020_15_05_39"]
#key_dats = ["Aug122020_12_52_44"]
key_dats = ["Jan082020_17_03_48"]
#key_dats  = ["Aug122020_13_30_23"]
#key_dats = ["Jan082020_17_03_48", "Jan092020_15_05_39", "Aug122020_12_52_44", "Aug122020_13_30_23", "Aug182020_15_45_27"]  # Ken
#key_dats = ["Aug182020_16_25_28", "Aug182020_16_44_18"]  # Ali 

win      = 256
slideby  = 64

frg      = "35-47"
#frg      = "18-25"
#frg      = "25-30"
lags_sec=30
slideby_sec = slideby/300
lags = int(lags_sec / slideby_sec)+1  #  (slideby/300)*lags
xticksD = [-20, -10, 0, 10, 20]
time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)

iexpt = -1

#time_lags = _N.linspace(-(slideby/300)*lags, lags*(slideby/300), 2*lags+1)

gk = gauKer(1)

pf_x = _N.arange(-15, 16)
all_x= _N.arange(-141, 142)
for key in key_dats:
    iexpt += 1
    f12 = frg.split("-")
    f1 = f12[0]
    f2 = f12[1]
    allWFs   = []
    vs = "13"
    if key == "Jan092020_15_05_39":
        vs = "43"
    pikdir     = getResultFN("%(dir)s/v%(vs)s" % {"dir" : key, "vs" : vs})

    lmXCr = depickle(getResultFN("%(k)s/v%(vs)s/xcorr_out_0_256_64_%(f1)s_%(f2)s_v%(vs)s.dmp" % {"k" : key, "vs" : vs, "f1" : f1, "f2" : f2}))
    lmACr = depickle(getResultFN("%(k)s/v%(vs)s/acorr_out_0_256_64_%(f1)s_%(f2)s_v%(vs)s.dmp" % {"k" : key, "vs" : vs, "f1" : f1, "f2" : f2}))
    lmBhv  = depickle("%(od)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(vs)s.dmp" % {"rk" : rpsms.rpsm_eeg_as_key[key], "w" : win, "s" : slideby, "vs" : vs, "od" : pikdir})

    #fig = _plt.figure(figsize=(12, 12))
    fig = _plt.figure(figsize=(4, 3))
    
    xcs = lmXCr["all_xcs_r"]
    all_shps = _N.array(lmXCr["all_shps"])
    all_shps = all_shps/_N.sum(all_shps)
    midp = 141
    add_this = []
    ipl = 0

    for ixc in range(len(xcs)):
         for icol in range(5):
              if not _N.isnan(xcs[ixc][icol][0]):
                   xcs[ixc][icol] -= _N.mean(xcs[ixc][icol])
                   ipl += 1
                   #fig.add_subplot(8, 7, ipl+1)
                   # coeffs = _N.polyfit(pf_x, xcs[ixc][icol][midp-15:midp+16], 2)
                   # #fxc  = _N.convolve(xcs[ixc][icol][midp-15:midp+16], 2))
                   # if coeffs[0] > 0:  #  pos coeff means min near 0
                   #      add_this.append(_N.abs(xcs[ixc][icol]))# * all_shps[ixc, icol])
                   # else:
                   add_this.append(_N.abs(xcs[ixc][icol]) * all_shps[ixc, icol])
                   # _plt.plot(time_lags, _N.abs(xcs[ixc][icol]), color="grey", lw=1)
                   # _plt.xticks([-30, -15, 0, 15, 30])
                   # _plt.axvline(x=0, lw=2, color="red")


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
    _plt.savefig("%(pd)s/cmp_xcorrs_%(ky)s_%(fr)s" % {"pd" : pikdir, "fr" : frg, "ky" : key})
