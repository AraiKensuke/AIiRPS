#!/usr/bin/python

from sklearn import linear_model
import sklearn.linear_model as _skl
import numpy as _N
import AIiRPS.utils.read_taisen as _rt
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
import pickle
import mne.time_frequency as mtf
from filter import gauKer
import GCoh.eeg_util as _eu
import AIiRPS.rpsms as rpsms

import AIiRPS.constants as _cnst
from AIiRPS.utils.dir_util import getResultFN
import GCoh.datconfig as datconf
import AIiRPS.models.CRutils as _crut
import AIiRPS.models.empirical as _emp

import GCoh.eeg_util as _eu
import matplotlib.ticker as ticker

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

def wnd_mean(x, y, winsz=4, mode=1, wins=8):
    if mode == 1:
        ix = x.argsort()

        xwm = _N.empty(x.shape[0] - (winsz-1))  #  5 - (3-1)
        ywm = _N.empty(x.shape[0] - (winsz-1))  #  5 - (3-1)
        
        for i in range(x.shape[0] - (winsz-1)):
            xwm[i] = _N.mean(x[ix[i:i+winsz]])
            ywm[i] = _N.mean(y[ix[i:i+winsz]])
        return xwm, ywm
    else:
        x_min = _N.min(x) 
        x_max = _N.max(x)
        x_min -= 0.1*(x_max - x_min)/wins
        x_max += 0.1*(x_max - x_min)/wins

        dx    = (x_max - x_min) / wins

        xwm = _N.empty(wins)  #  5 - (3-1)
        ywm = _N.empty(wins)  #  5 - (3-1)
        
        xl = x_min
        for iw in range(wins):
            ths = _N.where((x >= xl) & (x < xl + dx))[0]
            xwm[iw] = _N.mean(x[ths])
            ywm[iw] = _N.mean(y[ths])
            xl += dx
            
        return xwm, ywm
        
def interiorCC(x, y):
    ix = x.argsort()
    iy = y.argsort()    

    ths = _N.array([ix[0], ix[-1], iy[0], iy[-1]])
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    return _ss.pearsonr(x[interiorPts], y[interiorPts])

def rm_outliersCC_orig(x, y):
    ix = x.argsort()
    iy = y.argsort()    
    L = len(x)
    x_std = _N.std(x)
    y_std = _N.std(y)
    rmv   = []
    i = 0
    while x[ix[i+1]] - x[ix[i]] > 0.5*x_std:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > 0.5*x_std:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > 0.5*y_std:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > 0.5*y_std:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    #print("%(ths)d" % {"ths" : len(ths)})
    return _ss.pearsonr(x[interiorPts], y[interiorPts])

def rm_outliersCC_neighbors(x, y):
    ix = x.argsort()
    iy = y.argsort()
    dsx = _N.mean(_N.diff(_N.sort(x)))
    dsy = _N.mean(_N.diff(_N.sort(y)))

    L = len(x)
    x_std = _N.std(x)
    y_std = _N.std(y)
    rmv   = []
    i = 0
    while x[ix[i+1]] - x[ix[i]] > 4*dsx:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > 4*dsx:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > 4*dsy:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > 4*dsy:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    #print("%(ths)d" % {"ths" : len(ths)})
    return _ss.pearsonr(x[interiorPts], y[interiorPts])

def rm_outliersCC(x, y):
    ix = x.argsort()
    iy = y.argsort()    
    L = len(x)
    x_std = _N.std(x)
    y_std = _N.std(y)
    rmv   = []
    i = 0
    rmvd = 0
    while x[ix[i+1]] - x[ix[i]] > x_std:
        #rmvd += 1
        #rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > x_std:
        #rmvd += 1        
        #rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > y_std:
        #rmvd += 1        
        #rmv.append(iy[i])
        y[iy[i]] = y[iy[i+1]] - y_std*0.2
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > y_std:
        #rmvd += 1        
        #rmv.append(iy[L-1-i])
        y[iy[L-1-i]] = y[L-1-i-1] + y_std*0.2
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    #print("%(ths)d" % {"ths" : len(ths)})
    return rmvd, x[interiorPts], y[interiorPts]#, _ss.pearsonr(x[interiorPts], y[interiorPts])
 


lm = depickle("predictAQ28dat/AQ28_vs_RPS.dmp")


features_cab = lm["features_cab"]
features_stat = lm["features_stat"]
cmp_againsts = features_cab + features_stat

for ca in cmp_againsts:
    exec("%(ca)s = lm[\"%(ca)s\"]" % {"ca" : ca})

look_at_AQ    = True
marginalCRs   = eval("lm[\"marginalCRs\"]")
AQ28scrs      = eval("lm[\"AQ28scrs\"]")
soc_skils     = eval("lm[\"soc_skils\"]")
imag          = eval("lm[\"imag\"]")
rout          = eval("lm[\"rout\"]")
switch        = eval("lm[\"switch\"]")
fact_pat      = eval("lm[\"fact_pat\"]")
all_prob_mvsA = eval("lm[\"all_prob_mvsA\"]")
label         = eval("lm[\"label\"]")
signal_5_95   = eval("lm[\"signal_5_95\"]")
t0            = eval("lm[\"t0\"]")
t1            = eval("lm[\"t1\"]")
win           = eval("lm[\"win\"]")
all_maxs      = eval("lm[\"all_maxs\"]")
partIDs       = eval("lm[\"partIDs\"]")
imax_imin_pfrm36  = lm["imax_imin_pfrm36"]
imax_imin_pfrm69  = lm["imax_imin_pfrm69"]
imax_imin_pfrm912 = lm["imax_imin_pfrm912"]
data = lm["data"]
nDAT = len(partIDs)
show_mn = True
mn_mode = 2
allInds = _N.arange(nDAT)

winsz=15



entropyUD = _N.exp((entropyD + entropyU) / (_N.mean(entropyD + entropyU)))
#entropyUD2 = _N.exp(entropyUD2 / _N.mean(entropyUD2))
#entropyS2 = _N.exp(entropyS2 / _N.mean(entropyS2))

mmm  = 1.
entropyUDS = _N.exp((entropyD + entropyU + entropyS) / (_N.mean(entropyD + entropyU + entropyS)))
entropyD = _N.exp(entropyD / _N.mean(entropyD))
entropyS = _N.exp(entropyS / _N.mean(entropyS))
entropyU = _N.exp(entropyU / _N.mean(entropyU))
entropyWL = _N.exp((entropyW+entropyL) / _N.mean(entropyW+entropyL))
entropyW = _N.exp(entropyW / _N.mean(entropyW*mmm))
entropyT = _N.exp(entropyT / _N.mean(entropyT*mmm))
entropyL = _N.exp(entropyL / _N.mean(entropyL*mmm))
entropyT2 = _N.exp(entropyT2 / _N.mean(entropyT2*0.2)) / 10
entropyW2 = _N.exp(entropyW2 / _N.mean(entropyW2*0.3)) / 10
entropyL2 = _N.exp(entropyL2 / _N.mean(entropyL2*0.3)) / 10

ths = _N.where((AQ28scrs > 35) & (rout > 4))[0]
#ths = _N.where((AQ28scrs > 35))[0]

if look_at_AQ:
    pcpvs = {}
    #hlf1    = _N.random.choice(allInds, nDAThlf, replace=False)
    #hlf2    = _N.setdiff1d(allInds, hlf1)
    for cmp_against in cmp_againsts:
        pcs = _N.empty(6)
        pvs = _N.empty(6)
        pcpvs[cmp_against] = [pcs, pvs]
        #if cmp_against == "entropyUD":
        #    cmp_vs = entropyU + entropyD
        if cmp_against == "sum_sd":
            cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
            #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
            #cmp_vs = _N.log(sum_sd[:, 2, 0] + sum_sd[:, 2, 2])
            #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) / _N.sum(sum_sd[:, :, 2], axis=1))   #  INTERESTING
            #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 1], axis=1))
            #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 1], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
            #cmp_vs = _N.log(_N.sum(sum_sd[:, 0], axis=1) + _N.sum(sum_sd[:, 2], axis=1))
        elif cmp_against == "sum_cv":
            cmp_vs = _N.log(_N.sum(sum_cv[:, :, 2], axis=1))# + _N.sum(sum_cv[:, :, 2], axis=1)
        elif cmp_against == "entropyDSU":
            cmp_vs = _N.sum(entropyDSU, axis=1)
        elif cmp_against == "entropyTL":
            cmp_vs = (entropyT + entropyL)
        else:
            cmp_vs = eval(cmp_against)

        _AQ28scrs   = AQ28scrs[ths]
        _cmp_vs     = cmp_vs[ths]
        _soc_skils  = soc_skils[ths]
        _rout       = rout[ths]
        _imag       = imag[ths]
        _switch     = switch[ths]
        _fact_pat   = fact_pat[ths]
        dat = _N.empty((len(ths), 2))

        fig = _plt.figure(figsize=(14, 3))
        fig.add_subplot(1, 6, 1)
        _plt.xlabel("soc_skills", fontsize=14)
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_soc_skils, _cmp_vs, winsz=winsz, mode=mn_mode)
        if show_mn:
            _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        pc, pv = _ss.pearsonr(_soc_skils, _cmp_vs)
        pcs[0] = pc
        pvs[0] = pv
        rmvd, xI, yI = rm_outliersCC(_soc_skils, _cmp_vs)
        _plt.scatter(xI, yI, marker=".", s=50, color="#111111")                
        slope, intercept, pcI, pvI, std_err = _ss.linregress(xI, yI)
        _plt.plot(xI, slope*xI + intercept, color="red", lw=2)        
        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f (%(rm)d)" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI, "rm" : rmvd})
        _plt.grid(ls=":")
        dat[:, 0] = _soc_skils
        dat[:, 1] = _cmp_vs
        #dmp_dat[cmp_against] = _N.array(dat[:, 1])
        #################################
        fig.add_subplot(1, 6, 2)
        _plt.xlabel("imag", fontsize=14)
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_imag, _cmp_vs, winsz=winsz, mode=mn_mode)
        if show_mn:
            _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        pc, pv = _ss.pearsonr(_imag, _cmp_vs)
        pcs[1] = pc
        pvs[1] = pv        
        rmvd, xI, yI = rm_outliersCC(_imag, _cmp_vs)
        _plt.scatter(xI, yI, marker=".", s=50, color="#111111")                
        slope, intercept, pcI, pvI, std_err = _ss.linregress(xI, yI)
        _plt.plot(xI, slope*xI + intercept, color="red", lw=2)        
        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f (%(rm)d)" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI, "rm" : rmvd})
        _plt.grid(ls=":")
        dat[:, 0] = _imag
        dat[:, 1] = _cmp_vs
        #dmp_dat["imag_%s" % cmp_against] =  _N.array(dat)
        #################################        
        fig.add_subplot(1, 6, 3)
        _plt.xlabel("routing", fontsize=14)
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_rout, _cmp_vs, winsz=winsz, mode=mn_mode)
        if show_mn:
            _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        pc, pv = _ss.pearsonr(_rout, _cmp_vs)
        pcs[2] = pc
        pvs[2] = pv
        #pcI, pvI = rm_outliersCC(_rout, _cmp_vs)
        rmvd, xI, yI = rm_outliersCC(_rout, _cmp_vs)
        _plt.scatter(xI, yI, marker=".", s=50, color="#111111")                
        slope, intercept, pcI, pvI, std_err = _ss.linregress(xI, yI)
        _plt.plot(xI, slope*xI + intercept, color="red", lw=2)
        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f (%(rm)d)" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI, "rm" : rmvd})
        _plt.grid(ls=":")
        dat[:, 0] = _rout
        dat[:, 1] = _cmp_vs
        #dmp_dat["rout_%s" % cmp_against] = _N.array(dat)
        
        #################################
        fig.add_subplot(1, 6, 4)
        _plt.xlabel("switch", fontsize=14)
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_switch, _cmp_vs, winsz=winsz, mode=mn_mode)
        if show_mn:
            _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        pc, pv = _ss.pearsonr(_switch, _cmp_vs)
        pcs[3] = pc
        pvs[3] = pv
        rmvd, xI, yI = rm_outliersCC(_switch, _cmp_vs)
        _plt.scatter(xI, yI, marker=".", s=50, color="#111111")                
        slope, intercept, pcI, pvI, std_err = _ss.linregress(xI, yI)
        _plt.plot(xI, slope*xI + intercept, color="red", lw=2)        
        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f (%(rm)d)" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI, "rm" : rmvd})             
        _plt.grid(ls=":")
        dat[:, 0] = _switch
        dat[:, 1] = _cmp_vs
        #dmp_dat["switch_%s" % cmp_against] =  _N.array(dat)
        
        #################################        
        fig.add_subplot(1, 6, 5)
        _plt.xlabel("factor numb pats", fontsize=14)
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_fact_pat, _cmp_vs, winsz=winsz, mode=mn_mode)
        if show_mn:
            _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        pc, pv = _ss.pearsonr(_fact_pat, _cmp_vs)
        pcs[4] = pc
        pvs[4] = pv        
        #pcI, pvI = rm_outliersCC(_fact_pat, _cmp_vs)
        rmvd, xI, yI = rm_outliersCC(_fact_pat, _cmp_vs)
        _plt.scatter(xI, yI, marker=".", s=50, color="#111111")                
        slope, intercept, pcI, pvI, std_err = _ss.linregress(xI, yI)
        _plt.plot(xI, slope*xI + intercept, color="red", lw=2)        
        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f (%(rm)d)" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI, "rm" : rmvd})             
        _plt.grid(ls=":")
        dat[:, 0] = _fact_pat
        dat[:, 1] = _cmp_vs
        #dmp_dat["fact_pat_%s" % cmp_against] = _N.array(dat)
        
        #################################        
        ax = fig.add_subplot(1, 6, 6)
        _plt.xlabel("AQ28", fontsize=15, fontweight="bold")
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_AQ28scrs, _cmp_vs, winsz=winsz, mode=mn_mode)
        if show_mn:
            _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        pc, pv = _ss.pearsonr(_AQ28scrs, _cmp_vs)
        pcs[5] = pc
        pvs[5] = pv        
        rmvd, xI, yI = rm_outliersCC(_AQ28scrs, _cmp_vs)
        _plt.scatter(xI, yI, marker=".", s=50, color="#111111")        
        slope, intercept, pcI, pvI, std_err = _ss.linregress(xI, yI)
        _plt.plot(xI, slope*xI + intercept, color="red", lw=2)        
        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f (%(rm)d)" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI, "rm" : rmvd})             
        _plt.grid(ls=":")
        dat[:, 0] = _AQ28scrs
        dat[:, 1] = _cmp_vs
        #dmp_dat["AQ28_%s" % cmp_against] = _N.array(dat)

        ax.spines["left"].set_linewidth(4)
        ax.spines["right"].set_linewidth(4)
        ax.spines["top"].set_linewidth(4)
        ax.spines["bottom"].set_linewidth(4)        
        fig.subplots_adjust(hspace=0.4, left=0.06, bottom=0.17, top=0.82, right=0.98)
        _plt.suptitle("%(vs)s" % {"vs" : cmp_against})
        _plt.savefig("AQ_vs_%(vs)s_%(lb)d" % {"vs":cmp_against, "lb":label})
        _plt.close()

diffs36 = _N.diff(imax_imin_pfrm36, axis=1)
diffs69 = _N.diff(imax_imin_pfrm69, axis=1)
diffs912 = _N.diff(imax_imin_pfrm912, axis=1)
fig = _plt.figure(figsize=(9, 3))
fig.add_subplot(1, 3, 1)
_plt.hist(diffs36, bins=_N.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), density=True, color="black")
_plt.xlabel("min precedes max by")
_plt.ylim(0, 0.5)
fig.add_subplot(1, 3, 2)
_plt.hist(diffs69, bins=_N.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), density=True, color="black")
_plt.xlabel("min precedes max by")
_plt.ylim(0, 0.5)
fig.add_subplot(1, 3, 3)
_plt.hist(diffs912, bins=_N.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), density=True, color="black")
_plt.xlabel("min precedes max by")
_plt.ylim(0, 0.5)
fig.subplots_adjust(bottom=0.15)
_plt.savefig("min_precedes_max_pfrm")
_plt.close()

fig = _plt.figure()
for n in range(len(partIDs)):
    _plt.plot(signal_5_95[n, 0], color="#BCBCBC")
_plt.plot(_N.mean(signal_5_95[:, 0], axis=0), color="black")
_plt.xticks(_N.arange(t1-t0), _N.arange(t0 - win//2, t1 - win//2), fontsize=16)
_plt.yticks([-0.3, -0.15, 0, 0.15, 0.3], fontsize=16)
_plt.axvline(x=(-(t0-win//2)), lw=2, ls=":", color="red")
_plt.axhline(y=0, color="red", ls=":")
#_plt.yticks(_N.linspace(0., 1, 6))
_plt.grid(ls=":")
#_plt.ylim(0., 1)
_plt.ylim(-0.31, 0.31)
_plt.xlim(-0.1, 14.2)
_plt.xlabel("lags from rule change (# games)", fontsize=18)
_plt.ylabel("p(WIN, lag) - p(LOS, lag)", fontsize=18)
fig.subplots_adjust(left=0.18, bottom=0.18, top=0.98, right=0.98)
_plt.savefig("nrm_avg_%(fn)s_%(lb)d.png" % {"fn" : data, "lb" : label}, transparent=True)
_plt.close()

AQ28subs = ["SB:SS", "SB:IM", "SB:RT", "SB:SW", "NumPat", "AQ28"]

fig = _plt.figure(figsize=(13, 2))
ic  = 0
for cat in [soc_skils, imag, rout, switch, fact_pat]:
    ic += 1
    ax = fig.add_subplot(1, 6, ic)
    bns = _N.linspace(_N.min(cat)-0.5, _N.max(cat)+0.5, int(_N.max(cat) - _N.min(cat)+2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _plt.locator_params(axis='x', nbins=5)
    _plt.locator_params(axis='y', nbins=5)        
    _plt.hist(cat, bins=bns, color="black")
    # if cat is soc_skils:
    #     _plt.xticks(_N.arange(6, 25, 6))
    # elif cat is imag:
    #     _plt.xticks(_N.arange(9, 34, 6))
    # elif (cat is rout) or (cat is switch):
    #     _plt.xticks(_N.arange(4, 17, 4))
    # elif (cat is fact_pat):
    #     _plt.xticks(_N.arange(4, 21, 5))
        
    _plt.xlabel(AQ28subs[ic-1], fontsize=14)
    _plt.yticks(fontsize=12)
    _plt.xticks(fontsize=12)        
        
_plt.hist(cat, bins=bns, color="black")
ax = fig.add_subplot(1, 6, 6)
cat = AQ28scrs
bns = _N.arange(_N.min(cat)-0.5, _N.max(cat)+0.5, 1)
_plt.hist(cat, bins=bns, color="black")
ax.spines["left"].set_linewidth(4)
ax.spines["right"].set_linewidth(4)
ax.spines["top"].set_linewidth(4)
ax.spines["bottom"].set_linewidth(4)
#  https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
_plt.locator_params(axis='y', nbins=5)
_plt.locator_params(axis='x', nbins=5)
_plt.yticks(fontsize=12)
_plt.xticks(fontsize=12)

#_plt.xticks(_N.arange(30, 96, 10))
_plt.xlabel(AQ28subs[5], fontsize=14, fontweight="bold")
ax = fig.subplots_adjust(bottom=0.24, top=0.82, right=0.98, hspace=0.18)
_plt.savefig("TMB2_AQ_hist", transparent=True)



###  The idea is that when people make a decision whether to stay, up or down,
###  the decision to stay or switch might be a lot more calculated than
###  the decision to UP or DN.
###  Including UP and DN separately in entropy might introduce variability
###  that isn't reflective of cognitive thought, but noise

# fig = _plt.figure(figsize=(13, 3))
# if1 = -1
# for feat1 in [entropyS, entropyD, entropyU, entropyW2, entropyT2, entropyL2]:
#     if1 += 1
#     fig.add_subplot(1, 6, if1+1)
#     pc, pv = _ss.pearsonr(feat1, isis_cv)
#     _plt.title("%(pc).2f  %(pv).3f" % {"pc" : pc, "pv" : pv}) 
#     _plt.scatter(feat1, isis_cv, color="black", s=3)


# fig = _plt.figure(figsize=(13, 3))    
# if1 = -1
# for feat1 in [entropyS, entropyD, entropyU, entropyW2, entropyT2, entropyL2]:
#     if1 += 1
#     fig.add_subplot(1, 6, if1+1)
#     pc, pv = _ss.pearsonr(feat1, isis_corr)
#     _plt.title("%(pc).2f  %(pv).3f" % {"pc" : pc, "pv" : pv}) 
#     _plt.scatter(feat1, isis_corr, color="black", s=3)

# fig = _plt.figure(figsize=(10, 10))
# i   = -1
# for i in range(3):
#     for j in range(3):
#         if2 += 1
#         fig.add_subplot(3, 3, 3*i + j + 1)
#         pc, pv = _ss.pearsonr(AQ28scrs[ths], marginalCRs[ths, i, j])
#         _plt.title("%(pc).2f  %(pv).3f" % {"pc" : pc, "pv" : pv}) 
#         _plt.scatter(AQ28scrs[ths], marginalCRs[ths, i, j], color="black", s=3)

# fig = _plt.figure(figsize=(10, 10))
# aq28sig = fact_pat
# for i in range(3):
#     for j in range(3):
#         if2 += 1
#         fig.add_subplot(3, 3, 3*i + j + 1)
#         pc, pv = _ss.pearsonr(aq28sig[ths], marginalCRs[ths, i, j])
#         _plt.title("%(pc).2f  %(pv).3f" % {"pc" : pc, "pv" : pv}) 
#         _plt.scatter(aq28sig[ths], marginalCRs[ths, i, j], color="black", s=3)

#_ss.pearsonr(isis_corr, pfrm_change69)
#_plt.scatter(_N.sum(sum_sd[:, 1], axis=1), isis_corr)
#_plt.scatter(_N.sum(sum_sd[:, :, 2], axis=1), isis_corr)


# for i1 in range(3):
#     for i2 in range(i1+1, 3):
#         fig = _plt.figure(figsize=(10, 10))
#         _plt.suptitle("%(i1)d %(i2)d" % {"i1" : i1, "i2" : i2})
#         for j1 in range(3):
#             for j2 in range(3):
#                 fig.add_subplot(3, 3, 3*j1 + j2 + 1)
#                 pc, pv = _ss.pearsonr(marginalCRs[:, i1, j1], marginalCRs[:, i2, j2])
#                 _plt.title("%(pc).2f  %(pv).3f" % {"pc" : pc, "pv" : pv}) 
#                 _plt.scatter(marginalCRs[:, i1, j1], marginalCRs[:, i2, j2], color="black", s=3)

"""                
a_all_prob_pcs = _N.array(all_prob_pcs)
a_all_prob_pcs = a_all_prob_pcs.reshape(a_all_prob_pcs.shape[0], 9)

nstates, labs = _eu.find_GMM_labels(a_all_prob_pcs, TRs=[1, 2, 4, 6, 8, 10, 12, 14, 16])

nStrats = _N.empty(len(partIDs), dtype=_N.int)
for i in range(len(partIDs)):
    g0= strtend[i]
    G = strtend[i+1] - g0
    G2= g0 + G//2
    unqs = _N.unique(labs[g0:g0+G])
    nStrats[i] = len(unqs)




    f1 = labs[g0:g0+G2]
    f2 = labs[g0+G2:g0+G]
    u1 = _N.unique(f1)
    u2 = _N.unique(f2)    
    unb = 0
    for iu in u1:
        n1 = len(_N.where(iu == f1)[0])
        n2 = len(_N.where(iu == f2)[0])
        if (n1 <= 1) and (n2 >= 3):
           unb += 1 
    for iu in u2:
        n1 = len(_N.where(iu == f1)[0])
        n2 = len(_N.where(iu == f2)[0])
        if (n2 <= 1) and (n1 >= 3):
           unb += 1
    print(unb)
"""


# fig = _plt.figure(figsize=(10, 10))
# ii = 0
# for i in range(3):
#     for j in range(2):
#         ii+= 1
#         fig.add_subplot(3, 2, ii)
#         _plt.scatter(cond_distinguished[:, j], stay_amps[:, i])

#  entropyL - entropyW  (SW)
#  entropyL - entropyS  (SW)
#  entropyL + entropyW  (fact_pat)

sL = _N.std(entropyL)
eL = entropyL/sL
mL = _N.mean(eL)
entropyLs = eL - mL
sW = _N.std(entropyW)
eW = entropyW/sW
mW = _N.mean(eW)
entropyWs = eW - mW
sT = _N.std(entropyT)
eT = entropyT/sT
mT = _N.mean(eT)
entropyTs = eT - mT
sS = _N.std(entropyS)
eS = entropyS/sS
mS = _N.mean(eS)
entropySs = eS - mS
sD = _N.std(entropyD)
eD = entropyD/sD
mD = _N.mean(eD)
entropyDs = eD - mD

cmbs = ["entropyL - entropyW", "entropyL + entropyW",
        "entropyL - entropyT", "entropyL + entropyT",
        "entropyW - entropyT", "entropyW + entropyT",        
        "entropyT - entropyS", "entropyT + entropyS", 
        "entropyT - entropyD", "entropyT + entropyD", 
        "entropyL - entropyS", "entropyL + entropyS", 
        "entropyL - entropyD", "entropyL + entropyD", 
        "entropyW - entropyS", "entropyW + entropyS", 
        "entropyW - entropyD", "entropyW + entropyD"]
cmbs = ["entropyLs - entropyWs", "entropyLs + entropyWs",
        "entropyLs - entropyTs", "entropyLs + entropyTs",
        "entropyWs - entropyTs", "entropyWs + entropyTs",        
        "entropyTs - entropySs", "entropyTs + entropySs", 
        "entropyTs - entropyDs", "entropyTs + entropyDs", 
        "entropyLs - entropySs", "entropyLs + entropySs", 
        "entropyLs - entropyDs", "entropyLs + entropyDs", 
        "entropyWs - entropySs", "entropyWs + entropySs", 
        "entropyWs - entropyDs", "entropyWs + entropyDs"]

for scmb in cmbs:
    exec("feat = %s" % scmb)
    pc1, pv1 = _ss.pearsonr(soc_skils[ths], feat[ths])
    ss1      = "***" if pv1 < 0.05 else ""    
    pc2, pv2 = _ss.pearsonr(imag[ths], feat[ths])
    ss2      = "***" if pv2 < 0.05 else ""    
    pc3, pv3 = _ss.pearsonr(rout[ths], feat[ths])
    ss3      = "***" if pv3 < 0.05 else ""        
    pc4, pv4 = _ss.pearsonr(switch[ths], feat[ths])
    ss4      = "***" if pv4 < 0.05 else ""        
    pc5, pv5 = _ss.pearsonr(fact_pat[ths], feat[ths])
    ss5      = "***" if pv5 < 0.05 else ""
    pc6, pv6 = _ss.pearsonr(AQ28scrs[ths], feat[ths])
    ss6      = "***" if pv6 < 0.05 else ""
    print(scmb)
    print("1 %(1) .3f  %(2).3f  %(3)s" % {"1" : pc1, "2" : pv1, "3" : ss1})
    print("2 %(1) .3f  %(2).3f  %(3)s" % {"1" : pc2, "2" : pv2, "3" : ss2})
    print("3 %(1) .3f  %(2).3f  %(3)s" % {"1" : pc3, "2" : pv3, "3" : ss3})
    print("4 %(1) .3f  %(2).3f  %(3)s" % {"1" : pc4, "2" : pv4, "3" : ss4})
    print("5 %(1) .3f  %(2).3f  %(3)s" % {"1" : pc5, "2" : pv5, "3" : ss5})
    print("6 %(1) .3f  %(2).3f  %(3)s" % {"1" : pc6, "2" : pv6, "3" : ss6})
    



    
