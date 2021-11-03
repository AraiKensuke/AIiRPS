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
import AIiRPS.models.CRutils as _crut
import AIiRPS.models.empirical as _emp

import GCoh.eeg_util as _eu
import matplotlib.ticker as ticker

__1st__ = 0
__2nd__ = 1
__ALL__ = 2

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1

#  sum_sd
#  entropyL
#  isi_cv, isis_corr


def predict_new(Xs, y, fitdat):
    N  = y.shape[0]
    testdat = _N.setdiff1d(_N.arange(N), fitdat)
    Xs_fit = Xs[fitdat]
    y_fit  = y[fitdat]
    model = _skl.LinearRegression()
    model.fit(Xs_fit, y_fit)
    model_in = model.intercept_
    model_as = model.coef_
    Xs_test = Xs[testdat]
    y_test  = y[testdat]
    y_pred = _N.dot(Xs_test, model_as) + model_in
    return Xs_test, y_test, y_pred

def add_nz_probs(probs, nz_amp):
    probs_m1p1 = 2*(probs - 0.5)

    remap_probs_m1p1 = _N.arctanh(probs_m1p1)
    remap_probs_m1p1 += nz_amp*_N.random.randn(101)

    nz_probs_m1p1 = _N.tanh(remap_probs_m1p1)
    
    nz_probs = 0.5*nz_probs_m1p1 + 0.5
    return nz_probs

def only_complete_data(partIDs, TO, label, SHF_NUM):
    pid = -1
    incomplete_data = []
    for partID in partIDs:
        pid += 1

        dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_1.dmp" % {"rpsm" : partID, "lb" : label}))
        _prob_mvs = dmp["cond_probs"][SHF_NUM]
        __hnd_dat = dmp["all_tds"][SHF_NUM]
        _hnd_dat   = __hnd_dat[0:TO]

        if _hnd_dat.shape[0] < TO:
            incomplete_data.append(pid)
    for inc in incomplete_data[::-1]:
        #  remove from list 
        partIDs.pop(inc)
    return partIDs, incomplete_data

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

def cleanISI(isi, minISI=2):
    ths = _N.where(isi[1:-1] <= minISI)[0] + 1
    
    if len(ths) > 0:
        rebuild = isi.tolist()
        for ih in ths:
            rebuild[ih-1] += minISI//2
            rebuild[ih+1] += minISI//2
        for ih in ths[::-1]:
            rebuild.pop(ih)
        isi = _N.array(rebuild)
    return isi
            
def entropy3(_sig, N, repeat=None, nz=0):
    cube = _N.zeros((N, N, N))   #  W T L conditions or
    iN   = 1./N

    #print(sig.shape[0])

    if repeat is not None:
        newlen = _sig.shape[0]*repeat
        sig = _N.empty((newlen, 3))
        sig[:, 0] = _N.repeat(_sig[:, 0], repeat) + nz*_N.random.randn(newlen)
        sig[:, 1] = _N.repeat(_sig[:, 1], repeat) + nz*_N.random.randn(newlen)
        sig[:, 2] = _N.repeat(_sig[:, 2], repeat) + nz*_N.random.randn(newlen)
    else:
        sig = _sig
    
    for i in range(sig.shape[0]):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        iz = int(sig[i, 2]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        iz = iz if iz < N else N-1
        cube[ix, iy, iz] += 1

    entropy  = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                p_ijk = cube[i, j, k] / len(sig)
                if p_ijk > 0:
                    entropy += -p_ijk * _N.log(p_ijk)
    return entropy

def entropy2(sig, N):
    #  calculate the entropy
    square = _N.zeros((N, N))
    iN   = 1./N
    for i in range(len(sig)):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        square[ix, iy] += 1

    entropy  = 0
    for i in range(N):
        for j in range(N):
                p_ij = square[i, j] / len(sig)
                if p_ij > 0:
                    entropy += -p_ij * _N.log(p_ij)
    return entropy


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
    while x[ix[i+1]] - x[ix[i]] > 3*dsx:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > 3*dsx:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > 3*dsy:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > 3*dsy:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    print("%(ths)d" % {"ths" : len(ths)})
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

##  Then I expect wins following UPs and DOWNs to also be correlated to AQ28
look_at_AQ = False
data   = "TMB2"
partIDs1=["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07"]
partIDs2=["20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39"]
partIDs3=["20200108_1642-20", "20200109_1504-32"]
partIDs4=["20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]
partIDs5=["20200601_0748-03", "20210529_1923-44", "20210529_1419-14"]
#partIDs6=[ "20210606_1237-17"]
partIDs6=["20210609_1517-23"]
#partIDs7=["20201122_1108-25", "20201121_1959-30", "20201121_2131-38"]
#partIDs7 = ["20200410_2203-19", "20200410_2248-43", "20200415_2034-12", "20200418_2148-58"]
#partIDs7 = ["20200410_2248-43"]
if data == "EEG1":
    partIDs = partIDs1 + partIDs2 + partIDs3 + partIDs4 + partIDs5 + partIDs6# + partIDs7
if data == "RAND":
    USE = 43
    
    _partIDs = os.listdir("/Users/arai/Sites/taisen/DATA/RAND/20210803")
    partIDs = []
    these   = _N.random.choice(_N.arange(len(_partIDs)), USE)
    for i in range(USE):
        partIDs.append(_partIDs[these[i]])
    
if data == "TMB2":
    dates = _rt.date_range(start='7/13/2021', end='10/30/2021')
    #dates = _rt.date_range(start='7/13/2021', end='07/27/2021')
    #dates = _rt.date_range(start='7/27/2021', end='08/20/2021')
    #dates = _rt.date_range(start='07/27/2021', end='08/20/2021')
    #dates = _rt.date_range(start='07/13/2021', end='07/27/2021')
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=800, MinWinLossRat=0.7, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=800, MinWinLossRat=0.7, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=50, maxIGI=30000, MinWinLossRat=0.7, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    #  MinWinLossRat=0.3  ==>  net loss of -60 included.  Saw player like this
    #  at Neurable.  It is not an unreasonable
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=5, maxIGI=800, MinWinLossRat=0.05, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

A1 = []
show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################


win     = 4
smth    = 3
label          = win*10+smth
TO = 300
SHF_NUM = 0

partIDs, incmp_dat = only_complete_data(partIDs, TO, label, SHF_NUM)
strtTr=0
TO -= strtTr

#fig= _plt.figure(figsize=(14, 14))

SHUFFLES = 50

t0 = -5
t1 = 10
cut = 1
all_avgs = _N.empty((len(partIDs), SHUFFLES+1, t1-t0))
netwins  = _N.empty(len(partIDs), dtype=_N.int)
gk = gauKer(1)
gk /= _N.sum(gk)
#gk = None

corrs_all = _N.empty((3, 6))
corrs_sing = _N.empty((len(partIDs), 3, 6))

perform   = _N.empty(len(partIDs))

pid = 0

ts  = _N.arange(t0-2, t1-2)
signal_5_95 = _N.empty((len(partIDs), 4, t1-t0))

pc_sum = _N.empty(len(partIDs))
pc_sum01 = _N.empty(len(partIDs))
pc_sum02 = _N.empty(len(partIDs))
pc_sum12 = _N.empty(len(partIDs))
isis    = _N.empty(len(partIDs))
isis_sd    = _N.empty(len(partIDs))
isis_cv    = _N.empty(len(partIDs))
isis_lv    = _N.empty(len(partIDs))
isis_corr    = _N.empty(len(partIDs))
coherence    = _N.empty(len(partIDs))
ages      = _N.empty(len(partIDs))
gens      = _N.empty(len(partIDs))
Engs      = _N.empty(len(partIDs))

corr_UD    = _N.empty((len(partIDs), 3))

score  = _N.empty(len(partIDs))
moresim  = _N.empty(len(partIDs))
moresiment  = _N.empty(len(partIDs))
sum_sd = _N.empty((len(partIDs), 3, 3))
sum_sd2 = _N.empty((len(partIDs), 3, 3))
sum_cv = _N.empty((len(partIDs), 3, 3))
marginalCRs = _N.empty((len(partIDs), 3, 3))
entropyDSU = _N.empty((len(partIDs), 3))
entropyD = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropyS = _N.empty(len(partIDs))
entropyU = _N.empty(len(partIDs))
entropyUD2 = _N.empty(len(partIDs))
entropyS2 = _N.empty(len(partIDs))
entropyDr = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropySr = _N.empty(len(partIDs))
entropyUr = _N.empty(len(partIDs))
entropyW = _N.empty(len(partIDs))   #  
entropyT = _N.empty(len(partIDs))
entropyL = _N.empty(len(partIDs))
entropyW2 = _N.empty(len(partIDs))   #  
entropyT2 = _N.empty(len(partIDs))
entropyL2 = _N.empty(len(partIDs))

entropyST = _N.empty(len(partIDs))
entropySW = _N.empty(len(partIDs))
pfrm_change36 = _N.empty(len(partIDs))
pfrm_change69 = _N.empty(len(partIDs))
pfrm_change912= _N.empty(len(partIDs))

win_aft_win  = _N.empty(len(partIDs))
win_aft_los  = _N.empty(len(partIDs))
win_aft_tie  = _N.empty(len(partIDs))
tie_aft_win  = _N.empty(len(partIDs))
tie_aft_los  = _N.empty(len(partIDs))
tie_aft_tie  = _N.empty(len(partIDs))
los_aft_win  = _N.empty(len(partIDs))
los_aft_los  = _N.empty(len(partIDs))
los_aft_tie  = _N.empty(len(partIDs))
imax_imin_pfrm36 = _N.empty((len(partIDs), 2), dtype=_N.int)
imax_imin_pfrm69 = _N.empty((len(partIDs), 2), dtype=_N.int)
imax_imin_pfrm912 = _N.empty((len(partIDs), 2), dtype=_N.int)

u_or_d_res   = _N.empty(len(partIDs))
u_or_d_tie   = _N.empty(len(partIDs))
s_res        = _N.empty(len(partIDs))
s_tie        = _N.empty(len(partIDs))

up_res   = _N.empty(len(partIDs))
dn_res   = _N.empty(len(partIDs))
stay_res         = _N.empty(len(partIDs))
stay_tie         = _N.empty(len(partIDs))

AQ28scrs  = _N.empty(len(partIDs))
AQ28scrs_real= _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))    
end_strts     = _N.empty(len(partIDs))    

all_maxs  = []

aboves = []
belows = []

all_prob_mvs = []
all_prob_pcs = []
istrtend     = 0
strtend      = _N.zeros(len(partIDs)+1, dtype=_N.int)

incomplete_data = []
IGIs         = _N.empty(len(partIDs))

for partID in partIDs:
    pid += 1

    # if data == "TMB2":
    #     __hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB2")
    # if data == "EEG1":
    #     __hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=None, expt="EEG1")
    # if data == "RAND":
    #     __hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="RAND")
        
    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_1.dmp" % {"rpsm" : partID, "lb" : label}))
    _prob_mvs = dmp["cond_probs"][SHF_NUM][:, strtTr:]
    _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
    netwins[pid-1] = _N.sum(_hnd_dat[:, 2])
    IGIs[pid-1]    = _N.sum(_N.diff(_hnd_dat[:, 3])) / 300000 # IGIs


fig = _plt.figure()
throwAway = _N.where(IGIs < 0.8)[0]
keep      = _N.where(IGIs >= 0.8)[0]
_plt.scatter(IGIs[throwAway], netwins[throwAway], s=8, color="#CDCDCD")
_plt.scatter(IGIs[keep], netwins[keep], s=8, color="black")
_plt.axvline(x=0.8, ls=":", color="red")
_plt.axhline(y=0., ls=":", color="grey")
_plt.ylabel("#wins - #lose / 300 games", fontsize=18)
_plt.yticks(fontsize=16)
_plt.xlabel("mean inter-game interval (sec)", fontsize=18)
_plt.xticks(fontsize=16)
fig.subplots_adjust(left=0.2, bottom=0.2)
_plt.savefig("TMB2_score_dist")
