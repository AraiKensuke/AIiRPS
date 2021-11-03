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
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=10, maxIGI=30000, MinWinLossRat=0.1, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

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

t0 = -5
t1 = 10
cut = 1
all_avgs = _N.empty((len(partIDs), t1-t0))
netwins  = _N.empty(len(partIDs), dtype=_N.int)
gk = gauKer(1)
gk /= _N.sum(gk)
#gk = None

UD_diff   = _N.empty((len(partIDs), 3))
corrs_all = _N.empty((3, 6))
corrs_sing = _N.empty((len(partIDs), 3, 6))

perform   = _N.empty(len(partIDs))

pid = 0

ts  = _N.arange(t0-2, t1-2)

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

cond_distinguished = _N.empty((len(partIDs), 2))  #  for actions, conditions distinguished.  ST and SW.  
actions_independent     = _N.empty((len(partIDs), 3))  #  for actions, conditions distinguished
stay_amps     = _N.empty((len(partIDs), 3))  #  for actions, conditions distinguished

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
shrtst     = _N.empty(len(partIDs))
longst     = _N.empty(len(partIDs))    

all_maxs  = []

aboves = []
belows = []

all_prob_mvs = []
all_prob_pcs = []
istrtend     = 0
strtend      = _N.zeros(len(partIDs)+1, dtype=_N.int)

incomplete_data = []
gkISI = gauKer(1)
gkISI /= _N.sum(gkISI)


for partID in partIDs:
    pid += 1

    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_1.dmp" % {"rpsm" : partID, "lb" : label}))
    _prob_mvs = dmp["cond_probs"][SHF_NUM][:, strtTr:]
    _prob_mvs_STSW = dmp["cond_probsSTSW"][SHF_NUM][:, strtTr:]    
    _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
    end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])

    #_hnd_dat   = __hnd_dat[0:TO]

    inds =_N.arange(_hnd_dat.shape[0])
    igis = _N.diff(_hnd_dat[:, 3])
    srt_igis = _N.sort(igis)
    shrtst[pid-1] = _N.mean(srt_igis[0:50])
    longst[pid-1] = _N.mean(srt_igis[250:])

    """
    ####
    wins = _N.where(_hnd_dat[0:TO-2, 2] == 1)[0]
    ww   = _N.where(_hnd_dat[wins+1, 2] == 1)[0]
    wt   = _N.where(_hnd_dat[wins+1, 2] == 0)[0]
    wl   = _N.where(_hnd_dat[wins+1, 2] == -1)[0]        
    win_aft_win[pid-1] = len(ww) / len(wins)
    tie_aft_win[pid-1] = len(wt) / len(wins)
    los_aft_win[pid-1] = len(wl) / len(wins)
    ####    
    loses = _N.where(_hnd_dat[0:TO-2, 2] == -1)[0]
    lw   = _N.where(_hnd_dat[loses+1, 2] == 1)[0]
    lt   = _N.where(_hnd_dat[loses+1, 2] == 0)[0]
    ll   = _N.where(_hnd_dat[loses+1, 2] == -1)[0]    
    win_aft_los[pid-1] = len(lw) / len(loses)
    tie_aft_los[pid-1] = len(lt) / len(loses)
    los_aft_los[pid-1] = len(ll) / len(loses)    
    ####    
    ties = _N.where(_hnd_dat[0:TO-2, 2] == 0)[0]
    tw   = _N.where(_hnd_dat[ties+1, 2] == 1)[0]
    tt   = _N.where(_hnd_dat[ties+1, 2] == 0)[0]
    tl   = _N.where(_hnd_dat[ties+1, 2] == -1)[0]    
    win_aft_tie[pid-1] = len(tw) / len(ties)
    tie_aft_tie[pid-1] = len(tt) / len(ties)
    los_aft_tie[pid-1] = len(tl) / len(ties)    
    ####

    ###
    #  RSP   1 (R)   3  (P)  lose
    u_or_d = _N.where(_hnd_dat[0:TO-2, 0] != _hnd_dat[1:TO-1, 0])[0]
    #  R->S   S->P
    dn_only = _N.where(((_hnd_dat[0:TO-2, 0] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                       ((_hnd_dat[0:TO-2, 0] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                       ((_hnd_dat[0:TO-2, 0] == 3) & (_hnd_dat[1:TO-1, 0] == 1)))[0]
    up_only = _N.where(((_hnd_dat[0:TO-2, 0] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                       ((_hnd_dat[0:TO-2, 0] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                       ((_hnd_dat[0:TO-2, 0] == 3) & (_hnd_dat[1:TO-1, 0] == 2)))[0]

    u_or_d_res[pid-1] = _N.sum(_hnd_dat[u_or_d+1, 2])
    ties_after_u_or_d = _N.where(_hnd_dat[u_or_d+1, 2] == 0)[0]
    u_or_d_tie[pid-1] = len(ties_after_u_or_d)
    up_res[pid-1] = _N.sum(_hnd_dat[up_only+1, 2])
    dn_res[pid-1] = _N.sum(_hnd_dat[dn_only+1, 2])    
    stay   = _N.where(_hnd_dat[0:TO-2, 0] == _hnd_dat[1:TO-1, 0])[0]    
    stay_res[pid-1] = _N.sum(_hnd_dat[stay+1, 2])
    ties_after_stay = _N.where(_hnd_dat[stay+1, 2] == 0)[0]
    stay_tie[pid-1] = len(ties_after_stay)
    """
    marginalCRs[pid-1] = _emp.marginalCR(_hnd_dat)
    prob_mvs  = _prob_mvs[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size
    prob_mvs_STSW  = _prob_mvs_STSW[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size    
    prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
    prob_mvs_STSW = prob_mvs_STSW.reshape((3, 2, prob_mvs_STSW.shape[1]))
    #  _N.sum(prob_mvs_STSW[0], axis=0) = 1, 1, 1, 1, 1, 1, (except at ends)
    dbehv = _crut.get_dbehv(prob_mvs, gk)
    maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2) #  3 from label71
    MLAG = 20
    tlag, AC = _eu.autocorrelate(dbehv, MLAG)
    # dAC = _N.diff(AC)
    # AC_pks = _N.where((dAC[0:-1] > 0) & (dAC[1:] <= 0))[0]
    # coherence[pid-1] = _N.std(_N.diff(AC_pks))
    decr = True
    for i in range(2, MLAG-1):
        if decr:
            if (AC[MLAG+i-1] >= AC[MLAG+i]) and (AC[MLAG+i] <= AC[MLAG+i+1]):
                decr = False
                iLow = i+MLAG
        else:
            if (AC[MLAG+i-1] <= AC[MLAG+i]) and (AC[MLAG+i] >= AC[MLAG+i+1]):
                iHigh = i+MLAG
                break
    coherence[pid-1] = AC[iHigh] - AC[iLow]
    ACmin = _N.min(AC[MLAG:])
    coherence[pid-1] = ACmin
    # fig = _plt.figure()
    # _plt.suptitle("%.2f" % coherence[pid-1])
    # _plt.acorr(dbehv - _N.mean(dbehv), maxlags=MLAG)
    # _plt.ylim(-1, 1)
    # _plt.grid()

    all_prob_mvs.append(prob_mvs)    #  plot out to show range of CRs

    prob_pcs = _N.empty((len(maxs)-1, 3, 3))
    for i in range(len(maxs)-1):
        prob_pcs[i] = _N.mean(prob_mvs[:, :, maxs[i]:maxs[i+1]], axis=2)
        #  _N.sum(prob_mvs[:, :, 10], axis=1) == [1, 1, 1]
    all_prob_pcs.extend(prob_pcs)
    istrtend += prob_pcs.shape[0]
    strtend[pid-1+1] = istrtend

    PCS=5
    #  prob_mvs[:, 0] - for each time point, the DOWN probabilities following 3 different conditions
    #  prob_mvs[0]    - for each time point, the DOWN probabilities following 3 different conditions    


    #  ST | WIN  and SW | WIN
    #  probST[0] == _prob_mvs_STSW[0]
    #  probSW[0] == _prob_mvs_STSW[1]
    #  ST | TIE  and SW | TIE
    #  probST[1] == _prob_mvs_STSW[2]
    #  probSW[1] == _prob_mvs_STSW[3]
    #  ST | LOS  and SW | LOS
    #  probST[2] == _prob_mvs_STSW[4]
    #  probSW[2] == _prob_mvs_STSW[5]
    
    #probSW = (prob_mvs[:, 0] + prob_mvs[:, 2])
    #probST = (prob_mvs[:, 1])
    #  probST ->  the prob of stay in W, T, L
    #entsSTSW = _N.array([entropy3(probST.T, PCS), entropy3(probSW.T, PCS)])
    condition_distinguished = _N.array([entropy3(prob_mvs_STSW[:, 0].T, PCS), entropy3(prob_mvs_STSW[:, 1].T, PCS)])
    wtl_independent = _N.array([entropy2(prob_mvs_STSW[0].T, PCS), entropy2(prob_mvs_STSW[1].T, PCS), entropy2(prob_mvs_STSW[2].T, PCS)])
    stay_amp = _N.array([_N.std(prob_mvs_STSW[0, 0]), _N.std(prob_mvs_STSW[1, 0]), _N.std(prob_mvs_STSW[2, 0])])

    entsDSU = _N.array([entropy3(prob_mvs[:, 0].T, PCS),
                        entropy3(prob_mvs[:, 1].T, PCS),
                        entropy3(prob_mvs[:, 2].T, PCS)])

    ##  
    pUD_WTL = _N.array([prob_mvs[0, 0] + prob_mvs[0, 2],
                        prob_mvs[1, 0] + prob_mvs[1, 2],
                        prob_mvs[2, 0] + prob_mvs[2, 2]])
    pS_WTL  = _N.array([prob_mvs[0, 1], prob_mvs[1, 1], prob_mvs[2, 1]])
    entsUD_S = _N.array([entropy3(pUD_WTL.T, PCS),
                         entropy3(pS_WTL.T,  PCS)])

    # entsDSUr = _N.array([entropy3(prob_mvs[:, 0].T, PCS, repeat=10, nz=0.1),
    #                      entropy3(prob_mvs[:, 1].T, PCS, repeat=10, nz=0.1),
    #                      entropy3(prob_mvs[:, 2].T, PCS, repeat=10, nz=0.1)])
    
    #entsSTSW = _N.array([entropy2(prob_mvs[:, 0].T, PCS), entropy3(prob_mvs[:, 1].T, PCS), entropy3(prob_mvs[:, 2].T, PCS)])

    pW_stsw = _N.array([prob_mvs[0, 0] + prob_mvs[0, 2], prob_mvs[0, 1]])
    pT_stsw = _N.array([prob_mvs[1, 0] + prob_mvs[1, 2], prob_mvs[1, 1]])
    pL_stsw = _N.array([prob_mvs[2, 0] + prob_mvs[2, 2], prob_mvs[2, 1]])


    #  Is TIE like a WIN or TIE like a LOSE?
    #  ENT_WT = entropy of (UP|WIN and UP|TIE) + entropy (DN|WIN and DN|TIE) + entropy (UP|WIN and UP|TIE)
    #  ENT_LT = entropy of (UP|LOS and UP|TIE) + entropy (DN|LOS and DN|TIE) + entropy (UP|LOS and UP|TIE)
    probU  = _N.empty((2, prob_mvs.shape[2]))
    probD  = _N.empty((2, prob_mvs.shape[2]))
    probS  = _N.empty((2, prob_mvs.shape[2]))
    probU[0] = prob_mvs[0, 2]
    probU[1] = prob_mvs[1, 2]
    probS[0] = prob_mvs[0, 1]
    probS[1] = prob_mvs[1, 1]    
    probD[0] = prob_mvs[0, 0]
    probD[1] = prob_mvs[1, 0]    

    
    #ENT_WT = entropy2(probU.T, PCS) + entropy2(probS.T, PCS) + entropy2(probD.T, PCS)
    ENT_WT = entropy2(probS.T, PCS)
    probU[0] = prob_mvs[2, 2]
    probU[1] = prob_mvs[1, 2]
    probS[0] = prob_mvs[2, 1]
    probS[1] = prob_mvs[1, 1]    
    probD[0] = prob_mvs[2, 0]
    probD[1] = prob_mvs[1, 0]    

    #ENT_LT = entropy2(probU.T, PCS) + entropy2(probS.T, PCS) + entropy2(probD.T, PCS)
    ENT_LT = entropy2(probS.T, PCS)
    moresiment[pid-1] = ENT_WT - ENT_LT
    
    prob_mvs[0, 0]
    #entsWTL2 = _N.array([entropy2(pW_stsw.T, PCS), entropy2(pT_stsw.T, PCS), entropy2(pL_stsw.T, PCS)])
    entsWTL3 = _N.array([entropy3(prob_mvs[0].T, PCS),
                         entropy3(prob_mvs[1].T, PCS),
                         entropy3(prob_mvs[2].T, PCS)])
    
    probWst = prob_mvs[0, 1]
    probWsw = prob_mvs[0, 0] + prob_mvs[0, 2]
    datW    = _N.empty((prob_mvs.shape[2], 2))
    datW[:, 0] = probWst
    datW[:, 1] = probWsw
    probTst = prob_mvs[1, 1]
    probTsw = prob_mvs[1, 0] + prob_mvs[1, 2]
    datT    = _N.empty((prob_mvs.shape[2], 2))
    datT[:, 0] = probTst
    datT[:, 1] = probTsw
    probLst = prob_mvs[2, 1]
    probLsw = prob_mvs[2, 0] + prob_mvs[2, 2]
    datL    = _N.empty((prob_mvs.shape[2], 2))
    datL[:, 0] = probLst
    datL[:, 1] = probLsw

    #entsWTL = _N.array([entropy2(datW, PCS), entropy2(datT, PCS), entropy2(datL, PCS)])    

    # entropyDr[pid-1] = entsDSUr[0]
    # entropySr[pid-1] = entsDSUr[1]
    # entropyUr[pid-1] = entsDSUr[2]

    UD_diff[pid-1, 0] = _N.std(prob_mvs[0, 0] - prob_mvs[0, 2])
    UD_diff[pid-1, 1] = _N.std(prob_mvs[1, 0] - prob_mvs[1, 2])
    UD_diff[pid-1, 2] = _N.std(prob_mvs[2, 0] - prob_mvs[2, 2])
    
    entropyD[pid-1] = entsDSU[0]
    entropyS[pid-1] = entsDSU[1]
    entropyU[pid-1] = entsDSU[2]

    entropyUD2[pid-1] = entsUD_S[0]
    entropyS2[pid-1]  = entsUD_S[1]    
    entropyW[pid-1] = entsWTL3[0]
    entropyT[pid-1] = entsWTL3[1]
    entropyL[pid-1] = entsWTL3[2]
    entropyW2[pid-1] = wtl_independent[0]
    entropyT2[pid-1] = wtl_independent[1]
    entropyL2[pid-1] = wtl_independent[2]
    actions_independent[pid-1] = wtl_independent                   #  3
    cond_distinguished[pid-1] = condition_distinguished  #  2
    stay_amps[pid-1] = stay_amp     # 3 components

    THRisi = 2
    isi   = cleanISI(_N.diff(maxs), minISI=2)
    #isi   = _N.diff(maxs)

    # largeEnough = _N.where(_isi > THRisi)[0]
    # tooSmall    = _N.where(_isi <= 3)[0]
    # isi    = _isi[largeEnough]
    #pc, pv = _ss.pearsonr(isi[0:-1], isi[1:])

    #fisi = _N.convolve(isi, gkISI, mode="same")    
    pc, pv = rm_outliersCC_neighbors(isi[0:-1], isi[1:])
    #pc, pv = rm_outliersCC_neighbors(fisi[0:-1], fisi[1:])
    #pc, pv = _ss.pearsonr(fisi[0:-1], fisi[1:])
    #pc2, pv2 = rm_outliersCC_neighbors(fisi[0:-2], fisi[2:])
    #fig = _plt.figure()
    #_plt.plot(fisi)
    #_plt.suptitle("%(1).3f    %(2).3f" % {"1" : pc, "2" : pc2})

    #_plt.savefig("isi%d" % (pid-1))
    #_plt.close()

    isis_corr[pid-1] = pc
    isis_sd[pid-1] = _N.std(isi)
    isis[pid-1] = _N.mean(isi)        
    isis_cv[pid-1] = isis_sd[pid-1] / isis[pid-1]
    isis_lv[pid-1] = (3/(len(isi)-1))*_N.sum((isi[0:-1] - isi[1:])**2 / (isi[0:-1] + isi[1:])**2 )    
    all_maxs.append(isi)    

    sds = _N.std(prob_mvs, axis=2)
    #sds = _N.std(prob_pcs, axis=0)
    mns = _N.mean(prob_mvs, axis=2)    
    sum_cv[pid-1] = sds/mns
    sum_sd[pid-1] = sds
    score[pid-1] = _N.sum(_hnd_dat[:, 2])# / _hnd_dat.shape[0]

    pc01_0, pv01_0 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[1, 0])    
    pc01_1, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 1])
    pc01_2, pv01_2 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[1, 2])        

    pc02_0, pv02_0 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[2, 0])    
    pc02_1, pv02_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 1])
    pc02_2, pv02_2 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[2, 2])    

    pc12_0, pv12_0 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[2, 0])
    pc12_1, pv12_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 1])
    pc12_2, pv12_2 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 2])

    pc_sum01[pid-1] = pc01_0+pc01_1+pc01_2
    pc_sum02[pid-1] = pc02_0+pc02_1+pc02_2
    pc_sum12[pid-1] = pc12_0+pc12_1+pc12_1

    #  CC(T and L) - CC(W and T)q   Is TIE more similar to WIN or LOSE?
    moresim[pid-1] = pc12_0+pc12_1+pc12_2 - (pc01_0+pc01_1+pc01_2)
    #moresim[pid-1] = pc12_2+pc12_1 - (pc01_2+pc01_1)
    # trm01 = _N.empty(3)
    # trm12 = _N.empty(3)
    
    # for i in range(3):  # U, D, S
    #     bot01= _N.sum(_N.abs(prob_mvs[0, i])) + _N.sum(_N.abs(prob_mvs[1, i]))
    #     bot12= _N.sum(_N.abs(prob_mvs[2, i])) + _N.sum(_N.abs(prob_mvs[1, i]))
    #     top01= _N.sum(_N.abs(prob_mvs[0, i] - prob_mvs[1, i]))
    #     top12= _N.sum(_N.abs(prob_mvs[2, i] - prob_mvs[1, i]))
    #     trm01[i] = top01 / bot01
    #     trm12[i] = top12 / bot12        
    # moresim[pid-1] = _N.sum(trm01) - _N.sum(trm12)

    if look_at_AQ:
        AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})
        ages[pid-1], gens[pid-1], Engs[pid-1] = _rt.Demo("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/DQ1.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})        

    netwins[pid-1] = _N.sum(_hnd_dat[:, 2])
    wins = _N.where(_hnd_dat[:, 2] == 1)[0]
    losses = _N.where(_hnd_dat[:, 2] == -1)[0]
    perform[pid -1] = len(wins) / (len(wins) + len(losses))

    hnd_dat = _hnd_dat[inds]

    avgs = _N.empty((len(maxs)-2*cut, t1-t0))
    for im in range(cut, len(maxs)-cut):
        avgs[im-1, :] = hnd_dat[maxs[im]+t0:maxs[im]+t1, 2]

    all_avgs[pid-1] = _N.mean(avgs, axis=0)

    srtd   = _N.sort(all_avgs[pid-1, 1:], axis=0)

    #pfrm_change36[pid-1] = _N.max(signal_5_95[pid-1, 0, 3:6]) - _N.min(signal_5_95[pid-1, 0, 3:6])

    sInds = _N.argsort(all_avgs[pid-1, 3:6])
    if sInds[2] - sInds[0] > 0:
        m36 = 1
    else:
        m36 = -1
    sInds = _N.argsort(all_avgs[pid-1, 6:9])
    if sInds[2] - sInds[0] > 0:
        m69 = 1
    else:
        m69 = -1
    sInds = _N.argsort(all_avgs[pid-1, 9:12])
    if sInds[2] - sInds[0] > 0:
        m912 = 1
    else:
        m912 = -1

    imax36 = _N.argmax(all_avgs[pid-1, 3:6])+3
    imin36 = _N.argmin(all_avgs[pid-1, 3:6])+3
    imax69 = _N.argmax(all_avgs[pid-1, 6:9])+6
    imin69 = _N.argmin(all_avgs[pid-1, 6:9])+6    
    imax912= _N.argmax(all_avgs[pid-1, 9:12])+9
    imin912= _N.argmin(all_avgs[pid-1, 9:12])+9    

    imax_imin_pfrm36[pid-1, 0] = imin36
    imax_imin_pfrm36[pid-1, 1] = imax36
    imax_imin_pfrm69[pid-1, 0] = imin69
    imax_imin_pfrm69[pid-1, 1] = imax69
    imax_imin_pfrm912[pid-1, 0]= imin912
    imax_imin_pfrm912[pid-1, 1]= imax912
    
    pfrm_change36[pid-1] = all_avgs[pid-1, imax36] - all_avgs[pid-1, imin36]
    pfrm_change69[pid-1] = all_avgs[pid-1, imax69] - all_avgs[pid-1, imin69]
    pfrm_change912[pid-1]= all_avgs[pid-1, imax912] - all_avgs[pid-1, imin912]

    #fig.add_subplot(6, 6, pid)

    # _plt.title(netwins[pid-1] )
    # _plt.plot(ts, signal_5_95[pid-1, 0], marker=".", ms=10)
    # _plt.plot(ts, signal_5_95[pid-1, 1])
    # _plt.plot(ts, signal_5_95[pid-1, 2])
    # _plt.axvline(x=-0.5, ls="--")
        
pc_sum = pc_sum01 + pc_sum02 + pc_sum12
        
###############  sort by entropyU.  Show 
all_prob_mvsA = _N.array(all_prob_mvs)
sinds = entropyU.argsort()
min_prob_mvs = all_prob_mvsA[sinds[0]]
max_prob_mvs = all_prob_mvsA[sinds[-1]]

fig = _plt.figure(figsize=(5.2, 3))
fig.add_subplot(2, 1, 1)
_plt.plot(min_prob_mvs[2, 2], label=r"$p_k($UP$|$L$)$")
_plt.plot(min_prob_mvs[1, 2], label=r"$p_k($UP$|$T$)$")
_plt.plot(min_prob_mvs[0, 2], label=r"$p_k($UP$|$W$)$")
_plt.legend(fontsize=8)
_plt.xticks(_N.arange(0, 300, 70), fontsize=13)
_plt.xlim(0, 410)
fig.add_subplot(2, 1, 2)
_plt.plot(max_prob_mvs[2, 2], label=r"$p_k($UP$|$L$)$")
_plt.plot(max_prob_mvs[1, 2], label=r"$p_k($UP$|$T$)$")
_plt.plot(max_prob_mvs[0, 2], label=r"$p_k($UP$|$W$)$")
_plt.legend(fontsize=8)
_plt.xticks(_N.arange(0, 300, 70), fontsize=13)
_plt.xlabel(r"Game # $k$", fontsize=15)
_plt.xlim(0, 410)
fig.subplots_adjust(bottom=0.18, hspace=0.4)
_plt.savefig("EntropyU_prob_mvs_%(lb)d" % {"lb" : label}, transparent=True)

###############  sort by entropyU.  Show
#signal_5_95
sinds = pfrm_change69.argsort()
lo69 = _N.mean(all_avgs[sinds[0:10]], axis=0)
hi69 = _N.mean(all_avgs[sinds[-11:-1]], axis=0)
fig = _plt.figure(figsize=(5.5, 4))
fig.add_subplot(1, 1, 1)
_plt.plot(lo69, color="blue", lw=2)
_plt.plot(hi69, color="red", lw=5)
_plt.axhline(y=0, ls=":", color="grey")
_plt.axvline(x=(t1-t0-1)/2, ls=":", color="grey")
_plt.xticks(_N.arange(t1-t0), _N.arange(t0 - win//2, t1 - win//2), fontsize=19)
_plt.yticks(fontsize=19)
_plt.ylabel("p(W) - p(L)", fontsize=20)
_plt.xlabel("lag around rule-chng (#games)", fontsize=20)
_plt.xlim(0, t1-t0-1)
_plt.ylim(-0.19, 0.19)
fig.subplots_adjust(bottom=0.21, hspace=0.4, left=0.23, right=0.98)
_plt.savefig("jump69_%(lb)d" % {"lb" : label}, transparent=True)

###############  sort by moresim.  Show 
sinds = moresim.argsort()
min_prob_mvs = all_prob_mvsA[sinds[0]]
max_prob_mvs = all_prob_mvsA[sinds[-1]]


#  The Tie more similar to Win
lblsz=14
fig = _plt.figure(figsize=(6, 2.2))
#############################################
fig.add_subplot(3, 2, 1)
_plt.title(r"$|p(\cdot | $TIE$) - p(\cdot | $WIN$)|$", fontsize=(lblsz+3))
_plt.plot(_N.abs(min_prob_mvs[1, 0] - min_prob_mvs[0, 0]), color="black")   #  DN|WIN
_plt.ylim(0, 1)
_plt.xticks([])
_plt.ylabel("DN", fontsize=lblsz)
#############################################
fig.add_subplot(3, 2, 3)
_plt.plot(_N.abs(min_prob_mvs[1, 1] - min_prob_mvs[0, 1]), color="black")   #  ST|WIN
_plt.ylim(0, 1)
_plt.xticks([])
_plt.ylabel("ST", fontsize=lblsz)
#############################################
fig.add_subplot(3, 2, 5)
_plt.plot(_N.abs(min_prob_mvs[1, 2] - min_prob_mvs[0, 2]), color="black")   #  UP|WIN
_plt.ylim(0, 1)
_plt.xticks(fontsize=(lblsz-2))
_plt.ylabel("UP", fontsize=lblsz)
_plt.xlabel("Game #", fontsize=lblsz)
#############################################
fig.add_subplot(3, 2, 2)
_plt.title(r"$|p(\cdot | $TIE$) - p(\cdot | $LOS$)|$", fontsize=(lblsz+3))
_plt.plot(_N.abs(min_prob_mvs[1, 0] - min_prob_mvs[2, 0]), color="black")   #  DN|TIE
_plt.ylim(0, 1)
_plt.xticks([])
_plt.yticks([])
#############################################
fig.add_subplot(3, 2, 4)
_plt.plot(_N.abs(min_prob_mvs[1, 1] - min_prob_mvs[2, 1]), color="black")   #  ST|TIE
_plt.ylim(0, 1)
_plt.xticks([])
_plt.yticks([])
#############################################
fig.add_subplot(3, 2, 6)
_plt.plot(_N.abs(min_prob_mvs[1, 2] - min_prob_mvs[2, 2]), color="black")   #  UP|TIE
_plt.ylim(0, 1)
_plt.yticks([])
_plt.xticks(fontsize=(lblsz-2))
_plt.xlabel("Game #", fontsize=lblsz)
fig.subplots_adjust(wspace=0.1, bottom=0.22, hspace=0.3, top=0.86)
_plt.savefig("diff_win_similar_tie_%(lb)d.pdf" % {"lb" : label}, transparent=True)

#  The Tie more similar to Los
fig = _plt.figure(figsize=(5, 4))
fig.add_subplot(3, 2, 1)
_plt.title(r"|$p(\cdot | TIE) - p(\cdot | WIN)$|")
_plt.plot(max_prob_mvs[1, 0] - max_prob_mvs[0, 0], color="black")   #  DN|WIN
_plt.ylim(-1, 1)
_plt.xticks([])
fig.add_subplot(3, 2, 3)
_plt.plot(max_prob_mvs[1, 1] - max_prob_mvs[0, 1], color="black")   #  ST|WIN
_plt.ylim(-1, 1)
_plt.xticks([])
fig.add_subplot(3, 2, 5)
_plt.plot(max_prob_mvs[1, 2] - max_prob_mvs[0, 2], color="black")   #  UP|WIN
_plt.ylim(-1, 1)
fig.add_subplot(3, 2, 2)
_plt.title(r"|$p(\cdot | TIE) - p(\cdot | LOS)$|")
_plt.plot(max_prob_mvs[1, 0] - max_prob_mvs[2, 0], color="black")   #  DN|TIE
_plt.ylim(-1, 1)
_plt.xticks([])
fig.add_subplot(3, 2, 4)
_plt.plot(max_prob_mvs[1, 1] - max_prob_mvs[2, 1], color="black")   #  ST|TIE
_plt.ylim(-1, 1)
_plt.xticks([])
fig.add_subplot(3, 2, 6)
_plt.plot(max_prob_mvs[1, 2] - max_prob_mvs[2, 2], color="black")   #  UP|TIE
_plt.ylim(-1, 1)
_plt.savefig("diff_win_similar_los_%(lb)d" % {"lb" : label}, transparent=True)

###############  show me inter-event intervals

for by in ["isis_cv", "isis_corr"]:
    by_m = isis_cv.argsort() if by == "isis_cv" else isis_corr.argsort()
    fig = _plt.figure(figsize=(5.2, 3))    
    for ii in range(2):
        intvs = all_maxs[by_m[0]] if ii == 0 else all_maxs[by_m[-2]]
        fig.add_subplot(2, 1, ii+1)
        t = 0
        for i in range(intvs.shape[0]):
            _plt.plot([t, t], [0, 1], color="black")
            t += intvs[i]
        if ii == 1:
            _plt.xlabel(r"Game # $k$", fontsize=15)
        _plt.xticks(_N.arange(0, 300, 70), fontsize=13)
        _plt.xlim(0, 300)
        _plt.yticks([])        
    #_plt.suptitle("Rule change timing (at game)")


    fig.subplots_adjust(bottom=0.25, hspace=0.9)
    
    _plt.savefig("%(by)s_big_small_%(lb)d" % {"by" : by, "lb" : label}, transparent=True)


#  More sim:  If large
#cmp_vs = pc_sum
#cmp_against = "netwins"
#cmp_againsts = ["pc_sum", "netwins", "isis_cv", "rat_sd", "more_sim", "sum_sd", "sum_sdW", ]
#cmp_againsts = ["entropyW", "entropyL", "entropyT", "entropyD", "entropyS", "entropyU"]
#cmp_againsts = ["isis_cv", "isis_corr"]

#cmp_againsts = ["moresim"]
#cmp_againsts = ["entropyUD", "isis_cv", "u_or_d_res", "isis_corr", "entropyD", "entropyS", "entropyU", "entropyT", "entropyW", "entropyL", "entropyT2", "entropyW2", "entropyL2", "sum_sd", "netwins", "moresim"]#, "win_aft_win", "win_aft_los"]
cmp_againsts = ["isis", "isis_cv", "isis_corr", "isis_lv",
                "entropyUD", "entropyD", "entropyS", "entropyU",
                "entropyT2", "entropyW2", "entropyL2",
                "entropyT", "entropyW", "entropyL",                
                "u_or_d_res", "u_or_d_tie",
                "stay_res", "stay_tie",                
                "sum_sd", "netwins", "moresim",
                "win_aft_win", "tie_aft_win", "los_aft_win",
                "win_aft_tie", "tie_aft_tie", "los_aft_tie",
                "win_aft_los", "tie_aft_los", "los_aft_los",                
                "pfrm_change36", "pfrm_change69", "pfrm_change912"]

#cmp_againsts = ["entropyUD", "entropyU", "entropyD"]
#cmp_againsts =  ["pfrm_change36", "pfrm_change69", "pfrm_change912"]
#cmp_againsts = ["moresim"]
#cmp_againsts = ["entropyW", "entropyL"]
#cmp_againsts = ["entropyT", "entropyW", "entropyL", "win_aft_tie", "tie_aft_tie", "los_aft_tie"]

#cmp_againsts = ["sum_sd"]
#cmp_againsts = ["isis_corr", "isis_cv"]
#cmp_againsts = ["sum_cv"]
#cmp_againsts = ["sum_sd"]
#cmp_againsts = ["entropyW", "entropyT", "entropyL", "entropyD", "entropyS", "entropyU", "entropySW", "isis_cv", "netwins", "rat_sd", "sum_cv"]


old_pc = []
new_pc = []
old_pv = []
new_pv = []

nDAT = len(partIDs)
nDAThlf = nDAT//2
allInds = _N.arange(nDAT)

dmp_dat = {}
show_mn = True
mn_mode = 2

winsz=15

fig = _plt.figure(figsize=(10, 10))
if1 = -1
for sfeat1 in ["entropyS", "entropyD", "entropyU"]:
    feat1 = eval(sfeat1)
    if1 += 1
    if2 = -1
    for sfeat2 in ["entropyW2", "entropyT2", "entropyL2"]:
        feat2 = eval(sfeat2)
        if2 += 1
        fig.add_subplot(3, 3, if1*3 + if2 + 1)
        pc, pv = _ss.pearsonr(feat1, feat2)
        _plt.title("%(pc).2f  %(pv).1e" % {"pc" : pc, "pv" : pv}) 
        _plt.scatter(feat1, feat2, color="black", s=5)
        if if2 == 0:
            _plt.ylabel(sfeat1, fontsize=18)
        if if1 == 2:
            _plt.xlabel(sfeat2, fontsize=18)
        _plt.xticks(fontsize=13)
        _plt.yticks(fontsize=13)        
fig.subplots_adjust(wspace=0.25, hspace=0.25)
_plt.savefig("corr_btwn_ent_comps_real2")

fig = _plt.figure(figsize=(10, 10))
if1 = -1
for sfeat1 in ["entropyS", "entropyD", "entropyU"]:
    feat1 = eval(sfeat1)
    if1 += 1
    if2 = -1
    for sfeat2 in ["entropyW", "entropyT", "entropyL"]:
        feat2 = eval(sfeat2)
        if2 += 1
        fig.add_subplot(3, 3, if1*3 + if2 + 1)
        pc, pv = _ss.pearsonr(feat1, feat2)
        _plt.title("%(pc).2f  %(pv).1e" % {"pc" : pc, "pv" : pv}) 
        _plt.scatter(feat1, feat2, color="black", s=5)
        if if2 == 0:
            _plt.ylabel(sfeat1, fontsize=18)
        if if1 == 2:
            _plt.xlabel(sfeat2, fontsize=18)
        _plt.xticks(fontsize=13)
        _plt.yticks(fontsize=13)        
fig.subplots_adjust(wspace=0.25, hspace=0.25)
_plt.savefig("corr_btwn_ent_comps_real")

entropyUD = _N.exp((entropyD + entropyU) / (_N.mean(entropyD + entropyU)))
entropyUD2 = _N.exp(entropyUD2 / _N.mean(entropyUD2))
entropyS2 = _N.exp(entropyS2 / _N.mean(entropyS2))

entropyUDS = _N.exp((entropyD + entropyU + entropyS) / (_N.mean(entropyD + entropyU + entropyS)))
entropyD = _N.exp(entropyD / _N.mean(entropyD))
entropyS = _N.exp(entropyS / _N.mean(entropyS))
entropyU = _N.exp(entropyU / _N.mean(entropyU))
entropyWL = _N.exp((entropyW+entropyL) / _N.mean(entropyW+entropyL))
entropyW = _N.exp(entropyW / _N.mean(entropyW*0.3))
entropyT = _N.exp(entropyT / _N.mean(entropyT*0.3))
entropyL = _N.exp(entropyL / _N.mean(entropyL*0.3))
entropyT2 = _N.exp(entropyT2 / _N.mean(entropyT2*0.2)) / 10
entropyW2 = _N.exp(entropyW2 / _N.mean(entropyW2*0.3)) / 10
entropyL2 = _N.exp(entropyL2 / _N.mean(entropyL2*0.3)) / 10

srtdShrtArgs = _N.argsort(shrtst)
srtdShrtArgs = _N.argsort(shrtst)


fig = _plt.figure()
for n in range(len(partIDs)):
    _plt.plot(all_avgs[n], color="#BCBCBC")
_plt.plot(_N.mean(all_avgs, axis=0), color="black")
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
#_plt.savefig("nrm_avg_%(fn)s_%(lb)d.png" % {"fn" : data, "lb" : label}, transparent=True)
_plt.close()

t0t1 = (end_strts/300)/1000

fastOnes = _N.where((shrtst < 300) & (longst < 900))[0]
slowOnes = _N.where((shrtst > 800) & (longst < 4200))[0]

# shrtst = _N.where(t0t1 < 0.26)[0]
# longst = _N.where(t0t1 > 2.9)[0]

fig = _plt.figure()
_plt.plot(_N.mean(all_avgs[fastOnes], axis=0))
_plt.plot(_N.mean(all_avgs[slowOnes], axis=0))

fig  = _plt.figure()
_plt.hist([isis[fastOnes], isis[slowOnes]], bins=_N.linspace(0, 0.5, 26))

# fig  = _plt.figure()
# _plt.hist([isis_cv[shrtst], isis_cv[longst]], bins=_N.linspace(0, 0.5, 26))

fig  = _plt.figure()
_plt.hist([isis[fastOnes], isis[slowOnes]], bins=_N.linspace(3, 28, 26), label=["fast", "slow"])
_plt.legend()

# for smet1, smet2 in [["entropyS", "entropyW"], ["entropyL", "entropyW"], ["entropyT", "entropyW"], ["moresim", "entropyW"], ["entropyU", "entropyD"], ["pfrm_change69", "entropyW"]]:
#     met1 = eval(smet1)
#     met2 = eval(smet2)    
#     fig = _plt.figure()
#     _plt.suptitle("%(1)s  %(2)s" % {"1" : smet1, "2" : smet2})
#     _plt.scatter(met1[slowOnes], met2[slowOnes], label="slow")
#     _plt.scatter(met1[fastOnes], met2[fastOnes], label="fast")
#     _plt.legend()

fig = _plt.figure(figsize=(6, 2.5))
si   = 0
nknm = ["WIN", "TIE", "LOS", "STAY", "DOWN", "UP"]
for smet in ["entropyT", "entropyW", "entropyL", "entropyS", "entropyD", "entropyU"]:
    si += 1
    met = eval(smet)
    metmax = _N.max([_N.max(met[fastOnes]), _N.max(met[slowOnes])])
    metmin = _N.min([_N.min(met[fastOnes]), _N.min(met[slowOnes])])
    metF   = (met[fastOnes] - metmin) / (metmax - metmin)
    metS   = (met[slowOnes] - metmin) / (metmax - metmin)
    metF.sort()
    metS.sort()    
    metF10 = metF[int(metF.shape[0]*0.1)]
    metF90 = metF[int(metF.shape[0]*0.9)]
    metS10 = metS[int(metS.shape[0]*0.1)]
    metS90 = metS[int(metS.shape[0]*0.9)]    
    _plt.plot([metF10, metF90], [si-0.2, si-0.2], color="blue", marker="|", lw=5, ms=10)
    _plt.plot([metS10, metS90], [si, si], color="red", marker="|", lw=5, ms=10)

_plt.ylim(0.2, 6.8)
_plt.yticks(_N.arange(1, 7), nknm, fontsize=17)
_plt.xlabel("entropy (AU)", fontsize=18)
_plt.xticks([0, 0.5, 1], fontsize=17)
fig.subplots_adjust(bottom=0.31, left=0.2, right=0.97)
_plt.savefig("entropy_slow_fast", transparent=True)
