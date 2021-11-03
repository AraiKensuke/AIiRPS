#!/usr/bin/python

from sklearn import linear_model
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

__1st__ = 0
__2nd__ = 1
__ALL__ = 2

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

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

def entropy3(sig, N):
    cube = _N.zeros((N, N, N))
    iN   = 1./N
    for i in range(len(sig)):
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

def deriv_CRs(_hnd_dat, _prob_mvs, win):
    #####################################
    prob_mvs  = _prob_mvs[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size
    #print(prob_mvs.shape)
    prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
    ab_d_prob_mvs = _N.abs(_N.diff(prob_mvs, axis=2))
    behv = _N.sum(_N.sum(ab_d_prob_mvs, axis=1), axis=0)
    _dbehv = _N.diff(behv)
    dbehv = _N.convolve(_dbehv, gk, mode="same")
    return dbehv, prob_mvs

def parse_CR_states(dbehv, prob_mvs, TO):
    maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + 2 #  3 from label71
    isi   = _N.diff(maxs)

    prob_pcs = _N.empty((len(maxs)-1, 3, 3))
    for i in range(len(maxs)-1):
        prob_pcs[i] = _N.mean(prob_mvs[:, :, maxs[i]:maxs[i+1]], axis=2)
        #  _N.sum(prob_mvs[:, :, 10], axis=1) == [1, 1, 1]
    prob_SW  = _N.sum(prob_pcs[:, :, _N.array([0, 2])], axis=2)
    return prob_pcs, isi

def wnd_mean(x, y, winsz=4):
    ix = x.argsort()

    xwm = _N.empty(x.shape[0] - (winsz-1))  #  5 - (3-1)
    ywm = _N.empty(x.shape[0] - (winsz-1))  #  5 - (3-1)
    
    for i in range(x.shape[0] - (winsz-1)):
        xwm[i] = _N.mean(x[ix[i:i+winsz]])
        ywm[i] = _N.mean(y[ix[i:i+winsz]])
    return xwm, ywm

def interiorCC(x, y):
    ix = x.argsort()
    iy = y.argsort()    

    ths = _N.array([ix[0], ix[-1], iy[0], iy[-1]])
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
    while x[ix[i+1]] - x[ix[i]] > 0.3*x_std:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > 0.3*x_std:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > 0.3*y_std:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > 0.3*y_std:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    print("%(ths)d" % {"ths" : len(ths)})
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


gk_fot = gauKer(1)
gk_fot = gk_fot / _N.sum(gk_fot)

look_at_AQ = True
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
    #dates = _rt.date_range(start='7/13/2021', end='08/20/2021')
    dates = _rt.date_range(start='7/13/2021', end='10/30/2021')
    #dates = _rt.date_range(start='07/27/2021', end='08/20/2021')
    #dates = _rt.date_range(start='07/13/2021', end='07/27/2021')
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=800, MinWinLossRat=0.7, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=800, MinWinLossRat=0.7, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1, 2], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=100, maxIGI=1000000, MinWinLossRat=0.3, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

A1 = []
show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

MULT   = 1.   #  kernel width
fxdGK  = None
lags_sec=50
win     = 4
smth    = 3 
label          = win*10+smth

#fig= _plt.figure(figsize=(14, 14))

pid = 0

SHUFFLES = 1000

t0 = -5
t1 = 10
cut = 1
all_avgs = _N.empty((len(partIDs), t1-t0))
netwins  = _N.empty((len(partIDs), 2), dtype=_N.int)
gk = gauKer(1)
gk /= _N.sum(gk)

CR_avgs_all = []
CR_avgs_sing = []
CR_avgs_all_2 = []
CR_avgs_sing_2 = []

corrs_all = _N.empty((3, 6))
corrs_sing = _N.empty((len(partIDs), 3, 6))

perform   = _N.empty(len(partIDs))

pid = 0

ts  = _N.arange(t0-2, t1-2)
signal_5_95 = _N.empty((len(partIDs), t1-t0, 2))

win_aft_tie = _N.empty((len(partIDs), 2))
tie_aft_tie = _N.empty((len(partIDs), 2))
los_aft_tie = _N.empty((len(partIDs), 2))
win_aft_win = _N.empty((len(partIDs), 2))
tie_aft_win = _N.empty((len(partIDs), 2))
los_aft_win = _N.empty((len(partIDs), 2))
win_aft_los = _N.empty((len(partIDs), 2))
tie_aft_los = _N.empty((len(partIDs), 2))
los_aft_los = _N.empty((len(partIDs), 2))

pc_sum = _N.empty(len(partIDs))
moresim = _N.empty((len(partIDs), 2))
moresimUD = _N.empty(len(partIDs))
moresimST = _N.empty(len(partIDs))
moresimSTWL = _N.empty(len(partIDs))
moresimSW = _N.empty(len(partIDs))
pc_sum2 = _N.empty(len(partIDs))
isis    = _N.empty((len(partIDs), 2))
isis_sd    = _N.empty(len(partIDs))
isis_cv    = _N.empty((len(partIDs), 2))
isis_corr    = _N.empty((len(partIDs), 2))
isis_lv    = _N.empty(len(partIDs))

isisW    = _N.empty(len(partIDs))
isisW_sd    = _N.empty(len(partIDs))
isisL    = _N.empty(len(partIDs))
isisL_sd    = _N.empty(len(partIDs))

pfrm_change36 = _N.empty((len(partIDs), 2))
pfrm_change69 = _N.empty((len(partIDs), 2))
pfrm_change912= _N.empty((len(partIDs), 2))

score  = _N.empty(len(partIDs))
sum_cv = _N.empty((len(partIDs), 2, 3, 3))
sum_sd = _N.empty((len(partIDs), 2, 3, 3))
rat_sd = _N.empty(len(partIDs))
diff_sd = _N.empty(len(partIDs))
entropy = _N.empty(len(partIDs))
entropyD = _N.empty((len(partIDs), 2))
entropyS = _N.empty((len(partIDs), 2))
entropyU = _N.empty((len(partIDs), 2))
entropyW2 = _N.empty((len(partIDs), 2))
entropyT2 = _N.empty((len(partIDs), 2))
entropyL2 = _N.empty((len(partIDs), 2))

entropySW = _N.empty(len(partIDs))

AQ28scrs  = _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))
times     = _N.empty((len(partIDs), 2))

t0 = -5
t1 = 10

all_avgs = _N.empty((len(partIDs), t1-t0, 2))

all_maxs  = []

aboves = []
belows = []

all_ents_wtl = _N.empty((len(partIDs), 2, 3))
all_ents_dsu = _N.empty((len(partIDs), 2, 3))
    
for partID in partIDs:
    pid += 1

    __hnd_dat1, start_time1, end_time1, UA1, cnstr1            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB2")
    __hnd_dat2, start_time2, end_time2, UA2, cnstr2            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=2, expt="TMB2")
        
    TO         = 300
    _hnd_dat1   = __hnd_dat1[0:TO]
    _hnd_dat2   = __hnd_dat2[0:TO]

    times[pid-1, 0] = __hnd_dat1[-1, 3] - __hnd_dat1[0, 3]
    times[pid-1, 1] = __hnd_dat2[-1, 3] - __hnd_dat2[0, 3]    
    inds =_N.arange(_hnd_dat1.shape[0])
    dmp1       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_1.dmp" % {"rpsm" : partID, "lb" : label}))
    dmp2       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_2.dmp" % {"rpsm" : partID, "lb" : label}))    
    _prob_mvs1 = dmp1["cond_probs"][0]
    _prob_mvs2 = dmp2["cond_probs"][0]
    
    # 

    ties1 = _N.where(_hnd_dat1[0:TO-2, 2] == 0)[0]
    tw1   = _N.where(_hnd_dat1[ties1+1, 2] == 1)[0]
    tt1   = _N.where(_hnd_dat1[ties1+1, 2] == 0)[0]
    tl1   = _N.where(_hnd_dat1[ties1+1, 2] == -1)[0]    
    win_aft_tie[pid-1, 0] = len(tw1) / len(ties1)
    tie_aft_tie[pid-1, 0] = len(tt1) / len(ties1)
    los_aft_tie[pid-1, 0] = len(tl1) / len(ties1)    
    ties2 = _N.where(_hnd_dat2[0:TO-2, 2] == 0)[0]
    tw2   = _N.where(_hnd_dat2[ties2+1, 2] == 1)[0]
    tt2   = _N.where(_hnd_dat2[ties2+1, 2] == 0)[0]
    tl2   = _N.where(_hnd_dat2[ties2+1, 2] == -1)[0]    
    win_aft_tie[pid-1, 1] = len(tw2) / len(ties2)
    tie_aft_tie[pid-1, 1] = len(tt2) / len(ties2)
    los_aft_tie[pid-1, 1] = len(tl2) / len(ties2)    

    los1 = _N.where(_hnd_dat1[0:TO-2, 2] == -1)[0]
    lw1   = _N.where(_hnd_dat1[los1+1, 2] == 1)[0]
    lt1   = _N.where(_hnd_dat1[los1+1, 2] == 0)[0]
    ll1   = _N.where(_hnd_dat1[los1+1, 2] == -1)[0]    
    win_aft_los[pid-1, 0] = len(lw1) / len(los1)
    tie_aft_los[pid-1, 0] = len(lt1) / len(los1)
    los_aft_los[pid-1, 0] = len(ll1) / len(los1)    
    los2 = _N.where(_hnd_dat2[0:TO-2, 2] == -1)[0]
    lw2   = _N.where(_hnd_dat2[los2+1, 2] == 1)[0]
    lt2   = _N.where(_hnd_dat2[los2+1, 2] == 0)[0]
    ll2   = _N.where(_hnd_dat2[los2+1, 2] == -1)[0]    
    win_aft_los[pid-1, 1] = len(lw2) / len(los2)
    tie_aft_los[pid-1, 1] = len(lt2) / len(los2)
    los_aft_los[pid-1, 1] = len(ll2) / len(los2)    

    win1 = _N.where(_hnd_dat1[0:TO-2, 2] == 1)[0]
    ww1   = _N.where(_hnd_dat1[win1+1, 2] == 1)[0]
    wt1   = _N.where(_hnd_dat1[win1+1, 2] == 0)[0]
    wl1   = _N.where(_hnd_dat1[win1+1, 2] == -1)[0]    
    win_aft_win[pid-1, 0] = len(ww1) / len(win1)
    tie_aft_win[pid-1, 0] = len(wt1) / len(win1)
    los_aft_win[pid-1, 0] = len(wl1) / len(win1)    
    win2 = _N.where(_hnd_dat2[0:TO-2, 2] == 1)[0]
    ww2   = _N.where(_hnd_dat2[win2+1, 2] == 1)[0]
    wt2   = _N.where(_hnd_dat2[win2+1, 2] == 0)[0]
    wl2   = _N.where(_hnd_dat2[win2+1, 2] == -1)[0]    
    win_aft_win[pid-1, 1] = len(ww2) / len(win2)
    tie_aft_win[pid-1, 1] = len(wt2) / len(win2)
    los_aft_win[pid-1, 1] = len(wl2) / len(win2)    

    netwins[pid-1, 0] = _N.sum(_hnd_dat1[:, 2])
    netwins[pid-1, 1] = _N.sum(_hnd_dat2[:, 2])    
    #####################################
    dbehv1, prob_mvs1 = deriv_CRs(_hnd_dat1, _prob_mvs1, win)
    dbehv2, prob_mvs2 = deriv_CRs(_hnd_dat2, _prob_mvs2, win)

    prob_pcs1, isis1 = parse_CR_states(dbehv1, prob_mvs1, TO)
    prob_pcs2, isis2 = parse_CR_states(dbehv2, prob_mvs2, TO)

    #####################################
    pcWT_D_1, pvWT_D_1 = _ss.pearsonr(prob_mvs1[0, 0], prob_mvs1[1, 0])    
    pcWT_S_1, pvWT_S_1 = _ss.pearsonr(prob_mvs1[0, 1], prob_mvs1[1, 1])
    pcWT_U_1, pvWT_U_1 = _ss.pearsonr(prob_mvs1[0, 2], prob_mvs1[1, 2])        

    pcTL_D_1, pvTL_D_1 = _ss.pearsonr(prob_mvs1[1, 0], prob_mvs1[2, 0])
    pcTL_S_1, pvTL_S_1 = _ss.pearsonr(prob_mvs1[1, 1], prob_mvs1[2, 1])
    pcTL_U_1, pvTL_U_1 = _ss.pearsonr(prob_mvs1[1, 2], prob_mvs1[2, 2])

    pcWT_D_2, pvWT_D_2 = _ss.pearsonr(prob_mvs2[0, 0], prob_mvs2[1, 0])    
    pcWT_S_2, pvWT_S_2 = _ss.pearsonr(prob_mvs2[0, 1], prob_mvs2[1, 1])
    pcWT_U_2, pvWT_U_2 = _ss.pearsonr(prob_mvs2[0, 2], prob_mvs2[1, 2])        

    pcTL_D_2, pvTL_D_2 = _ss.pearsonr(prob_mvs2[1, 0], prob_mvs2[2, 0])
    pcTL_S_2, pvTL_S_2 = _ss.pearsonr(prob_mvs2[1, 1], prob_mvs2[2, 1])
    pcTL_U_2, pvTL_U_2 = _ss.pearsonr(prob_mvs2[1, 2], prob_mvs2[2, 2])

    moresim[pid-1, 0] = pcTL_D_1+pcTL_S_1+pcTL_U_1 - (pcWT_D_1+pcWT_S_1+pcWT_U_1)
    moresim[pid-1, 1] = pcTL_D_2+pcTL_S_2+pcTL_U_2 - (pcWT_D_2+pcWT_S_2+pcWT_U_2)    

    PCS = 5
    ents_wtl1 = _N.array([entropy3(prob_mvs1[:, 0].T, PCS), entropy3(prob_mvs1[:, 1].T, PCS), entropy3(prob_mvs1[:, 2].T, PCS)])
    ents_wtl2 = _N.array([entropy3(prob_mvs2[:, 0].T, PCS), entropy3(prob_mvs2[:, 1].T, PCS), entropy3(prob_mvs2[:, 2].T, PCS)])        
    ents_dsu1 = _N.array([entropy3(prob_mvs1[0].T, PCS), entropy3(prob_mvs1[1].T, PCS), entropy3(prob_mvs1[2].T, PCS)])
    ents_dsu2 = _N.array([entropy3(prob_mvs2[0].T, PCS), entropy3(prob_mvs2[1].T, PCS), entropy3(prob_mvs2[2].T, PCS)])        

    maxs1 = _N.where((dbehv1[0:TO-11] >= 0) & (dbehv1[1:TO-10] < 0))[0] + (win//2) #  3 from label71
    maxs2 = _N.where((dbehv2[0:TO-11] >= 0) & (dbehv2[1:TO-10] < 0))[0] + (win//2) #  3 from label71    

    avgs1 = _N.empty((len(maxs1)-2*cut, t1-t0))
    avgs2 = _N.empty((len(maxs2)-2*cut, t1-t0))    
    for im in range(cut, len(maxs1)-cut):
        #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2].shape)
        #print("%(1)d %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
        avgs1[im-1, :] = _hnd_dat1[maxs1[im]+t0:maxs1[im]+t1, 2]
    for im in range(cut, len(maxs2)-cut):
        #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2].shape)
        #print("%(1)d %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
        avgs2[im-1, :] = _hnd_dat2[maxs2[im]+t0:maxs2[im]+t1, 2]
    all_avgs[pid-1, :, 0] = _N.mean(avgs1, axis=0)
    all_avgs[pid-1, :, 1] = _N.mean(avgs2, axis=0)    
    
    
    all_ents_wtl[pid-1, 0] = ents_wtl1  #  ENTROPY for fixed action over different conditions)
    all_ents_wtl[pid-1, 1] = ents_wtl2
    all_ents_dsu[pid-1, 0] = ents_dsu1
    all_ents_dsu[pid-1, 1] = ents_dsu2

    isis_sd1 = _N.std(isis1)
    isis_sd2 = _N.std(isis2)    
    isis_cv[pid-1, 0] = isis_sd1 / _N.mean(isis1)
    isis_cv[pid-1, 1] = isis_sd2 / _N.mean(isis2)

    pW_stsw1 = _N.array([prob_mvs1[0, 0] + prob_mvs1[0, 2], prob_mvs1[0, 1]])
    pT_stsw1 = _N.array([prob_mvs1[1, 0] + prob_mvs1[1, 2], prob_mvs1[1, 1]])
    pL_stsw1 = _N.array([prob_mvs1[2, 0] + prob_mvs1[2, 2], prob_mvs1[2, 1]])
    pW_stsw2 = _N.array([prob_mvs2[0, 0] + prob_mvs2[0, 2], prob_mvs2[0, 1]])
    pT_stsw2 = _N.array([prob_mvs2[1, 0] + prob_mvs2[1, 2], prob_mvs2[1, 1]])
    pL_stsw2 = _N.array([prob_mvs2[2, 0] + prob_mvs2[2, 2], prob_mvs2[2, 1]])

    entsWTL1_2 = _N.array([entropy2(pW_stsw1.T, PCS), entropy2(pT_stsw1.T, PCS), entropy2(pL_stsw1.T, PCS)])
    entsWTL2_2 = _N.array([entropy2(pW_stsw2.T, PCS), entropy2(pT_stsw2.T, PCS), entropy2(pL_stsw2.T, PCS)])
    entsDSU1 = _N.array([entropy3(prob_mvs1[:, 0].T, PCS),
                         entropy3(prob_mvs1[:, 1].T, PCS),
                         entropy3(prob_mvs1[:, 2].T, PCS)])
    entsDSU2 = _N.array([entropy3(prob_mvs2[:, 0].T, PCS),
                         entropy3(prob_mvs2[:, 1].T, PCS),
                         entropy3(prob_mvs2[:, 2].T, PCS)])
    entropyU[pid-1, 0] = entsDSU1[2]
    entropyU[pid-1, 0] = entsDSU1[2]
    entropyU[pid-1, 0] = entsDSU1[2]    
    entropyU[pid-1, 1] = entsDSU2[2]
    entropyU[pid-1, 1] = entsDSU2[2]
    entropyU[pid-1, 1] = entsDSU2[2]
    ################################
    entropyD[pid-1, 0] = entsDSU1[0]
    entropyD[pid-1, 0] = entsDSU1[0]
    entropyD[pid-1, 0] = entsDSU1[0]    
    entropyD[pid-1, 1] = entsDSU2[0]
    entropyD[pid-1, 1] = entsDSU2[0]
    entropyD[pid-1, 1] = entsDSU2[0]
    ################################    
    entropyS[pid-1, 0] = entsDSU1[1]
    entropyS[pid-1, 0] = entsDSU1[1]
    entropyS[pid-1, 0] = entsDSU1[1]    
    entropyS[pid-1, 1] = entsDSU2[1]
    entropyS[pid-1, 1] = entsDSU2[1]
    entropyS[pid-1, 1] = entsDSU2[1]
    

    entropyW2[pid-1, 0] = entsWTL1_2[0]
    entropyW2[pid-1, 1] = entsWTL2_2[0]    
    entropyW2[pid-1, 0] = entsWTL1_2[0]
    entropyT2[pid-1, 1] = entsWTL2_2[1]    
    entropyT2[pid-1, 0] = entsWTL1_2[1]
    entropyT2[pid-1, 1] = entsWTL2_2[1]    
    entropyL2[pid-1, 1] = entsWTL2_2[2]    
    entropyL2[pid-1, 0] = entsWTL1_2[2]
    entropyL2[pid-1, 1] = entsWTL2_2[2]    
    
    sds1 = _N.std(prob_mvs1, axis=2)
    mns1 = _N.mean(prob_mvs1, axis=2)
    sds2 = _N.std(prob_mvs2, axis=2)
    mns2 = _N.mean(prob_mvs2, axis=2)    

    isis[pid-1, 0] = _N.mean(isis1)
    isis[pid-1, 1] = _N.mean(isis2)    
    pc1, pv1 = rm_outliersCC_neighbors(isis1[0:-1], isis1[1:])
    pc2, pv2 = rm_outliersCC_neighbors(isis2[0:-1], isis2[1:])    
    isis_corr[pid-1, 0] = pc1
    isis_corr[pid-1, 1] = pc2
    
    sum_cv[pid-1, 0] = sds1/mns1
    sum_cv[pid-1, 1] = sds2/mns2
    sum_sd[pid-1, 0] = sds1
    sum_sd[pid-1, 1] = sds2

    signal_5_95[pid-1] = all_avgs[pid-1]


    sInds1 = _N.argsort(signal_5_95[pid-1, 3:6, 0])
    sInds2 = _N.argsort(signal_5_95[pid-1, 3:6, 1])

    if sInds1[2] - sInds1[0] > 0:
        m36_1 = 1
    else:
        m36_1 = -1
    if sInds2[2] - sInds2[0] > 0:
        m36_2 = 1
    else:
        m36_2 = -1
        
    sInds1 = _N.argsort(signal_5_95[pid-1, 6:9, 0])
    sInds2 = _N.argsort(signal_5_95[pid-1, 6:9, 1])    
    if sInds1[2] - sInds1[0] > 0:
        m69_1 = 1
    else:
        m69_1 = -1
    if sInds2[2] - sInds2[0] > 0:
        m69_2 = 1
    else:
        m69_2 = -1
        
    sInds1 = _N.argsort(signal_5_95[pid-1, 9:12, 0])
    sInds2 = _N.argsort(signal_5_95[pid-1, 9:12, 1])    
    if sInds1[2] - sInds1[0] > 0:
        m912_1 = 1
    else:
        m912_1 = -1
    if sInds2[2] - sInds2[0] > 0:
        m912_2 = 1
    else:
        m912_2 = -1

    imax36_1 = _N.argmax(signal_5_95[pid-1, 3:6, 0])+3
    imin36_1 = _N.argmin(signal_5_95[pid-1, 3:6, 0])+3
    imax69_1 = _N.argmax(signal_5_95[pid-1, 6:9, 0])+6
    imin69_1 = _N.argmin(signal_5_95[pid-1, 6:9, 0])+6    
    imax912_1= _N.argmax(signal_5_95[pid-1, 9:12, 0])+9
    imin912_1= _N.argmin(signal_5_95[pid-1, 9:12, 0])+9
    imax36_2 = _N.argmax(signal_5_95[pid-1, 3:6, 1])+3
    imin36_2 = _N.argmin(signal_5_95[pid-1, 3:6, 1])+3
    imax69_2 = _N.argmax(signal_5_95[pid-1, 6:9, 1])+6
    imin69_2 = _N.argmin(signal_5_95[pid-1, 6:9, 1])+6    
    imax912_2= _N.argmax(signal_5_95[pid-1, 9:12, 1])+9
    imin912_2= _N.argmin(signal_5_95[pid-1, 9:12, 1])+9    
    
    """
    imax_imin_pfrm36[pid-1, 0] = imin36_1
    imax_imin_pfrm36[pid-1, 1] = imax36_1
    imax_imin_pfrm69[pid-1, 0] = imin69_1
    imax_imin_pfrm69[pid-1, 1] = imax69_1
    imax_imin_pfrm912[pid-1, 0]= imin912_1
    imax_imin_pfrm912[pid-1, 1]= imax912_1
    imax_imin_pfrm36[pid-1, 0] = imin36_2
    imax_imin_pfrm36[pid-1, 1] = imax36_2
    imax_imin_pfrm69[pid-1, 0] = imin69_2
    imax_imin_pfrm69[pid-1, 1] = imax69_2
    imax_imin_pfrm912[pid-1, 0]= imin912_2
    imax_imin_pfrm912[pid-1, 1]= imax912_2
    """    
    
    pfrm_change36[pid-1, 0] = signal_5_95[pid-1, imax36_1, 0] - signal_5_95[pid-1, imin36_1, 0]
    pfrm_change69[pid-1, 0] = signal_5_95[pid-1, imax69_1, 0] - signal_5_95[pid-1, imin69_1, 0]
    pfrm_change912[pid-1, 0]= signal_5_95[pid-1, imax912_1, 0] - signal_5_95[pid-1, imin912_1, 0]
    pfrm_change36[pid-1, 1] = signal_5_95[pid-1, imax36_2, 1] - signal_5_95[pid-1, imin36_2, 1]
    pfrm_change69[pid-1, 1] = signal_5_95[pid-1, imax69_2, 1] - signal_5_95[pid-1, imin69_2, 1]
    pfrm_change912[pid-1, 1]= signal_5_95[pid-1, imax912_2, 1] - signal_5_95[pid-1, imin912_2, 1]


entropyUD = _N.exp((entropyD + entropyU) / (_N.mean(entropyD + entropyU)))
entropyT2 = _N.exp(entropyT2 / _N.mean(entropyT2*0.2)) / 10
entropyW2 = _N.exp(entropyW2 / _N.mean(entropyW2*0.3)) / 10
entropyL2 = _N.exp(entropyL2 / _N.mean(entropyL2*0.3)) / 10

markers = [isis, isis_cv, isis_corr, pfrm_change69, # 3.5# 2.5
           entropyL2, entropyW2, entropyT2, entropyU, entropyD, entropyUD, entropyS, # 10.5

           netwins, win_aft_tie, tie_aft_tie, los_aft_tie, win_aft_win, tie_aft_win, los_aft_win, win_aft_los, tie_aft_los, los_aft_los]

im = 0
fig = _plt.figure(figsize=(11, 11))
pcs = _N.empty(len(markers))
for marker in markers:
    im += 1
    fig.add_subplot(5, 5, im)
    _plt.scatter(marker[:, 0], marker[:, 1], color="black")
    pc, pv = _ss.pearsonr(marker[:, 0], marker[:, 1])
    pcs[im-1] = pc
    minM = _N.min(marker)
    maxM = _N.max(marker)
    A = maxM - minM
    _plt.plot([minM - 0.1*A, maxM + 0.1*A], [minM - 0.1*A, maxM + 0.1*A])
    _plt.title("%.3f" % pc)

fig.subplots_adjust(hspace=0.5)

fig = _plt.figure(figsize=(7, 3))
_plt.scatter(_N.arange(len(pcs)), pcs, color="black", marker=".", s=80)
_plt.ylim(-1, 1)
#_plt.axvline(x=3.5, color="grey", ls=":")
#_plt.axvline(x=10.5, color="grey", ls=":")
_plt.axhline(y=0, ls="--", color="grey")
_plt.xlabel("feature #", fontsize=27)
_plt.xticks(_N.arange(0, 24, 5), fontsize=25)
_plt.ylabel(r"$r$ round 1 & 2", fontsize=27)
_plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=25)
fig.subplots_adjust(left=0.25, bottom=0.27)
_plt.savefig("retest", transparent=True)


# pcL2 = _ss.pearsonr(
# for i in range(len(partIDs)):
#     _plt.plot([i, i], [mn_ents_wtl[i, 0], mn_ents_wtl[i, 1]], marker=".", ms=10)

# for i in range(len(partIDs)):
#     _plt.plot([i, i], [isis_corr[i, 0], isis_corr[i, 1]], marker=".", ms=10)

# for i in range(len(partIDs)):
#     _plt.plot([i, i], [netwins[i, 0], netwins[i, 1]], marker=".", ms=10)

# for i in range(len(partIDs)):
#     _plt.plot([i, i], [win_aft_tie[i, 0], win_aft_tie[i, 1]], marker=".", ms=10)

# for i in range(len(partIDs)):
#     _plt.plot([i, i], [tie_aft_tie[i, 0], tie_aft_tie[i, 1]], marker=".", ms=10)

# for i in range(len(partIDs)):
#     _plt.plot([i, i], [los_aft_tie[i, 0], los_aft_tie[i, 1]], marker=".", ms=10)
    
