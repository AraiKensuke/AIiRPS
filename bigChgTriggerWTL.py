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
import AIiRPS.models.CRutils as _emp
import GCoh.eeg_util as _eu

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
            
def entropy3(sig, N):
    cube = _N.zeros((N, N, N))   #  W T L conditions or
    iN   = 1./N

    #print(sig.shape[0])
    for i in range(sig.shape[0]):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        iz = int(sig[i, 2]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        iz = iz if iz < N else N-1
        cube[ix, iy, iz] += 1

    #print(cube)
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
    while x[ix[i+1]] - x[ix[i]] > x_std:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > x_std:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > y_std:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > y_std:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    #print("%(ths)d" % {"ths" : len(ths)})
    return _ss.pearsonr(x[interiorPts], y[interiorPts])

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
    dates = _rt.date_range(start='7/13/2021', end='08/30/2021')
    #dates = _rt.date_range(start='7/13/2021', end='07/27/2021')
    #dates = _rt.date_range(start='7/27/2021', end='08/20/2021')
    #dates = _rt.date_range(start='07/27/2021', end='08/20/2021')
    #dates = _rt.date_range(start='07/13/2021', end='07/27/2021')
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=800, MinWinLossRat=0.7, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=800, MinWinLossRat=0.7, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=650, maxIGI=30000, MinWinLossRat=0.3, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

A1 = []
show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

MULT   = 1.   #  kernel width
fxdGK  = None

win     = 4
smth    = 3
label          = win*10+smth

#fig= _plt.figure(figsize=(14, 14))

pid = 0

SHUFFLES = 1000

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
isis_lv    = _N.empty(len(partIDs))
isis_cv    = _N.empty(len(partIDs))
isis_corr    = _N.empty(len(partIDs))
coherence    = _N.empty(len(partIDs))

corr_UD    = _N.empty((len(partIDs), 3))

score  = _N.empty(len(partIDs))
moresim  = _N.empty(len(partIDs))
sum_sd = _N.empty((len(partIDs), 3, 3))
sum_cv = _N.empty((len(partIDs), 3, 3))
entropyDSU = _N.empty((len(partIDs), 3))
entropyWTL = _N.empty((len(partIDs), 3))
entropyD = _N.empty(len(partIDs))
entropyS = _N.empty(len(partIDs))
entropyU = _N.empty(len(partIDs))
entropyW = _N.empty(len(partIDs))
entropyT = _N.empty(len(partIDs))
entropyL = _N.empty(len(partIDs))

win_aft_win  = _N.empty(len(partIDs))
win_aft_los  = _N.empty(len(partIDs))

AQ28scrs  = _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))    

all_maxs  = []

aboves = []
belows = []

all_prob_mvs = []

for partID in partIDs:
    pid += 1

    if data == "TMB2":
        __hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB2")
    if data == "EEG1":
        __hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=None, expt="EEG1")
    if data == "RAND":
        __hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="RAND")
        
    TO         = 300
    _hnd_dat   = __hnd_dat[0:TO]
    inds =_N.arange(_hnd_dat.shape[0])

    wins = _N.where(_hnd_dat[0:TO-2, 2] == 1)[0]
    ww   = _N.where(_hnd_dat[wins+1, 2] == 1)[0]
    win_aft_win[pid-1] = len(ww) / len(wins)
    loses = _N.where(_hnd_dat[0:TO-2, 2] == -1)[0]
    lw   = _N.where(_hnd_dat[loses+1, 2] == 1)[0]
    win_aft_los[pid-1] = len(lw) / len(loses)
    
    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_1.dmp" % {"rpsm" : partID, "lb" : label}))
    _prob_mvs = dmp["cond_probs"][0]
    prob_mvs  = _prob_mvs[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size
    prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
    dbehv = _emp.get_dbehv(prob_mvs, gk)
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

    PCS=8
    #  prob_mvs[:, 0] - for each time point, the DOWN probabilities following 3 different conditions
    #  prob_mvs[0]    - for each time point, the DOWN probabilities following 3 different conditions    

    entsDSU = _N.array([entropy3(prob_mvs[:, 0].T, PCS), entropy3(prob_mvs[:, 1].T, PCS), entropy3(prob_mvs[:, 2].T, PCS)])    
    entropyD[pid-1] = entsDSU[0]
    entropyS[pid-1] = entsDSU[1]
    entropyU[pid-1] = entsDSU[2]

    THRisi = 2
    isi   = cleanISI(_N.diff(maxs), minISI=2)

    # largeEnough = _N.where(_isi > THRisi)[0]
    # tooSmall    = _N.where(_isi <= 3)[0]
    # isi    = _isi[largeEnough]
    pc, pv = _ss.pearsonr(isi[0:-1], isi[1:])
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
    
    if look_at_AQ:
        AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})

    netwins[pid-1] = _N.sum(_hnd_dat[:, 2])
    wins = _N.where(_hnd_dat[:, 2] == 1)[0]
    losses = _N.where(_hnd_dat[:, 2] == -1)[0]
    perform[pid -1] = len(wins) / (len(wins) + len(losses))
    
    for sh in range(SHUFFLES+1):
        if sh > 0:
            _N.random.shuffle(inds)
        hnd_dat = _hnd_dat[inds]

        avgs = _N.empty((len(maxs)-2*cut, t1-t0))
        for im in range(cut, len(maxs)-cut):
            #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2].shape)
            #print("%(1)d %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
            avgs[im-1, :] = hnd_dat[maxs[im]+t0:maxs[im]+t1, 2]

        all_avgs[pid-1, sh] = _N.mean(avgs, axis=0)
        #fig.add_subplot(5, 5, pid)
        #_plt.plot(_N.mean(avgs, axis=0))

    srtd   = _N.sort(all_avgs[pid-1, 1:], axis=0)
    signal_5_95[pid-1, 1] = srtd[int(0.05*SHUFFLES)]
    signal_5_95[pid-1, 2] = srtd[int(0.95*SHUFFLES)]
    signal_5_95[pid-1, 0] = all_avgs[pid-1, 0]
    signal_5_95[pid-1, 3] = (signal_5_95[pid-1, 0] - signal_5_95[pid-1, 1]) / (signal_5_95[pid-1, 2] - signal_5_95[pid-1, 1])

    #fig.add_subplot(6, 6, pid)

    # _plt.title(netwins[pid-1] )
    # _plt.plot(ts, signal_5_95[pid-1, 0], marker=".", ms=10)
    # _plt.plot(ts, signal_5_95[pid-1, 1])
    # _plt.plot(ts, signal_5_95[pid-1, 2])
    # _plt.axvline(x=-0.5, ls="--")

    be = _N.where(signal_5_95[pid-1, 0] < signal_5_95[pid-1, 1])[0]
    if len(be) > 0:
        belows.extend(be)
    ab = _N.where(signal_5_95[pid-1, 0] > signal_5_95[pid-1, 2])[0]        
    if len(ab) > 0:
        aboves.extend(ab)

pc_sum = pc_sum01 + pc_sum02 + pc_sum12
        
###############  sort by entropyU.  Show 
all_prob_mvsA = _N.array(all_prob_mvs)
sinds = entropyU.argsort()
min_prob_mvs = all_prob_mvsA[sinds[0]]
max_prob_mvs = all_prob_mvsA[sinds[-1]]

fig = _plt.figure(figsize=(10, 6))
fig.add_subplot(2, 1, 1)
_plt.plot(min_prob_mvs[2, 2], label=r"$p_k($UP$|$L$)$")
_plt.plot(min_prob_mvs[1, 2], label=r"$p_k($UP$|$T$)$")
_plt.plot(min_prob_mvs[0, 2], label=r"$p_k($UP$|$W$)$")
_plt.legend()
fig.add_subplot(2, 1, 2)
_plt.plot(max_prob_mvs[2, 2], label=r"$p_k($UP$|$L$)$")
_plt.plot(max_prob_mvs[1, 2], label=r"$p_k($UP$|$T$)$")
_plt.plot(max_prob_mvs[0, 2], label=r"$p_k($UP$|$W$)$")
_plt.legend()
_plt.xlabel(r"Game # $k$")
_plt.savefig("EntropyU_prob_mvs_%(lb)d" % {"lb" : label})

###############  show me inter-event intervals

for by in ["isis_cv", "isis_corr"]:
    by_m = isis_cv.argsort() if by == "isis_cv" else isis_corr.argsort()
    fig = _plt.figure(figsize=(10, 3))    
    for ii in range(2):
        intvs = all_maxs[by_m[0]] if ii == 0 else all_maxs[by_m[-2]]
        fig.add_subplot(2, 1, ii+1)
        t = 0
        for i in range(intvs.shape[0]):
            _plt.plot([t, t], [0, 1], color="black")
            t += intvs[i]
        if ii == 1:
            _plt.xlabel(r"Game # $k$")
    _plt.suptitle("Rule change timing (at game)")
    fig.subplots_adjust(bottom=0.15)
    _plt.savefig("%(by)s_big_small_%(lb)d" % {"by" : by, "lb" : label})
    
fig = _plt.figure()
_plt.scatter(pc_sum, score)
pc, pv = _ss.pearsonr(pc_sum, score)
_plt.xlabel("CC[p(ST|W),p(ST|T)] + CC[p(ST|W),p(ST|L)] + CC[p(ST|T),p(ST|L)]")
_plt.xlim(-5, 5)
_plt.ylabel("Net win")
_plt.suptitle("pc: %(pc).2f  %(pv).2e    pcstd %(std).2f" % {"pc" : pc, "pv" : pv, "std" : _N.std(pc_sum)})
_plt.savefig("corr_w_sum_stay_CRs_%(fn)s_%(lb)d.png" % {"fn" : data, "lb" : label})
_plt.close()


#  More sim:  If large
#cmp_vs = pc_sum
#cmp_against = "netwins"
#cmp_againsts = ["pc_sum", "netwins", "isis_cv", "rat_sd", "more_sim", "sum_sd", "sum_sdW", ]
#cmp_againsts = ["entropyW", "entropyL", "entropyT", "entropyD", "entropyS", "entropyU"]
#cmp_againsts = ["isis_cv", "isis_corr"]

#cmp_againsts = ["entropyL"]
cmp_againsts = ["entropyUD", "isis_cv", "isis_corr", "entropyD", "entropyS", "entropyU", "sum_cv", "netwins", "isis_lv"]#, "win_aft_win", "win_aft_los"]
#cmp_againsts = ["win_aft_win", "win_aft_los"]
#cmp_againsts = ["entropyWTL"]
#cmp_againsts = ["moresim"]
#cmp_againsts = ["entropyU"]

#cmp_againsts = ["sum_sd"]
#cmp_againsts = ["isis_corr"]
#cmp_againsts = ["sum_cv"]
#cmp_againsts = ["sum_sd"]
#cmp_againsts = ["entropyWTL", "isis_cv"]#, "entropyDSU", "entropyW", "entropyT", "entropyL", "entropyTL"]
#cmp_againsts = ["entropyW", "entropyT", "entropyL", "entropyD", "entropyS", "entropyU", "entropySW", "isis_cv", "netwins", "rat_sd", "sum_cv"]


old_pc = []
new_pc = []
old_pv = []
new_pv = []

nDAT = len(partIDs)
nDAThlf = nDAT//2
allInds = _N.arange(nDAT)

if look_at_AQ:
    hlf1    = _N.random.choice(allInds, nDAThlf, replace=False)
    hlf2    = _N.setdiff1d(allInds, hlf1)
    for cmp_against in cmp_againsts:
        if cmp_against == "netwins":
            cmp_vs = netwins
        elif cmp_against == "entropyUD":
            cmp_vs = entropyU + entropyD
        elif cmp_against == "pc_sum":
            cmp_vs = pc_sum
        elif cmp_against == "win_aft_win":
            cmp_vs = win_aft_win
        elif cmp_against == "win_aft_los":
            cmp_vs = win_aft_los
        elif cmp_against == "isis_cv":
            cmp_vs = isis_cv
        elif cmp_against == "isis_corr":
            cmp_vs = isis_corr
        elif cmp_against == "moresim":
            cmp_vs = moresim
        elif cmp_against == "sum_sd":
            #cmp_vs = _N.sum(_N.sum(sum_sd, axis=2), axis=1)
            cmp_vs = _N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1)
            #cmp_vs = _N.sum(sum_sd[:, :, 1], axis=1)
            #cmp_vs = _N.sum(sum_sd[:, 2], axis=1)
            #cmp_vs = _N.sum(sum_sd[:, 2], axis=1) / _N.sum(sum_sd[:, 0], axis=1)
            #cmp_vs  = _N.sum(_N.sum(sum_sd, axis=2), axis=1)
            #cmp_vs  = _N.sum(sum_sd[:, :, 1], axis=1)
        elif cmp_against == "sum_cv":
            #cmp_vs = _N.sum(_N.sum(sum_cv, axis=2), axis=1)
            #cmp_vs = _N.sum(sum_cv[:, :, 0], axis=1) + _N.sum(sum_cv[:, :, 2], axis=1)
            cmp_vs = _N.sum(sum_cv[:, :, 2], axis=1)# + _N.sum(sum_cv[:, :, 2], axis=1)
            #cmp_vs = sum_cv[:, 2, 1] / (sum_cv[:, 2, 0] + sum_cv[:, 2, 2])
        elif cmp_against == "entropyWTL":
            cmp_vs = _N.sum(entropyWTL, axis=1)
        elif cmp_against == "entropyDSU":
            cmp_vs = _N.sum(entropyDSU, axis=1)
        elif cmp_against == "entropyTL":
            cmp_vs = (entropyT + entropyL)
        elif cmp_against == "entropyD":
            cmp_vs = entropyD
        elif cmp_against == "entropyS":
            cmp_vs = entropyS
        elif cmp_against == "entropyU":
            cmp_vs = entropyU
        else:
            cmp_vs = eval(cmp_against)

        _cmp_vs = cmp_vs
        _soc_skils = soc_skils
        _rout       = rout
        _imag       = imag
        _switch     = switch
        _fact_pat   = fact_pat
        _AQ28scrs   = AQ28scrs

        fig = _plt.figure(figsize=(14, 3))
        fig.add_subplot(1, 6, 1)
        _plt.xlabel("soc_skills")
        _plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_soc_skils, _cmp_vs)
        _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        _plt.scatter(_soc_skils, _cmp_vs, marker=".", s=50, color="black")
        pc, pv = _ss.pearsonr(_soc_skils, _cmp_vs)
        pcI, pvI = rm_outliersCC(_soc_skils, _cmp_vs)

        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI})                
        _plt.grid(ls=":")        
        fig.add_subplot(1, 6, 2)
        _plt.xlabel("imag")
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_imag, _cmp_vs)
        _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        _plt.scatter(_imag, _cmp_vs, marker=".", s=50, color="black")
        pc, pv = _ss.pearsonr(_imag, _cmp_vs)
        pcI, pvI = rm_outliersCC(_imag, _cmp_vs)

        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI})                
        _plt.grid(ls=":")        
        fig.add_subplot(1, 6, 3)
        _plt.xlabel("routing")
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_rout, _cmp_vs)
        _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        _plt.scatter(_rout, _cmp_vs, marker=".", s=50, color="black")
        pc, pv = _ss.pearsonr(_rout, _cmp_vs)
        pcI, pvI = rm_outliersCC(_rout, _cmp_vs)

        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI})                
        _plt.grid(ls=":")        
        fig.add_subplot(1, 6, 4)
        _plt.xlabel("switch")
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_switch, _cmp_vs)
        _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        _plt.scatter(_switch, _cmp_vs, marker=".", s=50, color="black")
        pc, pv = _ss.pearsonr(_switch, _cmp_vs)
        pcI, pvI = rm_outliersCC(_switch, _cmp_vs)

        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI})                
        _plt.grid(ls=":")
        fig.add_subplot(1, 6, 5)
        _plt.xlabel("factor numbers patterns")
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_fact_pat, _cmp_vs)
        _plt.scatter(xm, ym, marker="o", s=55, color="blue")
        _plt.scatter(_fact_pat, _cmp_vs, marker=".", s=50, color="black")
        pc, pv = _ss.pearsonr(_fact_pat, _cmp_vs)
        pcI, pvI = rm_outliersCC(_fact_pat, _cmp_vs)

        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI})        
        _plt.grid(ls=":")
        fig.add_subplot(1, 6, 6)
        _plt.xlabel("AQ28")
        #_plt.ylabel(cmp_against)
        xm, ym = wnd_mean(_AQ28scrs, _cmp_vs)
        _plt.scatter(xm, ym, marker="o", s=55, color="blue")        
        _plt.scatter(_AQ28scrs, _cmp_vs, marker=".", s=50, color="black")
        pc, pv = _ss.pearsonr(_AQ28scrs, _cmp_vs)
        pcI, pvI = rm_outliersCC(_AQ28scrs, _cmp_vs)

        _plt.title("CC=%(pc).2f  pv=%(pv).3f\nCC=%(pcI).2f  pv=%(pvI).3f" % {"pc" : pc, "pv" : pv, "pcI" : pcI, "pvI" : pvI})

        _plt.grid(ls=":")

        fig.subplots_adjust(hspace=0.4, left=0.06, bottom=0.15, top=0.82, right=0.98)
        _plt.suptitle("%(vs)s" % {"vs" : cmp_against})
        _plt.savefig("AQ_vs_%(vs)s_%(lb)d" % {"vs":cmp_against, "lb" : label})


"""
SHUFFLES = 200
pcspvs_pcIspvIs = _N.empty((6, 4, 2, SHUFFLES))

if look_at_AQ:
    for cmp_against in cmp_againsts:
        if cmp_against == "isis_cv":
            cmp_vs = isis_cv
        elif cmp_against == "entropyW":
            cmp_vs = entropyW
        elif cmp_against == "entropyT":
            cmp_vs = entropyT
        elif cmp_against == "entropyL":
            cmp_vs = entropyL
        elif cmp_against == "entropyD":
            cmp_vs = entropyD
        elif cmp_against == "entropyS":
            cmp_vs = entropyS
        elif cmp_against == "entropyU":
            cmp_vs = entropyU
        elif cmp_against == "pc_sum":
            cmp_vs = pc_sum
        elif cmp_against == "sum_sd":
            cmp_vs = _N.sum(_N.sum(sum_sd, axis=2), axis=1)
        elif cmp_against == "sum_cv":
            cmp_vs = _N.sum(_N.sum(sum_cv, axis=2), axis=1)
        #_N.random.shuffle(cmp_vs)
        
        for shf in range(SHUFFLES):
            hlf1    = _N.random.choice(allInds, nDAThlf, replace=False)
            hlf2    = _N.setdiff1d(allInds, hlf1)

            for h in range(2):
                ths = hlf1 if (h == 0) else hlf2
                _cmp_vs = cmp_vs[ths]
                _soc_skils = soc_skils[ths]
                _rout       = rout[ths]
                _imag       = imag[ths]
                _switch     = switch[ths]
                _fact_pat   = fact_pat[ths]
                _AQ28scrs   = AQ28scrs[ths]            

                pc, pv = _ss.pearsonr(_soc_skils, _cmp_vs)
                pcI, pvI = rm_outliersCC(_soc_skils, _cmp_vs)
                pcspvs_pcIspvIs[0, :, h, shf] = pc, pv, pcI, pvI
                
                pc, pv = _ss.pearsonr(_imag, _cmp_vs)
                pcI, pvI = rm_outliersCC(_imag, _cmp_vs)
                pcspvs_pcIspvIs[1, :, h, shf] = pc, pv, pcI, pvI                

                pc, pv = _ss.pearsonr(_rout, _cmp_vs)
                pcI, pvI = rm_outliersCC(_rout, _cmp_vs)
                pcspvs_pcIspvIs[2, :, h, shf] = pc, pv, pcI, pvI                

                pc, pv = _ss.pearsonr(_switch, _cmp_vs)
                pcI, pvI = rm_outliersCC(_switch, _cmp_vs)
                pcspvs_pcIspvIs[3, :, h, shf] = pc, pv, pcI, pvI                

                pc, pv = _ss.pearsonr(_fact_pat, _cmp_vs)
                pcI, pvI = rm_outliersCC(_fact_pat, _cmp_vs)
                pcspvs_pcIspvIs[4, :, h, shf] = pc, pv, pcI, pvI                

                pc, pv = _ss.pearsonr(_AQ28scrs, _cmp_vs)
                pcI, pvI = rm_outliersCC(_AQ28scrs, _cmp_vs)
                pcspvs_pcIspvIs[5, :, h, shf] = pc, pv, pcI, pvI                

        ##
        fig = _plt.figure(figsize=(13, 3))
        _plt.suptitle(cmp_against)
        for cat in range(6):
            ax = fig.add_subplot(1, 6, cat+1)
            _plt.scatter(pcspvs_pcIspvIs[cat, 2, 0], pcspvs_pcIspvIs[cat, 2, 1], color="black", s=5)
            _plt.axvline(x=0, ls=":")
            _plt.axhline(y=0, ls=":")    
            _plt.xlim(-1, 1)
            _plt.ylim(-1, 1)
        fig.subplots_adjust(left=0.03, right=0.98)
"""
fig = _plt.figure()
_plt.plot(_N.mean(signal_5_95[:, 3], axis=0))
_plt.yticks(_N.linspace(0.1, 0.9, 9))
_plt.grid(ls=":")
_plt.ylim(0, 1)
# ifn = 1
# while os.access("nrm_avg_%(fn)s_%(i)d.png" % {"fn" : data, "i" : ifn}, os.F_OK):
#      ifn += 1
#_plt.savefig("nrm_avg_%(fn)s_%(i)d.png" % {"fn" : data, "i" : ifn})
_plt.savefig("nrm_avg_%(fn)s_%(lb)d.png" % {"fn" : data, "lb" : label})
_plt.close()

