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
import AIiRPS.models.empirical_ken as _emp

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
    while x[ix[i+1]] - x[ix[i]] > 2.5*dsx:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > 2.5*dsx:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > 2.5*dsy:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > 2.5*dsy:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    #print("%(ths)d" % {"ths" : len(ths)})
    return _ss.pearsonr(x[interiorPts], y[interiorPts])

def only_complete_data(partIDs, TO, label, SHF_NUM):
    pid = -1
    incomplete_data = []
    for partID in partIDs:
        pid += 1

        dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_1.dmp" % {"rpsm" : partID, "lb" : label}))
        _prob_mvs = dmp["cond_probs"][SHF_NUM]
        _prob_mvsRPS = dmp["cond_probsRPS"][SHF_NUM]        
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
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    ths = _N.where(isi[1:-1] <= minISI)[0] + 1
    #print(len(ths))
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
    """
    _sig   T x 3
    """
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


##  Then I expect wins following UPs and DOWNs to also be correlated to AQ28
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

#visit = 2
#visits= [1, 2]   #  if I want 1 of [1, 2], set this one to [1, 2]
visit = 1
visits= [1]   #  if I want 1 of [1, 2], set this one to [1, 2]
    
if data == "TMB2":
    dates = _rt.date_range(start='7/13/2021', end='12/30/2021')
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=15000, minIGI=20, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=800, max_meanIGI=8000, minIGI=200, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    ####  use this for reliability
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_AND_FALSE_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=8000, minIGI=50, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

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

SHUFFLES = 1

t0 = -5
t1 = 10
trigger_temp = _N.empty(t1-t0)
cut = 1
all_avgs = _N.empty((len(partIDs), SHUFFLES+1, t1-t0))
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
signal_5_95 = _N.empty((len(partIDs), 4, t1-t0))

hnd_dat_all = _N.zeros((len(partIDs), TO, 4), dtype=_N.int)

pc_sum = _N.empty(len(partIDs))
pc_sum01 = _N.empty(len(partIDs))
pc_sum02 = _N.empty(len(partIDs))
pc_sum12 = _N.empty(len(partIDs))
isis    = _N.empty(len(partIDs))
isis_sd    = _N.empty(len(partIDs))
isis_cv    = _N.empty(len(partIDs))
isis_lv    = _N.empty(len(partIDs))
isis_corr    = _N.empty(len(partIDs))
rsp_tms_cv    = _N.empty(len(partIDs))
coherence    = _N.empty(len(partIDs))
ages      = _N.empty(len(partIDs))
gens      = _N.empty(len(partIDs))
Engs      = _N.empty(len(partIDs))

corr_UD    = _N.empty((len(partIDs), 3))

score  = _N.empty(len(partIDs))
pcW_UD  = _N.empty(len(partIDs))
pcT_UD  = _N.empty(len(partIDs))
pcL_UD  = _N.empty(len(partIDs))
pc0001s  = _N.empty(len(partIDs))
pc0002s  = _N.empty(len(partIDs))
pc0010s  = _N.empty(len(partIDs))
pc0011s  = _N.empty(len(partIDs))
pc0012s  = _N.empty(len(partIDs))
pc0020s  = _N.empty(len(partIDs))
pc0021s  = _N.empty(len(partIDs))
pc0022s  = _N.empty(len(partIDs))
##########
pc0102s  = _N.empty(len(partIDs))
pc0110s  = _N.empty(len(partIDs))
pc0111s  = _N.empty(len(partIDs))
pc0112s  = _N.empty(len(partIDs))
pc0120s  = _N.empty(len(partIDs))
pc0121s  = _N.empty(len(partIDs))
pc0122s  = _N.empty(len(partIDs))
##########
pc0210s  = _N.empty(len(partIDs))
pc0211s  = _N.empty(len(partIDs))
pc0212s  = _N.empty(len(partIDs))
pc0220s  = _N.empty(len(partIDs))
pc0221s  = _N.empty(len(partIDs))
pc0222s  = _N.empty(len(partIDs))
##########
pc1011s  = _N.empty(len(partIDs))
pc1012s  = _N.empty(len(partIDs))
pc1020s  = _N.empty(len(partIDs))
pc1021s  = _N.empty(len(partIDs))
pc1022s  = _N.empty(len(partIDs))
##########
pc1112s  = _N.empty(len(partIDs))
pc1120s  = _N.empty(len(partIDs))
pc1121s  = _N.empty(len(partIDs))
pc1122s  = _N.empty(len(partIDs))
##########
pc1220s  = _N.empty(len(partIDs))
pc1221s  = _N.empty(len(partIDs))
pc1222s  = _N.empty(len(partIDs))
##########
pc2021s  = _N.empty(len(partIDs))
pc2022s  = _N.empty(len(partIDs))
##########
pc2122s  = _N.empty(len(partIDs))


moresimV1  = _N.empty(len(partIDs))
moresimV2  = _N.empty(len(partIDs))
moresimV3  = _N.empty(len(partIDs))
moresimV4  = _N.empty(len(partIDs))
moresimST  = _N.empty(len(partIDs))
moresimSW  = _N.empty(len(partIDs))
moresim  = _N.empty(len(partIDs))
moresiment  = _N.empty(len(partIDs))
sum_sd = _N.empty((len(partIDs), 3, 3))
sum_sd_RPS = _N.empty((len(partIDs), 3, 3))

sum_sd2 = _N.empty((len(partIDs), 3, 3))
sum_cv = _N.empty((len(partIDs), 3, 3))
marginalCRs = _N.empty((len(partIDs), 3, 3))
entropyDSU = _N.empty((len(partIDs), 3))
entropyD = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropyS = _N.empty(len(partIDs))
entropyU = _N.empty(len(partIDs))
#entropyUD2 = _N.empty(len(partIDs))
entropyS2 = _N.empty(len(partIDs))
entropyDr = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropySr = _N.empty(len(partIDs))
entropyUr = _N.empty(len(partIDs))
entropyW = _N.empty(len(partIDs))   #  
entropyT = _N.empty(len(partIDs))
entropyL = _N.empty(len(partIDs))
entropyRPS1 = _N.empty(len(partIDs))   #  
entropyRPS2 = _N.empty(len(partIDs))
entropyRPS3 = _N.empty(len(partIDs))
entropyW2 = _N.empty(len(partIDs))   #  
entropyT2 = _N.empty(len(partIDs))
entropyL2 = _N.empty(len(partIDs))

entropyM  = _N.empty(len(partIDs))
entropyB  = _N.empty(len(partIDs))
sd_M      = _N.empty(len(partIDs))
sd_MW      = _N.empty(len(partIDs))
sd_MT      = _N.empty(len(partIDs))
sd_ML      = _N.empty(len(partIDs))
sd_BW      = _N.empty(len(partIDs))
sd_LW      = _N.empty(len(partIDs))
sd_BW2      = _N.empty(len(partIDs))

sd_BT      = _N.empty(len(partIDs))
sd_BL      = _N.empty(len(partIDs))
m_M      = _N.empty(len(partIDs))
pc_M1      = _N.empty(len(partIDs))
pc_M2      = _N.empty(len(partIDs))
pc_M3      = _N.empty(len(partIDs))
m_MW      = _N.empty(len(partIDs))
m_MT      = _N.empty(len(partIDs))
m_ML      = _N.empty(len(partIDs))
m_BW      = _N.empty(len(partIDs))
m_BT      = _N.empty(len(partIDs))
m_BL      = _N.empty(len(partIDs))

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

R_aft_win  = _N.empty(len(partIDs))
R_aft_los  = _N.empty(len(partIDs))
R_aft_tie  = _N.empty(len(partIDs))
P_aft_win  = _N.empty(len(partIDs))
P_aft_los  = _N.empty(len(partIDs))
P_aft_tie  = _N.empty(len(partIDs))
S_aft_win  = _N.empty(len(partIDs))
S_aft_los  = _N.empty(len(partIDs))
S_aft_tie  = _N.empty(len(partIDs))

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

all_AI_weights = _N.empty((len(partIDs), 301, 3, 3, 2))

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

RPS_ratios = _N.empty((len(partIDs), 3))
RPS_ratiosMet = _N.empty(len(partIDs))

#  DISPLAYED AS R,S,P
#  look for RR RS RP
#  look for SR SS SP
#  look for PR PS PP


L30  = 30
for partID in partIDs:
    pid += 1

    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit}))
    _prob_mvs = dmp["cond_probs"][SHF_NUM][:, strtTr:]
    _prob_mvsRPS = dmp["cond_probsRPS"][SHF_NUM][:, strtTr:]    
    _prob_mvs_STSW = dmp["cond_probsSTSW"][SHF_NUM][:, strtTr:]    
    _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
    end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])

    hdcol = 0

    hnd_dat_all[pid-1] = _hnd_dat
    # if _hnd_dat[0, hdcol] == 1:
    #     nRock0 += 1
    # elif _hnd_dat[0, hdcol] == 2:
    #     nScissor0 += 1
    # elif _hnd_dat[0, hdcol] == 3:
    #     nPaper0 += 1
    # if _hnd_dat[30, hdcol] == 1:
    #     nRock30 += 1
    # elif _hnd_dat[30, hdcol] == 2:
    #     nScissor30 += 1
    # elif _hnd_dat[30, hdcol] == 3:
    #     nPaper30 += 1
    # if _hnd_dat[60, hdcol] == 1:
    #     nRock60 += 1
    # elif _hnd_dat[60, hdcol] == 2:
    #     nScissor60 += 1
    # elif _hnd_dat[60, hdcol] == 3:
    #     nPaper60 += 1
        
    nR = len(_N.where(_hnd_dat[:, hdcol] == 1)[0])
    nS = len(_N.where(_hnd_dat[:, hdcol] == 2)[0])
    nP = len(_N.where(_hnd_dat[:, hdcol] == 3)[0])
    #nRock    += nR
    #nScissor += nS
    #nPaper   += nP

    #_hnd_dat   = __hnd_dat[0:TO]

    inds =_N.arange(_hnd_dat.shape[0])

    all_AI_weights[pid-1] = dmp["AI_weights"]
    
    ####
    wins = _N.where(_hnd_dat[0:TO-2, 2] == 1)[0]
    ww   = _N.where(_hnd_dat[wins+1, 2] == 1)[0]
    wt   = _N.where(_hnd_dat[wins+1, 2] == 0)[0]
    wl   = _N.where(_hnd_dat[wins+1, 2] == -1)[0]
    wr   = _N.where(_hnd_dat[wins+1, 0] == 1)[0]
    wp   = _N.where(_hnd_dat[wins+1, 0] == 2)[0]
    ws   = _N.where(_hnd_dat[wins+1, 0] == 3)[0]        
    
    win_aft_win[pid-1] = len(ww) / len(wins)
    tie_aft_win[pid-1] = len(wt) / len(wins)
    los_aft_win[pid-1] = len(wl) / len(wins)
    R_aft_win[pid-1] = len(wr) / len(wins)
    P_aft_win[pid-1] = len(wp) / len(wins)
    S_aft_win[pid-1] = len(ws) / len(wins)
    
    ####    
    loses = _N.where(_hnd_dat[0:TO-2, 2] == -1)[0]
    lw   = _N.where(_hnd_dat[loses+1, 2] == 1)[0]
    lt   = _N.where(_hnd_dat[loses+1, 2] == 0)[0]
    ll   = _N.where(_hnd_dat[loses+1, 2] == -1)[0]
    lr   = _N.where(_hnd_dat[loses+1, 0] == 1)[0]
    lp   = _N.where(_hnd_dat[loses+1, 0] == 2)[0]
    ls   = _N.where(_hnd_dat[loses+1, 0] == 3)[0]        
    
    win_aft_los[pid-1] = len(lw) / len(loses)
    tie_aft_los[pid-1] = len(lt) / len(loses)
    los_aft_los[pid-1] = len(ll) / len(loses)
    R_aft_los[pid-1] = len(lr) / len(loses)
    P_aft_los[pid-1] = len(lp) / len(loses)
    S_aft_los[pid-1] = len(ls) / len(loses)    
    
    ####    
    ties = _N.where(_hnd_dat[0:TO-2, 2] == 0)[0]
    tw   = _N.where(_hnd_dat[ties+1, 2] == 1)[0]
    tt   = _N.where(_hnd_dat[ties+1, 2] == 0)[0]
    tl   = _N.where(_hnd_dat[ties+1, 2] == -1)[0]
    tr   = _N.where(_hnd_dat[ties+1, 0] == 1)[0]
    tp   = _N.where(_hnd_dat[ties+1, 0] == 2)[0]
    ts   = _N.where(_hnd_dat[ties+1, 0] == 3)[0]        
    
    win_aft_tie[pid-1] = len(tw) / len(ties)
    tie_aft_tie[pid-1] = len(tt) / len(ties)
    los_aft_tie[pid-1] = len(tl) / len(ties)
    R_aft_tie[pid-1] = len(tr) / len(ties)
    P_aft_tie[pid-1] = len(tp) / len(ties)
    S_aft_tie[pid-1] = len(ts) / len(ties)    
    
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

    cv_sum = 0
    dhd = _N.empty(300)
    dhd[0:299] = _N.diff(_hnd_dat[:, 3])
    dhd[299] = dhd[298]
    dhdr = dhd.reshape((20, 15))
    rsp_tms_cv[pid-1] = _N.mean(_N.std(dhdr, axis=1) / _N.mean(dhdr, axis=1))
    
    
    #rsp_tms_cv[pid-1] = _N.std(_hnd_dat[:, 3]) / _N.mean(_hnd_dat[:, 3])
    marginalCRs[pid-1] = _emp.marginalCR(_hnd_dat)
    prob_mvs  = _prob_mvs[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size
    prob_mvsRPS  = _prob_mvsRPS[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size    
    prob_mvs_STSW  = _prob_mvs_STSW[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size    
    prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
    prob_mvs_RPS = prob_mvsRPS.reshape((3, 3, prob_mvsRPS.shape[1]))    
    prob_mvs_STSW = prob_mvs_STSW.reshape((3, 2, prob_mvs_STSW.shape[1]))
    #  _N.sum(prob_mvs_STSW[0], axis=0) = 1, 1, 1, 1, 1, 1, (except at ends)
    #dbehv = _crut.get_dbehv(prob_mvs, gk, equalize=True)
    dbehv = _crut.get_dbehv(prob_mvs, gkISI, equalize=True)



    y0 = _N.abs(_N.diff(prob_mvs_RPS[2, 0]))
    y1 = _N.abs(_N.diff(prob_mvs_RPS[2, 1]))
    y2 = _N.abs(_N.diff(prob_mvs_RPS[2, 2]))
    # y3 = _N.abs(_N.diff(prob_mvs_RPS[1, 0]))
    # y4 = _N.abs(_N.diff(prob_mvs_RPS[1, 1]))
    # y5 = _N.abs(_N.diff(prob_mvs_RPS[1, 2]))
    # y6 = _N.abs(_N.diff(prob_mvs_RPS[0, 0]))
    # y7 = _N.abs(_N.diff(prob_mvs_RPS[0, 1]))
    # y8 = _N.abs(_N.diff(prob_mvs_RPS[0, 2]))
    
    y  = (y0 + y1 + y2)# + y3 + y4 + y5 + y6 + y7 + y8)
    dy = _N.diff(y)       #  use to find maxes of time derivative
    fdy = _N.convolve(dy, gk, mode="same")
    dbehv += fdy

    #fdbehv = _N.convolve(dbehv, gkISI, mode="same")
    #maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2) #  3 from label71
    maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2) #  3 from label71

    PCS=5    
    prob_Mimic            = _N.empty((2, prob_mvs.shape[2]))
    #sd_M[pid-1]               = _N.std(prob_mvs[0, 0] + prob_mvs[1, 1] + prob_mvs[2, 2])
    sd_M[pid-1]               = _N.std(prob_mvs[0, 0] + prob_mvs[2, 2])
    pc_M1[pid-1],pv               = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 1])
    pc_M2[pid-1],pv               = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 1])
    pc_M3[pid-1],pv               = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 2])
    #  
    sd_MW[pid-1]               = _N.std(prob_mvs[0, 0])
    sd_BW[pid-1]               = _N.std(prob_mvs[0, 1])# / _N.mean(prob_mvs[0, 1])
    sd_BW2[pid-1]               = _N.std(prob_mvs[0, 2])# / _N.mean(prob_mvs[0, 1])

    sd_MT[pid-1]               = _N.std(prob_mvs[1, 1])    
    sd_BT[pid-1]               = _N.std(prob_mvs[1, 2])# / _N.mean(prob_mvs[1, 2])
    sd_BL[pid-1]               = _N.std(prob_mvs[2, 0])# / _N.mean(prob_mvs[2, 0    

    sd_ML[pid-1]               = _N.std(prob_mvs[2, 2])
    #sd_B[pid-1]               = _N.std(prob_mvs[0, 1] + prob_mvs[1, 2] + prob_mvs[2, 0])
    #sd_B[pid-1]               = _N.std(prob_mvs[0, 1])


    m_BW[pid-1]               = _N.mean(prob_mvs[0, 1])# / _N.mean(prob_mvs[0, 1])    

    m_BT[pid-1]               = _N.mean(prob_mvs[1, 2])# / _N.mean(prob_mvs[1, 2])    

    m_BL[pid-1]               = _N.mean(prob_mvs[2, 0])# / _N.mean(prob_mvs[2, 0])            
    #sd_BT[pid-1]               = _N.std(prob_mvs[1, 2])# / (_N.abs(mn - 0.5)+0.1)
    #sd_BL[pid-1]               = _N.std(prob_mvs[2, 0] + prob_mvs[2, 2])# / (_N.abs(mn - 0.5)+0.1)        
    prob_Mimic[0]      = prob_mvs[0, 0]   #  DN | WIN
    #prob_Mimic[1]      = prob_mvs[1, 1]   #  ST | TIE
    prob_Mimic[1]      = prob_mvs[2, 2]   #  UP | LOS
    prob_Beat            = _N.empty((3, prob_mvs.shape[2]))
    prob_Beat[0]       = prob_mvs[0, 1]
    prob_Beat[1]       = prob_mvs[1, 2]
    prob_Beat[2]       = prob_mvs[2, 0]
    entropyB[pid-1] = entropy3(prob_Beat.T, PCS)    
    entropyM[pid-1] = entropy2(prob_Mimic.T, PCS)
    
    #l_maxs = maxs.tolist()
    #if l_maxs[-1] == _hnd_dat.shape[0] - t1:
    #    print("woawoawoa")
    #if maxs[cut] + t0 < 0:
    #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!  woa")
    # MLAG = 20
    # tlag, AC = _eu.autocorrelate(dbehv, MLAG)
    # # dAC = _N.diff(AC)
    # # AC_pks = _N.where((dAC[0:-1] > 0) & (dAC[1:] <= 0))[0]
    # # coherence[pid-1] = _N.std(_N.diff(AC_pks))
    # decr = True
    # for i in range(2, MLAG-1):
    #     if decr:
    #         if (AC[MLAG+i-1] >= AC[MLAG+i]) and (AC[MLAG+i] <= AC[MLAG+i+1]):
    #             decr = False
    #             iLow = i+MLAG
    #     else:
    #         if (AC[MLAG+i-1] <= AC[MLAG+i]) and (AC[MLAG+i] >= AC[MLAG+i+1]):
    #             iHigh = i+MLAG
    #             break
    # coherence[pid-1] = AC[iHigh] - AC[iLow]
    # ACmin = _N.min(AC[MLAG:])
    # coherence[pid-1] = ACmin
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
    #condition_distinguished = _N.array([entropy3(prob_mvs_STSW[:, 0].T, PCS), entropy3(prob_mvs_STSW[:, 1].T, PCS)])
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

    entropyRPS1[pid-1] = entropy3(prob_mvs_RPS[:, 0].T, PCS)
    entropyRPS2[pid-1] = entropy3(prob_mvs_RPS[:, 1].T, PCS)
    entropyRPS3[pid-1] = entropy3(prob_mvs_RPS[:, 2].T, PCS)
    #  _ss.pearsonr(entropyRPS3[ths], rout[ths]) <--  
    #entropyUD2[pid-1] = entsUD_S[0]
    entropyS2[pid-1]  = entsUD_S[1]    
    entropyW[pid-1] = entsWTL3[0]
    entropyT[pid-1] = entsWTL3[1]
    entropyL[pid-1] = entsWTL3[2]
    entropyW2[pid-1] = wtl_independent[0]
    entropyT2[pid-1] = wtl_independent[1]
    entropyL2[pid-1] = wtl_independent[2]
    actions_independent[pid-1] = wtl_independent                   #  3
    #cond_distinguished[pid-1] = condition_distinguished  #  2
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
    #pc, pv = _ss.pearsonr(isi[0:-1], isi[1:])
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
    sdsRPS = _N.std(prob_mvs_RPS, axis=2)
    
    #sds = _N.std(prob_pcs, axis=0)
    mns = _N.mean(prob_mvs, axis=2)
    mnsRPS = _N.mean(prob_mvs_RPS, axis=2)        
    sum_cv[pid-1] = sds/mns
    sum_sd[pid-1] = sds
    sum_sd_RPS[pid-1] = sdsRPS
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


    pc0001, pv01_0 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[0, 1])    
    pc0002, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[0, 2])
    pc0010, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[1, 0])
    pc0011, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[1, 1])
    pc0012, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[1, 2])        
    pc0020, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[2, 0])
    pc0021, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[2, 1])
    pc0022, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[2, 2])
    ################
    pc0102, pv01_0 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[0, 2])    
    pc0110, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 0])
    pc0111, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 1])
    pc0112, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 2])        
    pc0120, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 0])
    pc0121, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 1])
    pc0122, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 2])        
    ################
    pc0210, pv01_0 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[1, 0])    
    pc0211, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[1, 1])
    pc0212, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[1, 2])        
    pc0220, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[2, 0])
    pc0221, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[2, 1])
    pc0222, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[2, 2])        
    ################
    pc1011, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[1, 1])
    pc1012, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[1, 2])        
    pc1020, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[2, 0])
    pc1021, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[2, 1])
    pc1022, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[2, 2])        
    ################
    pc1112, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[1, 2])        
    pc1120, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 0])
    pc1121, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 1])
    pc1122, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 2])        
    ################
    pc1220, pv01_1 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 0])
    pc1221, pv01_1 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 1])
    pc1222, pv01_1 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 2])        
    ################
    pc2021, pv01_1 = _ss.pearsonr(prob_mvs[2, 0], prob_mvs[2, 1])
    pc2022, pv01_1 = _ss.pearsonr(prob_mvs[2, 0], prob_mvs[2, 2])
    ################    
    pc2122, pv01_1 = _ss.pearsonr(prob_mvs[2, 1], prob_mvs[2, 2])            
    

    pc0001s[pid-1]    = pc0001
    pc0002s[pid-1]    = pc0002
    pc0010s[pid-1]    = pc0010
    pc0011s[pid-1]    = pc0011
    pc0012s[pid-1]    = pc0012
    pc0020s[pid-1]    = pc0020
    pc0021s[pid-1]    = pc0021
    pc0022s[pid-1]    = pc0022
    ###################
    pc0102s[pid-1]    = pc0102
    pc0110s[pid-1]    = pc0110
    pc0111s[pid-1]    = pc0111
    pc0112s[pid-1]    = pc0112
    pc0120s[pid-1]    = pc0120
    pc0121s[pid-1]    = pc0121
    pc0122s[pid-1]    = pc0122
    ###################
    pc0210s[pid-1]    = pc0210
    pc0211s[pid-1]    = pc0211
    pc0212s[pid-1]    = pc0212
    pc0220s[pid-1]    = pc0220
    pc0221s[pid-1]    = pc0221
    pc0222s[pid-1]    = pc0222
    ###################
    pc1011s[pid-1]    = pc1011
    pc1012s[pid-1]    = pc1012
    pc1020s[pid-1]    = pc1020
    pc1021s[pid-1]    = pc1021
    pc1022s[pid-1]    = pc1022
    ###################
    pc1112s[pid-1]    = pc1112
    pc1120s[pid-1]    = pc1120
    pc1121s[pid-1]    = pc1121
    pc1122s[pid-1]    = pc1122
    ###################
    pc1220s[pid-1]    = pc1220
    pc1221s[pid-1]    = pc1221
    pc1222s[pid-1]    = pc1222
    ###################
    pc2021s[pid-1]    = pc2021
    pc2022s[pid-1]    = pc2022
    ###################
    pc2122s[pid-1]    = pc2122
    
    #pc12_2, pv12_2 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 2])    

    

    #  CC(T and L) - CC(W and T)q   Is TIE more similar to WIN or LOSE?
    #moresimV1[pid-1] = pc12_0+pc12_1+pc12_2 - (pc01_0+pc01_1+pc01_2)
    #moresimV2[pid-1] = pc12_2+pc12_1 - (pc01_0+pc01_1)


    #moresimV3[pid-1] = pc01_0+pc01_1 - (pc12_2 - pc01_0)
    moresimV3[pid-1] = 2*pc01_0+pc01_1 - pc12_2
    moresimV2[pid-1] = pc01_2 - 5*pc01_1
    moresimV4[pid-1] = pc12_2 - pc01_2
    
    #pc_sum01-moresimV3   pc01_0 + pc01_1 + pc01_2 - (pc12_2 - pc01_0)
    #moresimV4[pid-1] = pc12_0 - (pc01_2)        
    moresim[pid-1] = pc12_0 + pc12_2+pc12_1 - (pc01_0+pc01_2+pc01_1)
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

    for sh in range(SHUFFLES+1):
        if sh > 0:
            _N.random.shuffle(inds)
        hnd_dat = _hnd_dat[inds]

        avgs = _N.empty((len(maxs)-2*cut, t1-t0))
        #print("len(maxs)  %d" % len(maxs))
        #print(maxs)

        for im in range(cut, len(maxs)-cut):
            #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2].shape)
            #print("%(1)d %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
            st = 0
            en = t1-t0
            if maxs[im] + t0 < 0:   #  just don't use this one
                print("DON'T USE THIS ONE")
                avgs[im-1, :] = 0
            else:
                avgs[im-1, :] = hnd_dat[maxs[im]+t0:maxs[im]+t1, 2]

        all_avgs[pid-1, sh] = _N.mean(avgs, axis=0)
        #fig.add_subplot(5, 5, pid)
        #_plt.plot(_N.mean(avgs, axis=0))

    srtd   = _N.sort(all_avgs[pid-1, 1:], axis=0)
    signal_5_95[pid-1, 1] = srtd[int(0.05*SHUFFLES)]
    signal_5_95[pid-1, 2] = srtd[int(0.95*SHUFFLES)]
    signal_5_95[pid-1, 0] = all_avgs[pid-1, 0]
    signal_5_95[pid-1, 3] = (signal_5_95[pid-1, 0] - signal_5_95[pid-1, 1]) / (signal_5_95[pid-1, 2] - signal_5_95[pid-1, 1])

    #pfrm_change36[pid-1] = _N.max(signal_5_95[pid-1, 0, 3:6]) - _N.min(signal_5_95[pid-1, 0, 3:6])

    sInds = _N.argsort(signal_5_95[pid-1, 0, 3:6])
    #sInds = _N.argsort(signal_5_95[pid-1, 0, 1:5])
    if sInds[2] - sInds[0] > 0:
        m36 = 1
    else:
        m36 = -1
    sInds = _N.argsort(signal_5_95[pid-1, 0, 6:9])
    #sInds = _N.argsort(signal_5_95[pid-1, 0, 5:10])
    if sInds[2] - sInds[0] > 0:
        m69 = 1
    else:
        m69 = -1
    sInds = _N.argsort(signal_5_95[pid-1, 0, 9:12])
    #sInds = _N.argsort(signal_5_95[pid-1, 0, 10:15])
    if sInds[2] - sInds[0] > 0:
        m912 = 1
    else:
        m912 = -1

    imax36 = _N.argmax(signal_5_95[pid-1, 0, 3:6])+3
    imin36 = _N.argmin(signal_5_95[pid-1, 0, 3:6])+3
    imax69 = _N.argmax(signal_5_95[pid-1, 0, 6:9])+6
    imin69 = _N.argmin(signal_5_95[pid-1, 0, 6:9])+6    
    imax912= _N.argmax(signal_5_95[pid-1, 0, 9:12])+9
    imin912= _N.argmin(signal_5_95[pid-1, 0, 9:12])+9    

    # imax36 = _N.argmax(signal_5_95[pid-1, 0, 1:5])+1
    # imin36 = _N.argmin(signal_5_95[pid-1, 0, 1:5])+1
    # imax69 = _N.argmax(signal_5_95[pid-1, 0, 5:9])+5
    # imin69 = _N.argmin(signal_5_95[pid-1, 0, 5:9])+5    
    # imax912= _N.argmax(signal_5_95[pid-1, 0, 9:12])+9
    # imin912= _N.argmin(signal_5_95[pid-1, 0, 9:12])+9    
    
    imax_imin_pfrm36[pid-1, 0] = imin36
    imax_imin_pfrm36[pid-1, 1] = imax36
    imax_imin_pfrm69[pid-1, 0] = imin69
    imax_imin_pfrm69[pid-1, 1] = imax69
    imax_imin_pfrm912[pid-1, 0]= imin912
    imax_imin_pfrm912[pid-1, 1]= imax912
    
    pfrm_change36[pid-1] = signal_5_95[pid-1, 0, imax36] - signal_5_95[pid-1, 0, imin36]
    pfrm_change69[pid-1] = signal_5_95[pid-1, 0, imax69] - signal_5_95[pid-1, 0, imin69]
    pfrm_change912[pid-1]= signal_5_95[pid-1, 0, imax912] - signal_5_95[pid-1, 0, imin912]

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

#############  AI WEIGHTS
aAw = all_AI_weights   #  len(partIDs) x (T+1) x 3 x 3 x 2
################
# FEAT1 = _N.mean(_N.std(_N.diff(_N.std(aAw, axis=3), axis=3), axis=2), axis=1)[:, 0]
# FEAT2 = _N.std(_N.std(_N.diff(_N.std(aAw, axis=3), axis=3), axis=2), axis=1)[:, 0]
# FEAT3 = _N.mean(_N.mean(_N.diff(_N.std(aAw, axis=2), axis=3), axis=2), axis=1)[:, 0]
# FEAT4 = _N.std(_N.std(_N.diff(_N.std(aAw, axis=2), axis=3), axis=2), axis=1)[:, 0]

f1 = _N.std(aAw, axis=3)    #  across RPS predictors
f2 = _N.diff(f1, axis=3)
f3 = _N.max(f2, axis=2)
f4 = _N.mean(f3, axis=2)
FEAT5 = _N.std(f4, axis=1)

sds00 = sum_sd[:, 0, 0]
sds01 = sum_sd[:, 0, 1]
sds02 = sum_sd[:, 0, 2]
sds10 = sum_sd[:, 1, 0]
sds11 = sum_sd[:, 1, 1]
sds12 = sum_sd[:, 1, 2]
sds20 = sum_sd[:, 2, 0]
sds21 = sum_sd[:, 2, 1]
sds22 = sum_sd[:, 2, 2]
diffAIw = _N.diff(aAw, axis=4).reshape(aAw.shape[0], aAw.shape[1], aAw.shape[2], aAw.shape[3])         #  len(partIDs) x (T+1) x 3 x 3
stg1  = _N.std(diffAIw, axis=3)   #  len(partIDs) x (T+1) x 3
stg2  = _N.mean(diffAIw, axis=3)   #  len(partIDs) x (T+1) x 3

AIfts = _N.std(stg1, axis=1)      #  len(partIDs) x 3  difference in R,P,S
AIfts0 = AIfts[:, 0]
AIfts5 = AIfts[:, 2]
AIent1  = _N.empty(len(partIDs))
for pid in range(len(partIDs)):
    AIent1[pid] = entropy3(stg1[pid], 5)

# ################
AIftsM = _N.mean(stg1, axis=1)    # the diff   #  len(partIDs) x (T+1) x 3
AIfts4 = AIftsM[:, 0]

# ################
sumAIw = _N.sum(aAw, axis=4).reshape(aAw.shape[0], aAw.shape[1], aAw.shape[2], aAw.shape[3])
stg2  = _N.std(sumAIw, axis=2)   #  len(partIDs) x (T+1) x 3

AIfts1allcomps = _N.mean(_N.sum(sumAIw, axis=3), axis=1)
AIfts1 = AIfts1allcomps[:, 0]
AIfts2 = AIfts1allcomps[:, 1]
AIfts3 = AIfts1allcomps[:, 2]
AIent2  = _N.empty(len(partIDs))
for pid in range(len(partIDs)):
    AIent2[pid] = entropy3(stg2[pid], 5)    

USDdiff0 = _N.std(marginalCRs, axis=2)[:, 0]   #  how different are USD in LOSE condition
USDdiff1 = _N.std(marginalCRs, axis=2)[:, 1]   #  how different are USD in LOSE condition
USDdiff2 = _N.std(marginalCRs, axis=2)[:, 1]   #  how different are USD in LOSE condition

USDdiff3 = _N.std(marginalCRs, axis=1)[:, 0]   #  how different are USD in LOSE condition
USDdiff4 = _N.std(marginalCRs, axis=1)[:, 1]   #  how different are USD in LOSE condition
USDdiff5 = _N.std(marginalCRs, axis=1)[:, 1]   #  how different are USD in LOSE condition


    
#for pid in range(len(partIDs)):
#    AIent2[pid] = entropy3(stg2[pid], 8)

################

# AIfts1allcomps = _N.std(_N.sum(sumAIw, axis=3), axis=1)
# AIfts1 = AIfts1allcomps[:, 0]
# AIfts2 = AIfts1allcomps[:, 1]
# AIfts3 = AIfts1allcomps[:, 2]

sumsdRPS0 = sum_sd_RPS[:, 2, 0]
sumsdRPS1 = sum_sd_RPS[:, 2, 1]
sumsdRPS2 = sum_sd_RPS[:, 2, 2]
#  More sim:  If large
features_cab = ["isis", "isis_cv", "isis_corr", "isis_lv",
                "entropyD", "entropyS", "entropyU",
                "entropyT2", "entropyW2", "entropyL2",
                "entropyT", "entropyW", "entropyL", "entropyM", "entropyB",
                "sds00", "sds01", "sds02",
                "sds10", "sds11", "sds12",
                "sds20", "sds21", "sds22",                
                #"sd_M", "sd_BW", "sd_BW2", "sd_BT", "sd_BL", "sd_MW", "sd_MT", "sd_ML",
                "pc_M1", "pc_M2", "pc_M3", "pfrm_change69", "AIfts0", "AIfts1", "AIfts2", "AIfts3", "AIfts4", "AIfts5", "AIent1", "USDdiff0", "USDdiff1", "USDdiff2", "USDdiff3", "USDdiff4", "USDdiff5", "isis", "isis_lv", "isis_cv", "isis_corr",
                #"moresim", "moresimV4", "moresimV3", "moresimV2",
                "pc0220s", "pc0110s", "pc0010s", 
                "sumsdRPS0", "sumsdRPS1", "sumsdRPS2"]
#                "entropyT", "entropyW", "entropyL", "entropyM", "entropyB", "sd_M", "sd_BW", "sd_BW2", "sd_BT", "sd_BL", "sd_MW", "sd_MT", "sd_ML", "pc_M1", "pc_M2", "pc_M3", "pfrm_change69", "AIfts0", "AIfts1", "AIfts2", "AIfts3", "AIfts4", "AIfts5", "AIent1", "USDdiff", "isis", "isis_lv", "isis_cv", "isis_corr",
    # "pc0001s", "pc0002s", "pc0010s", "pc0011s", "pc0012s", "pc0020s", "pc0021s", "pc0022s",
    # "pc0102s", "pc0110s", "pc0111s", "pc0112s", "pc0120s", "pc0121s", "pc0122s",
    # "pc0210s", "pc0211s", "pc0212s", "pc0220s", "pc0221s", "pc0222s",
    # "pc1011s", "pc1012s", "pc1020s", "pc1021s", "pc1022s",
    # "pc1112s", "pc1120s", "pc1121s", "pc1122s",
    # "pc1220s", "pc1221s", "pc1222s",
    # "pc2021s", "pc2022s",
    # "pc2122s",

#features_cab = ["moresimV4"]
#    "m_BW", "m_BT", "m_BL", "sd_MW", 
#                "pfrm_change36", "pfrm_change69", "pfrm_change912"]
features_stat= ["u_or_d_res", "u_or_d_tie","up_res", "dn_res",
                "stay_res", "stay_tie",                
                #"netwins",
                # "moresimV2", "moresimV3", "moresimV4",
                "win_aft_win", "win_aft_tie", "win_aft_los", 
                "tie_aft_win", "tie_aft_tie", "tie_aft_los", 
                "los_aft_win", "los_aft_tie", "los_aft_los",
                "R_aft_win", "R_aft_tie", "R_aft_los"]
#features_stat = []

cmp_againsts = features_cab + features_stat
dmp_dat = {}
for cmp_vs in cmp_againsts:
    dmp_dat[cmp_vs] = eval(cmp_vs)

# = _N.std(marginalCRs, axis=2)   #  how different are USD in LOSE condition

dmp_dat["features_cab"]  = features_cab
dmp_dat["features_stat"] = features_stat
dmp_dat["marginalCRs"] = marginalCRs
dmp_dat["AQ28scrs"]    = AQ28scrs
dmp_dat["soc_skils"] = soc_skils
dmp_dat["imag"] = imag
dmp_dat["rout"] = rout
dmp_dat["switch"] = switch
dmp_dat["fact_pat"] = fact_pat
dmp_dat["all_prob_mvsA"] = _N.array(all_prob_mvs)
dmp_dat["label"] = label
dmp_dat["signal_5_95"] = signal_5_95
dmp_dat["t0"]  = t0
dmp_dat["t1"]  = t1
dmp_dat["win"] = win
dmp_dat["all_maxs"] = all_maxs
dmp_dat["partIDs"] = partIDs
dmp_dat["imax_imin_pfrm36"] = imax_imin_pfrm36
dmp_dat["imax_imin_pfrm69"] = imax_imin_pfrm69
dmp_dat["imax_imin_pfrm912"] = imax_imin_pfrm912
dmp_dat["all_AI_weights"] = all_AI_weights
dmp_dat["data"] = data
dmp_dat["end_strts"] = end_strts
dmp_dat["hnd_dat_all"] = hnd_dat_all


dmpout = open("predictAQ28dat/AQ28_vs_RPS_%d.dmp" % visit, "wb")
pickle.dump(dmp_dat, dmpout, -1)
dmpout.close()

# # R S P
# #  see if RS > SR
# #  see if PR > RP
# transSkew = _N.abs(RSPtrans[:, 0, 1] - RSPtrans[:, 1, 0]) + _N.abs(RSPtrans[:, 0, 2] - RSPtrans[:, 2, 0]) + _N.abs(RSPtrans[:, 1, 2] - RSPtrans[:, 2, 1]) 


# #  If I favor R->S over S->R, I will also favor P->R over R->P
# #  I prefer finger 1->2, then I prefer finger 2->3, then I prefer finger 3->1

# preferFinger12  = RSPtrans[:, 0, 1] - RSPtrans[:, 1, 0]  #  1->2 is a DN
# preferFinger23  = RSPtrans[:, 1, 2] - RSPtrans[:, 2, 1]  #  2->3 is a DN
# preferFinger31  = RSPtrans[:, 2, 0] - RSPtrans[:, 0, 2]  #  3->1 is a DN

# #  Preference R #1, P #2, S #3
# #  people who do 

# #  Inwards from outer key
# #  (RSPtrans[:, 0, 1]-RSPtrans[:, 2, 1])
# #  _ss.pearsonr(RSPtrans[:, 0, 1], RSPtrans[:, 2, 1])   (negative)
# #  Lots of 12 tends to mean less 32
# #  

sfeats  = ["R_aft_tie", "R_aft_win", "R_aft_los",
           "P_aft_tie", "P_aft_win", "P_aft_los",
           "S_aft_tie", "S_aft_win", "S_aft_los"]
ths = _N.where((AQ28scrs > 35))[0]
for sfeat in sfeats:
    exec("feat = %s" % sfeat)
    print("---------   %s" % sfeat)
    pc, pv = _ss.pearsonr(feat[ths], soc_skils[ths])
    print("SS pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], imag[ths])
    print("IM pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], rout[ths])
    print("RT pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], switch[ths])
    print("SW %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], fact_pat[ths])
    print("FP pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})


sfeats  = ["sum_sd_RPS[:, 0, 0]", "sum_sd_RPS[:, 0, 1]", "sum_sd_RPS[:, 0, 2]",
           "sum_sd_RPS[:, 1, 0]", "sum_sd_RPS[:, 1, 1]", "sum_sd_RPS[:, 1, 2]",
           "sum_sd_RPS[:, 2, 0]", "sum_sd_RPS[:, 2, 1]", "sum_sd_RPS[:, 2, 2]"]

ths = _N.where((AQ28scrs > 35))[0]
for sfeat in sfeats:
    exec("feat = %s" % sfeat)
    print("---------   %s" % sfeat)
    pc, pv = _ss.pearsonr(feat[ths], soc_skils[ths])
    print("SS pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], imag[ths])
    print("IM pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], rout[ths])
    print("RT pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], switch[ths])
    print("SW %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
    pc, pv = _ss.pearsonr(feat[ths], fact_pat[ths])
    print("FP pc %(pc).3f   pv %(pv).1e" % {"pc" : pc, "pv" : pv})
