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
from sklearn.decomposition import PCA

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

#visit = 2
#visits= [1, 2]   #  if I want 1 of [1, 2], set this one to [1, 2]
visit = 1
visits= [1, ]   #  if I want 1 of [1, 2], set this one to [1, 2]
    
if data == "TMB2":
    dates = _rt.date_range(start='7/13/2021', end='12/30/2021')
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=15000, minIGI=20, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

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
nMimics = _N.empty(len(partIDs), dtype=_N.int)
t0 = -5
t1 = 10
trigger_temp = _N.empty(t1-t0)
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
signal_5_95 = _N.empty((len(partIDs), t1-t0))

hnd_dat_all = _N.zeros((len(partIDs), TO, 4), dtype=_N.int)

ages      = _N.empty(len(partIDs))
gens      = _N.empty(len(partIDs))
Engs      = _N.empty(len(partIDs))

pc_sum = _N.empty(len(partIDs))
isis    = _N.empty(len(partIDs))
isis_sd    = _N.empty(len(partIDs))
isis_cv    = _N.empty(len(partIDs))
isis_lv    = _N.empty(len(partIDs))
isis_corr    = _N.empty(len(partIDs))
rsp_tms_cv    = _N.empty(len(partIDs))

cntrs = _N.empty((len(partIDs), 2))

maxCs  = _N.empty(len(partIDs))
pcW_UD  = _N.empty(len(partIDs))
pcT_UD  = _N.empty(len(partIDs))
pcL_UD  = _N.empty(len(partIDs))
nTies   = _N.empty(len(partIDs))

du_diffs = _N.empty(len(partIDs))

sum_sd = _N.empty((len(partIDs), 3, 3))
sum_mn = _N.empty((len(partIDs), 3, 3))
sum_sd_RPS = _N.empty((len(partIDs), 3, 3))

sum_sd2 = _N.empty((len(partIDs), 3, 3))
sum_cv = _N.empty((len(partIDs), 3, 3))
marginalCRs = _N.empty((len(partIDs), 3, 3))
sd_M      = _N.empty(len(partIDs))
sd_MW      = _N.empty(len(partIDs))
sd_MT      = _N.empty(len(partIDs))
sd_ML      = _N.empty(len(partIDs))
sd_BW      = _N.empty(len(partIDs))
sd_LW      = _N.empty(len(partIDs))
sd_BW2      = _N.empty(len(partIDs))

predBA      = _N.empty(len(partIDs))

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

mn_stayL      = _N.empty(len(partIDs))
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

pfrm_1st2nd        = _N.empty(len(partIDs))

up_res   = _N.empty(len(partIDs))
dn_res   = _N.empty(len(partIDs))
stay_res         = _N.empty(len(partIDs))
stay_tie         = _N.empty(len(partIDs))

AQ28scrs  = _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))
ans_soc_skils = _N.empty((len(partIDs), 7), dtype=_N.int)
ans_rout      = _N.empty((len(partIDs), 4), dtype=_N.int)
ans_switch    = _N.empty((len(partIDs), 4), dtype=_N.int)
ans_imag      = _N.empty((len(partIDs), 8), dtype=_N.int)
ans_fact_pat  = _N.empty((len(partIDs), 5), dtype=_N.int)

end_strts     = _N.empty(len(partIDs))

all_AI_weights = _N.empty((len(partIDs), TO+1, 3, 3, 2))
all_AI_preds = _N.empty((len(partIDs), TO+1, 3))

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

for partID in partIDs:
    pid += 1

    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit}))
    ##  Conditional Response (UP, DN, STAY | WTL)
    _prob_mvs = dmp["cond_probs"][SHF_NUM][:, strtTr:]
    ##  Conditional Response (R, P, S | WTL)
    _prob_mvsRPS = dmp["cond_probsRPS"][SHF_NUM][:, strtTr:]
    ##  Conditional Response (R, P, S | WTL)    
    _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]    
    ##  Conditional Response (STAY, SWITCH | WTL)
    _prob_mvs_STSW = dmp["cond_probsSTSW"][SHF_NUM][:, strtTr:]
    ##  Other things we might look at:
    ##  prob(UP, DN, ST | RPS)  prob(ST | R)
    _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
    end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])
    all_AI_weights[pid-1] = dmp["AI_weights"][0:TO+1]
    all_AI_preds[pid-1] = dmp["AI_preds"][0:TO+1]

    ans_soc_skils[pid-1], ans_rout[pid-1], ans_switch[pid-1], ans_imag[pid-1], ans_fact_pat[pid-1] = _rt.AQ28ans("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})
    AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})
    ages[pid-1], gens[pid-1], Engs[pid-1] = _rt.Demo("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/DQ1.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})        
    
    hdcol = 0

    hnd_dat_all[pid-1] = _hnd_dat[0:TO]
        
    nR = len(_N.where(_hnd_dat[:, hdcol] == 1)[0])
    nS = len(_N.where(_hnd_dat[:, hdcol] == 2)[0])
    nP = len(_N.where(_hnd_dat[:, hdcol] == 3)[0])

    #nRock    += nR
    #nScissor += nS
    #nPaper   += nP

    #_hnd_dat   = __hnd_dat[0:TO]

    inds =_N.arange(_hnd_dat.shape[0])

    #################  Static features
    ####
    wins = _N.where(_hnd_dat[0:TO-2, 2] == 1)[0]
    ww   = _N.where(_hnd_dat[wins+1, 2] == 1)[0]   #  win followed by win
    wt   = _N.where(_hnd_dat[wins+1, 2] == 0)[0]
    wl   = _N.where(_hnd_dat[wins+1, 2] == -1)[0]
    wr   = _N.where(_hnd_dat[wins+1, 0] == 1)[0]
    wp   = _N.where(_hnd_dat[wins+1, 0] == 2)[0]
    ws   = _N.where(_hnd_dat[wins+1, 0] == 3)[0]        
    
    win_aft_win[pid-1] = len(ww) / len(wins)
    tie_aft_win[pid-1] = len(wt) / len(wins)
    los_aft_win[pid-1] = len(wl) / len(wins)
    
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


    ####    
    ties = _N.where(_hnd_dat[0:TO-2, 2] == 0)[0]
    tw   = _N.where(_hnd_dat[ties+1, 2] == 1)[0]
    tt   = _N.where(_hnd_dat[ties+1, 2] == 0)[0]
    tl   = _N.where(_hnd_dat[ties+1, 2] == -1)[0]
    tr   = _N.where(_hnd_dat[ties+1, 0] == 1)[0]
    tp   = _N.where(_hnd_dat[ties+1, 0] == 2)[0]
    ts   = _N.where(_hnd_dat[ties+1, 0] == 3)[0]
    nTies[pid-1] = len(ties)    
    
    win_aft_tie[pid-1] = len(tw) / len(ties)
    tie_aft_tie[pid-1] = len(tt) / len(ties)
    los_aft_tie[pid-1] = len(tl) / len(ties)
    
    ####

    ################################
    cv_sum = 0
    marginalCRs[pid-1] = _emp.marginalCR(_hnd_dat)

    ################################
    prob_mvs  = _prob_mvs[:, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPS  = _prob_mvsRPS[:, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSURPS  = _prob_mvsDSURPS[:, 0:TO - win]  #  is bigger than hand by win size        
    prob_mvs_STSW  = _prob_mvs_STSW[:, 0:TO - win]  #  is bigger than hand by win size    
    prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
    prob_mvs_RPS = prob_mvsRPS.reshape((3, 3, prob_mvsRPS.shape[1]))
    prob_mvs_DSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))
    prob_mvs_STSW = prob_mvs_STSW.reshape((3, 2, prob_mvs_STSW.shape[1]))
    #  _N.sum(prob_mvs_STSW[0], axis=0) = 1, 1, 1, 1, 1, 1, (except at ends)
    #  get_dbehv is the sum of absolute value of derivatives of CR prob components 3 x 3 of them
    dbehv = _crut.get_dbehv(prob_mvs, None, equalize=True)
    dbehv_RPS = _crut.get_dbehv(prob_mvs_RPS, None, equalize=True)
    dbehv_DSURPS = _crut.get_dbehv(prob_mvs_DSURPS, None, equalize=True)    

    tMv = _N.diff(_hnd_dat[:, 3])
    succ = _hnd_dat[1:, 2]

    ###  smmooth it
    dbehv = _N.convolve(dbehv + 0.115*dbehv_RPS, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2)#  3 from label71

    preds = all_AI_preds[pid-1]
    
    PCS=5    
    prob_Mimic            = _N.empty((2, prob_mvs.shape[2]))
    #sd_M[pid-1]               = _N.std(prob_mvs[0, 0] + prob_mvs[1, 1] + prob_mvs[2, 2])
    sd_M[pid-1]               = _N.std(prob_mvs[0, 0] + prob_mvs[2, 2])
    t00 = 0
    t01 = prob_mvs.shape[2]

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

    #  Is TIE like a WIN or TIE like a LOSE?
    #  ENT_WT = entropy of (UP|WIN and UP|TIE) + entropy (DN|WIN and DN|TIE) + entropy (UP|WIN and UP|TIE)
    #  ENT_LT = entropy of (UP|LOS and UP|TIE) + entropy (DN|LOS and DN|TIE) + entropy (UP|LOS and UP|TIE)

    #actions_independent[pid-1] = wtl_independent                   #  3
    #cond_distinguished[pid-1] = condition_distinguished  #  2

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

    ####  how much does a probability fluctuate over the game?
    sds = _N.std(prob_mvs, axis=2)
    sdsRPS = _N.std(prob_mvs_RPS, axis=2)
    
    #sds = _N.std(prob_pcs, axis=0)
    mns = _N.mean(prob_mvs, axis=2)
    mnsRPS = _N.mean(prob_mvs_RPS, axis=2)        
    sum_cv[pid-1] = sds/(1-_N.abs(0.5-mns))
    sum_sd[pid-1] = sds
    sum_mn[pid-1] = mns
    sum_sd_RPS[pid-1] = sdsRPS
    
    #pc12_2, pv12_2 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 2])    

    netwins[pid-1] = _N.sum(_hnd_dat[:, 2])

    hnd_dat = _hnd_dat[inds]

    avgs = _N.empty((len(maxs)-2*cut, t1-t0))

    for im in range(cut, len(maxs)-cut):
        st = 0
        en = t1-t0
        if maxs[im] + t0 < 0:   #  just don't use this one
            print("DON'T USE THIS ONE")
            avgs[im-1, :] = 0
        else:
            try:
                avgs[im-1, :] = hnd_dat[maxs[im]+t0:maxs[im]+t1, 2]
            except ValueError:
                print("*****  %(1)d  %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
                print(avgs[im-1, :].shape)
                print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2])


    all_avgs[pid-1] = _N.mean(avgs, axis=0)
    #fig.add_subplot(5, 5, pid)
    #_plt.plot(_N.mean(avgs, axis=0))

    srtd   = _N.sort(all_avgs[pid-1, 1:], axis=0)
    signal_5_95[pid-1] = all_avgs[pid-1]

    #pfrm_change36[pid-1] = _N.max(signal_5_95[pid-1, 0, 3:6]) - _N.min(signal_5_95[pid-1, 0, 3:6])


    imax36 = _N.argmax(signal_5_95[pid-1, 3:6])+3
    imin36 = _N.argmin(signal_5_95[pid-1, 3:6])+3
    imax69 = _N.argmax(signal_5_95[pid-1, 6:9])+6
    imin69 = _N.argmin(signal_5_95[pid-1, 6:9])+6    
    imax912= _N.argmax(signal_5_95[pid-1, 9:12])+9
    imin912= _N.argmin(signal_5_95[pid-1, 9:12])+9    

    imax_imin_pfrm36[pid-1, 0] = imin36
    imax_imin_pfrm36[pid-1, 1] = imax36
    imax_imin_pfrm69[pid-1, 0] = imin69
    imax_imin_pfrm69[pid-1, 1] = imax69
    imax_imin_pfrm912[pid-1, 0]= imin912
    imax_imin_pfrm912[pid-1, 1]= imax912
    
    pfrm_change36[pid-1] = signal_5_95[pid-1, imax36] - signal_5_95[pid-1, imin36]
    pfrm_change69[pid-1] = signal_5_95[pid-1, imax69] - signal_5_95[pid-1, imin69]
    pfrm_change912[pid-1]= signal_5_95[pid-1, imax912] - signal_5_95[pid-1, imin912]


#############  AI WEIGHTS
aAw = all_AI_weights   #  len(partIDs) x (T+1) x 3 x 3 x 2
diffAIw = _N.diff(aAw, axis=4).reshape(aAw.shape[0], aAw.shape[1], aAw.shape[2], aAw.shape[3])         #  len(partIDs) x (T+1) x 3 x 3
stg1  = _N.std(diffAIw, axis=3)   #  len(partIDs) x (T+1) x 3
stg2  = _N.mean(diffAIw, axis=3)   #  len(partIDs) x (T+1) x 3

AIfts = _N.std(stg1, axis=1)      #  len(partIDs) x 3  difference in R,P,S
AIfts0 = AIfts[:, 0]

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
AIfts = _N.std(_N.mean(diffAIw, axis=3), axis=1)      #  len(partIDs) x 3  difference in R,P,S
#  AIfts[:, 0]    <--- an example of an AI feature.  Try this

features_cab2 = ["isis", "isis_cv", "isis_corr", "isis_lv",
                "pfrm_change69"]
features_cab1 = ["sds00", "sds01", "sds02",
                 "sds10", "sds11", "sds12",
                 "sds20", "sds21", "sds22"]
features_AI  = ["AIfts0"]
features_stat= ["netwins",
                "win_aft_win", "win_aft_tie", "win_aft_los", 
                "tie_aft_win", "tie_aft_tie", "tie_aft_los", 
                "los_aft_win", "los_aft_tie", "los_aft_los"]
#features_stat = []

cmp_againsts = features_cab1 + features_cab2 + features_stat + features_AI
dmp_dat = {}
for cmp_vs in cmp_againsts:
    dmp_dat[cmp_vs] = eval(cmp_vs)

# = _N.std(marginalCRs, axis=2)   #  how different are USD in LOSE condition


dmp_dat["features_cab1"]  = features_cab1
dmp_dat["features_cab2"]  = features_cab2
dmp_dat["features_stat"]  = features_stat
dmp_dat["features_AI"]    = features_AI
dmp_dat["marginalCRs"] = marginalCRs
dmp_dat["AQ28scrs"]    = AQ28scrs
dmp_dat["soc_skils"] = soc_skils
dmp_dat["imag"] = imag
dmp_dat["rout"] = rout
dmp_dat["switch"] = switch
dmp_dat["fact_pat"] = fact_pat
dmp_dat["ans_soc_skils"] = ans_soc_skils
dmp_dat["ans_imag"] = ans_imag
dmp_dat["ans_rout"] = ans_rout
dmp_dat["ans_switch"] = ans_switch
dmp_dat["ans_fact_pat"] = ans_fact_pat
dmp_dat["all_prob_mvsA"] = _N.array(all_prob_mvs)
dmp_dat["label"] = label
dmp_dat["signal_5_95"] = signal_5_95
dmp_dat["t0"]  = t0
dmp_dat["t1"]  = t1
dmp_dat["win"] = win
dmp_dat["ages"] = ages
dmp_dat["gens"] = gens
dmp_dat["Engs"] = Engs
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
