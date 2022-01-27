#!/usr/bin/python

from sklearn import linear_model
import sklearn.linear_model as _skl
import numpy as _N
import AIiRPS.utils.read_taisen as _rt
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import AIiRPS.utils.read_taisen as _rd
import AIiRPS.utils.misc as _Am
from scipy.signal import savgol_filter
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, shuffle_discrete_contiguous_regions, mtfftc
import AIiRPS.skull_plot as _sp
import os
import sys
from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
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

import feature_names as _feat_names
import AIRPSfeatures as _aift


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

#def set_resp_time_to_cutoff(ts, pctl=0.95):
    
def amplitude_ts(probs, win, axis=2):
    if axis == 2:
        L = probs.shape[2]
        amp_ts = _N.empty(L - win)
        
        for t in range(L-win):
            amp_ts[t] = _N.sum(_N.std(probs[:, :, t:t+win], axis=axis))
    else:
        L = probs.shape[1]
        amp_ts = _N.empty(L - win)
        
        for t in range(L-win):
            amp_ts[t] = _N.sum(_N.std(probs[:, t:t+win], axis=axis))
            
    return amp_ts

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
        _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM]                
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

mouseOffset = 400
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
visits= [1, ]   #  if I want 1 of [1, 2], set this one to [1, 2]
    
if data == "TMB2":
    dates = _rt.date_range(start='7/13/2021', end='12/30/2021')
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=15000, minIGI=20, maxIGI=30000, MinWinLossRat=0.35, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=15000, minIGI=20, maxIGI=30000, MinWinLossRat=0.35, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=800, max_meanIGI=8000, minIGI=200, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    ####  use this for reliability
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_AND_FALSE_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=8000, minIGI=50, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

A1 = []
show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

win_type = 2   #  window is of fixed number of games
win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
win     = 4
smth    = 1
smth    = 3
label          = win_type*100+win*10+smth
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
all_avgs = _N.empty((len(partIDs), SHUFFLES+1, t1-t0))
netwins  = _N.empty(len(partIDs), dtype=_N.int)
gk = _Am.gauKer(1)
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
iwis_cv    = _N.empty(len(partIDs))
itis_cv    = _N.empty(len(partIDs))
ilis_cv    = _N.empty(len(partIDs))
isis_lv    = _N.empty(len(partIDs))
isis_corr    = _N.empty(len(partIDs))
rsp_tms_cv    = _N.empty(len(partIDs))
coherence    = _N.empty(len(partIDs))
ages      = _N.empty(len(partIDs))
gens      = _N.empty(len(partIDs))
Engs      = _N.empty(len(partIDs))

amp_fluc12 =  _N.empty(len(partIDs))
amp_fluc13 =  _N.empty(len(partIDs))
amp_fluc23 =  _N.empty(len(partIDs))
pMimic_Beat =  _N.empty(len(partIDs))

corr_UD    = _N.empty((len(partIDs), 3))

cntrsDSUWTL = _N.empty((len(partIDs), 2))
cntrsRPSWTL = _N.empty((len(partIDs), 2))
cntrsDSURPS = _N.empty((len(partIDs), 2))

time_aft_los = _N.empty(len(partIDs))
time_aft_tie  = _N.empty(len(partIDs))
time_aft_win = _N.empty(len(partIDs))
score  = _N.empty(len(partIDs))
maxCs  = _N.empty(len(partIDs))
pcW_UD  = _N.empty(len(partIDs))
pcT_UD  = _N.empty(len(partIDs))
pcL_UD  = _N.empty(len(partIDs))

DSUWTL_corrs = _N.empty((36, len(partIDs)))
DSURPS_corrs = _N.empty((36, len(partIDs)))
RPSWTL_corrs = _N.empty((36, len(partIDs)))

up_cvs    = _N.empty(len(partIDs))
st_cvs    = _N.empty(len(partIDs))
dn_cvs    = _N.empty(len(partIDs))
R_cvs    = _N.empty(len(partIDs))
S_cvs    = _N.empty(len(partIDs))
P_cvs    = _N.empty(len(partIDs))

du_diffs = _N.empty(len(partIDs))

moresimV1  = _N.empty(len(partIDs))
moresimV2  = _N.empty(len(partIDs))
moresimV3  = _N.empty(len(partIDs))
moresimV4  = _N.empty(len(partIDs))
moresimST  = _N.empty(len(partIDs))
moresimSW  = _N.empty(len(partIDs))
moresim  = _N.empty(len(partIDs))
moresiment  = _N.empty(len(partIDs))
sum_sd_DSUWTL = _N.empty((len(partIDs), 3, 3))
sum_ent_DSUWTL = _N.empty((len(partIDs), 3, 3))

sum_mn = _N.empty((len(partIDs), 3, 3))
sum_mn_DSURPS = _N.empty((len(partIDs), 3, 3))
sum_sd_RPSWTL = _N.empty((len(partIDs), 3, 3))
sum_sd_DSURPS = _N.empty((len(partIDs), 3, 3))

sum_sd2 = _N.empty((len(partIDs), 3, 3))
sum_cv = _N.empty((len(partIDs), 3, 3))
marginalCRs = _N.empty((len(partIDs), 3, 3))
entropyDSU = _N.empty((len(partIDs), 3))
entropyDSUWTL_D = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropyDSUWTL_S = _N.empty(len(partIDs))
entropyDSUWTL_U = _N.empty(len(partIDs))

entropyRPSWTL_R = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropyRPSWTL_P = _N.empty(len(partIDs))
entropyRPSWTL_S = _N.empty(len(partIDs))

entropyDSURPS_D = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropyDSURPS_S = _N.empty(len(partIDs))
entropyDSURPS_U = _N.empty(len(partIDs))


#entropyUD2 = _N.empty(len(partIDs))
entropyS2 = _N.empty(len(partIDs))
entropyDr = _N.empty(len(partIDs))   #  how different are D across WTL conditions
entropySr = _N.empty(len(partIDs))
entropyUr = _N.empty(len(partIDs))
entropyDSUWTL_W = _N.empty(len(partIDs))   #  
entropyDSUWTL_T = _N.empty(len(partIDs))
entropyDSUWTL_L = _N.empty(len(partIDs))
entropyRPS1 = _N.empty(len(partIDs))   #  
entropyRPS2 = _N.empty(len(partIDs))
entropyRPS3 = _N.empty(len(partIDs))
entropyRPS_W = _N.empty(len(partIDs))   #  
entropyRPS_T = _N.empty(len(partIDs))
entropyRPS_L = _N.empty(len(partIDs))

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
stay_amps     = _N.empty((len(partIDs), 3))  #  for actions, conditions distinguished

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
gkISI = _Am.gauKer(1)
gkISI /= _N.sum(gkISI)

#  DISPLAYED AS R,S,P
#  look for RR RS RP
#  look for SR SS SP
#  look for PR PS PP

ths = _N.where((AQ28scrs > 35))[0]
L30  = 30
for partID in partIDs:
    pid += 1

    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit}))
    _prob_mvs = dmp["cond_probs"][SHF_NUM][:, strtTr:]
    _prob_mvsRPS = dmp["cond_probsRPS"][SHF_NUM][:, strtTr:]
    _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]    
    
    _prob_mvs_STSW = dmp["cond_probsSTSW"][SHF_NUM][:, strtTr:]
    inp_meth = dmp["inp_meth"]
    _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
    end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])

    hdcol = 0

    hnd_dat_all[pid-1] = _hnd_dat[0:TO]

    all_AI_weights[pid-1] = dmp["AI_weights"][0:TO+1]
    all_AI_preds[pid-1] = dmp["AI_preds"][0:TO+1]

    if look_at_AQ:
        ans_soc_skils[pid-1], ans_rout[pid-1], ans_switch[pid-1], ans_imag[pid-1], ans_fact_pat[pid-1] = _rt.AQ28ans("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})
        AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})
        ages[pid-1], gens[pid-1], Engs[pid-1] = _rt.Demo("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/DQ1.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})        
    
    _aift.resptime_aft_wtl(_hnd_dat, TO, pid, inp_meth, time_aft_win, time_aft_tie, time_aft_los)

    _aift.wtl_after_wtl(_hnd_dat, TO, pid, win_aft_win, tie_aft_win, los_aft_win, win_aft_tie, tie_aft_tie, los_aft_tie, win_aft_los, tie_aft_los, los_aft_los, R_aft_win, S_aft_win, P_aft_win, R_aft_tie, S_aft_tie, P_aft_tie, R_aft_los, S_aft_los, P_aft_los)
    #resptime_after_wtl()
    ####
    #x()

    cv_sum = 0
    dhd = _N.empty(TO)
    dhd[0:TO-1] = _N.diff(_hnd_dat[0:TO, 3])
    dhd[TO-1] = dhd[TO-2]
    #dhdr = dhd.reshape((20, 15))
    #rsp_tms_cv[pid-1] = _N.mean(_N.std(dhdr, axis=1) / _N.mean(dhdr, axis=1))
    
    #rsp_tms_cv[pid-1] = _N.std(_hnd_dat[:, 3]) / _N.mean(_hnd_dat[:, 3])
    marginalCRs[pid-1] = _emp.marginalCR(_hnd_dat)
    prob_mvs  = _prob_mvs[:, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPS  = _prob_mvsRPS[:, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSURPS  = _prob_mvsDSURPS[:, 0:TO - win]  #  is bigger than hand by win size        
    prob_mvs_STSW  = _prob_mvs_STSW[:, 0:TO - win]  #  is bigger than hand by win size    
    prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
    prob_mvs_RPS = prob_mvsRPS.reshape((3, 3, prob_mvsRPS.shape[1]))
    prob_mvs_DSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))
    prob_mvs_STSW = prob_mvs_STSW.reshape((3, 2, prob_mvs_STSW.shape[1]))
    #  _N.sum(prob_mvs_STSW[0], axis=0) = 1, 1, 1, 1, 1, 1, (except at ends)
    #dbehv = _crut.get_dbehv(prob_mvs, gk, equalize=True)
    #dbehv = _crut.get_dbehv(prob_mvs, gkISI, equalize=True)
    #dbehv_RPS = _crut.get_dbehv(prob_mvs_RPS, gkISI, equalize=True)
    #dbehv_DSURPS = _crut.get_dbehv(prob_mvs_DSURPS, gkISI, equalize=True)    
    dbehv_DSUWTL = _crut.get_dbehv(prob_mvs, None, equalize=True)
    dbehv_RPSWTL = _crut.get_dbehv(prob_mvs_RPS, None, equalize=True)
    dbehv_DSURPS = _crut.get_dbehv(prob_mvs_DSURPS, None, equalize=True)    

    tMv = _N.diff(_hnd_dat[:, 3])
    succ = _hnd_dat[1:, 2]

    #dbehv = dbehv + 0.115*dbehv_RPS# + 0.01*dbehv_DSURPS
    #dbehv = _N.convolve(dbehv + 0.115*dbehv_RPS, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    #dbehv = _N.convolve(dbehv + 0.5*dbehv_RPS, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    #dbehv = _N.convolve(dbehv_DSURPS, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    dbehv = _N.convolve(dbehv_DSUWTL + 0.5*dbehv_DSURPS, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    
    dbehv_DSURPS = _N.convolve(dbehv_DSURPS, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    dbehv_DSUWTL = _N.convolve(dbehv_DSUWTL, gkISI, mode="same")# + 0.01*dbehv_DSURPS    
    #dbehv = dbehv+dbehv_DSURPS
    #dbehv = _N.convolve(dbehv + 0.5*dbehv_RPS, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    #dbehv = _N.convolve(dbehv, gkISI, mode="same")# + 0.01*dbehv_DSURPS
    #pfrm_1st2nd[pid-1] = _N.mean(tMv[_N.where(succ == 1)]) - _N.mean(tMv[_N.where(succ == -1)])
    
    #fdbehv = _N.convolve(dbehv, gkISI, mode="same")
    maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2) #  3 from label71
    maxs_DSUWTL = _N.where((dbehv_DSUWTL[0:TO-11] >= 0) & (dbehv_DSUWTL[1:TO-10] < 0))[0] + (win//2)#  3 from label71
    maxs_DSURPS = _N.where((dbehv_DSURPS[0:TO-11] >= 0) & (dbehv_DSURPS[1:TO-10] < 0))[0] + (win//2)#  3 from label71

    preds = all_AI_preds[pid-1]
    
    PCS=6
    prob_Mimic            = _N.empty((3, prob_mvs.shape[2]))
    #sd_M[pid-1]               = _N.std(prob_mvs[0, 0] + prob_mvs[1, 1] + prob_mvs[2, 2])
    sd_M[pid-1]               = _N.std(prob_mvs[0, 0] + prob_mvs[2, 2])
    t00 = 5
    t01 = prob_mvs.shape[2]-5
    ctprob_mvs          = prob_mvs[:, :, t00:t01]
    pc_M1[pid-1],pv               = _ss.pearsonr(ctprob_mvs[0, 1], ctprob_mvs[1, 1])
    pc_M2[pid-1],pv               = _ss.pearsonr(ctprob_mvs[0, 1], ctprob_mvs[2, 1])
    pc_M3[pid-1],pv               = _ss.pearsonr(ctprob_mvs[1, 1], ctprob_mvs[2, 2])
    #

    prob_Mimic[0]      = prob_mvs[0, 0]   #  DN | WIN
    prob_Mimic[1]      = prob_mvs[1, 1]   #  ST | TIE
    prob_Mimic[2]      = prob_mvs[2, 2]   #  UP | LOS
    prob_Beat            = _N.empty((3, prob_mvs.shape[2]))
    prob_Beat[0]       = prob_mvs[0, 1]
    prob_Beat[1]       = prob_mvs[1, 2]
    prob_Beat[2]       = prob_mvs[2, 0]
    pMimic_Beat[pid-1] = _N.mean(_N.sum(prob_Mimic, axis=0)) - _N.mean(_N.sum(prob_Beat, axis=0))
    #entropyB[pid-1] = entropy3(prob_Beat.T, PCS)    
    #entropyM[pid-1] = entropy3(prob_Mimic.T, PCS)
    

    all_prob_mvs.append(prob_mvs)    #  plot out to show range of CRs

    # prob_pcs = _N.empty((len(maxs)-1, 3, 3))
    # for i in range(len(maxs)-1):
    #     prob_pcs[i] = _N.mean(prob_mvs[:, :, maxs[i]:maxs[i+1]], axis=2)
    #     #  _N.sum(prob_mvs[:, :, 10], axis=1) == [1, 1, 1]
    # all_prob_pcs.extend(prob_pcs)
    # istrtend += prob_pcs.shape[0]
    # strtend[pid-1+1] = istrtend

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

    ##  
    pUD_WTL = _N.array([ctprob_mvs[0, 0] + ctprob_mvs[0, 2],
                        ctprob_mvs[1, 0] + ctprob_mvs[1, 2],
                        ctprob_mvs[2, 0] + ctprob_mvs[2, 2]])
    pS_WTL  = _N.array([ctprob_mvs[0, 1], ctprob_mvs[1, 1], ctprob_mvs[2, 1]])
    #entsUD_S = _N.array([entropy3(pUD_WTL.T, PCS),
    #                     entropy3(pS_WTL.T,  PCS)])

    # entsDSUr = _N.array([entropy3(ctprob_mvs[:, 0].T, PCS, repeat=10, nz=0.1),
    #                      entropy3(ctprob_mvs[:, 1].T, PCS, repeat=10, nz=0.1),
    #                      entropy3(ctprob_mvs[:, 2].T, PCS, repeat=10, nz=0.1)])
    
    #entsSTSW = _N.array([entropy2(ctprob_mvs[:, 0].T, PCS), entropy3(ctprob_mvs[:, 1].T, PCS), entropy3(ctprob_mvs[:, 2].T, PCS)])

    pW_stsw = _N.array([ctprob_mvs[0, 0] + ctprob_mvs[0, 2], ctprob_mvs[0, 1]])
    pT_stsw = _N.array([ctprob_mvs[1, 0] + ctprob_mvs[1, 2], ctprob_mvs[1, 1]])
    pL_stsw = _N.array([ctprob_mvs[2, 0] + ctprob_mvs[2, 2], ctprob_mvs[2, 1]])


    #  Is TIE like a WIN or TIE like a LOSE?
    #  ENT_WT = entropy of (UP|WIN and UP|TIE) + entropy (DN|WIN and DN|TIE) + entropy (UP|WIN and UP|TIE)
    #  ENT_LT = entropy of (UP|LOS and UP|TIE) + entropy (DN|LOS and DN|TIE) + entropy (UP|LOS and UP|TIE)
    probU  = _N.empty((2, ctprob_mvs.shape[2]))
    probD  = _N.empty((2, ctprob_mvs.shape[2]))
    probS  = _N.empty((2, ctprob_mvs.shape[2]))
    probU[0] = ctprob_mvs[0, 2]
    probU[1] = ctprob_mvs[1, 2]
    probS[0] = ctprob_mvs[0, 1]
    probS[1] = ctprob_mvs[1, 1]    
    probD[0] = ctprob_mvs[0, 0]
    probD[1] = ctprob_mvs[1, 0]    

    #ENT_WT = entropy2(probU.T, PCS) + entropy2(probS.T, PCS) + entropy2(probD.T, PCS)
    ENT_WT = entropy2(probS.T, PCS)
    probU[0] = ctprob_mvs[2, 2]
    probU[1] = ctprob_mvs[1, 2]
    probS[0] = ctprob_mvs[2, 1]
    probS[1] = ctprob_mvs[1, 1]    
    probD[0] = ctprob_mvs[2, 0]
    probD[1] = ctprob_mvs[1, 0]    

    #ENT_LT = entropy2(probU.T, PCS) + entropy2(probS.T, PCS) + entropy2(probD.T, PCS)
    ENT_LT = entropy2(probS.T, PCS)
    #moresiment[pid-1] = ENT_WT - ENT_LT
    
    ctprob_mvs[0, 0]
    entsWTL2 = _N.array([entropy2(pW_stsw.T, PCS), entropy2(pT_stsw.T, PCS), entropy2(pL_stsw.T, PCS)])
    #entsWTL3 = _N.array([entropy3(ctprob_mvs[0].T, PCS),
    #                     entropy3(ctprob_mvs[1].T, PCS),
    #                     entropy3(ctprob_mvs[2].T, PCS)])
    
    probWst = ctprob_mvs[0, 1]
    probWsw = ctprob_mvs[0, 0] + ctprob_mvs[0, 2]
    datW    = _N.empty((ctprob_mvs.shape[2], 2))
    datW[:, 0] = probWst
    datW[:, 1] = probWsw
    probTst = ctprob_mvs[1, 1]
    probTsw = ctprob_mvs[1, 0] + ctprob_mvs[1, 2]
    datT    = _N.empty((ctprob_mvs.shape[2], 2))
    datT[:, 0] = probTst
    datT[:, 1] = probTsw
    probLst = ctprob_mvs[2, 1]
    probLsw = ctprob_mvs[2, 0] + ctprob_mvs[2, 2]
    datL    = _N.empty((ctprob_mvs.shape[2], 2))
    datL[:, 0] = probLst
    datL[:, 1] = probLsw

    #entsWTL = _N.array([entropy2(datW, PCS), entropy2(datT, PCS), entropy2(datL, PCS)])    

    # entropyDr[pid-1] = entsDSUr[0]
    # entropySr[pid-1] = entsDSUr[1]
    # entropyUr[pid-1] = entsDSUr[2]

    UD_diff[pid-1, 0] = _N.std(ctprob_mvs[0, 0] - ctprob_mvs[0, 2])
    UD_diff[pid-1, 1] = _N.std(ctprob_mvs[1, 0] - ctprob_mvs[1, 2])
    UD_diff[pid-1, 2] = _N.std(ctprob_mvs[2, 0] - ctprob_mvs[2, 2])

    entropyDSUWTL_D[pid-1], entropyDSUWTL_S[pid-1], entropyDSUWTL_U[pid-1] = _aift.entropyCRprobs(prob_mvs, fix="action", normalize=True, PCS=6, PCS1=10)
    entropyDSUWTL_W[pid-1], entropyDSUWTL_T[pid-1], entropyDSUWTL_L[pid-1] = _aift.entropyCRprobs(prob_mvs, fix="condition", normalize=True, PCS1=10)
    
    entropyRPSWTL_R[pid-1], entropyRPSWTL_S[pid-1], entropyRPSWTL_P[pid-1] = _aift.entropyCRprobs(prob_mvs_RPS, fix="action", normalize=True, PCS=PCS, PCS1=10)
    entropyDSURPS_D[pid-1], entropyDSURPS_S[pid-1], entropyDSURPS_U[pid-1] = _aift.entropyCRprobs(prob_mvs_DSURPS, fix="action", normalize=True, PCS=PCS, PCS1=10)    
    
    entropyW2[pid-1] = wtl_independent[0]
    entropyT2[pid-1] = wtl_independent[1]
    entropyL2[pid-1] = wtl_independent[2]
    actions_independent[pid-1] = wtl_independent                   #  3
    #cond_distinguished[pid-1] = condition_distinguished  #  2
    stay_amps[pid-1] = stay_amp     # 3 components

    THRisi = 2
    #isi   = _N.diff(maxs)

    # largeEnough = _N.where(_isi > THRisi)[0]
    # tooSmall    = _N.where(_isi <= 3)[0]
    # isi    = _isi[largeEnough]
    #pc, pv = _ss.pearsonr(isi[0:-1], isi[1:])

    #fisi = _N.convolve(isi, gkISI, mode="same")    

    #pc, pv = _ss.pearsonr(isi[0:-1], isi[1:])
    #fig = _plt.figure()
    #_plt.plot(fisi)
    #_plt.suptitle("%(1).3f    %(2).3f" % {"1" : pc, "2" : pc2})

    #_plt.savefig("isi%d" % (pid-1))
    #_plt.close()
    entDSUWTL = entropyDSUWTL_D[pid-1] + entropyDSUWTL_S[pid-1] + entropyDSUWTL_U[pid-1]
    entDSURPS = entropyDSURPS_D[pid-1] + entropyDSURPS_S[pid-1] + entropyDSURPS_U[pid-1]    
    if entDSUWTL > entDSURPS:
        isi   = cleanISI(_N.diff(maxs_DSUWTL), minISI=1)
    else:
        isi   = cleanISI(_N.diff(maxs_DSURPS), minISI=1)
    maxs = maxs_DSUWTL
    _aift.rulechange(_hnd_dat, signal_5_95, all_avgs, SHUFFLES, t0, t1, maxs, cut, pid)
    #isi   = cleanISI(_N.diff(maxs), minISI=2)
    pc, pv = rm_outliersCC_neighbors(isi[0:-1], isi[1:])    
    isis_corr[pid-1] = pc
    isis_sd[pid-1] = _N.std(isi)
    isis[pid-1] = _N.mean(isi)        
    isis_cv[pid-1] = isis_sd[pid-1] / isis[pid-1]
    isis_lv[pid-1] = (3/(len(isi)-1))*_N.sum((isi[0:-1] - isi[1:])**2 / (isi[0:-1] + isi[1:])**2 )    
    all_maxs.append(isi)    

    sdsDSUWTL = _N.std(ctprob_mvs, axis=2)
    for i in range(3):
        for j in range(3):
            sum_ent_DSUWTL[pid-1, i, j] = _aift.entropy1(ctprob_mvs[i, j], 10)
    sdsRPSWTL = _N.std(prob_mvs_RPS, axis=2)
    sdsDSURPS = _N.std(prob_mvs_DSURPS, axis=2)    

    #sds = _N.std(prob_pcs, axis=0)
    mns = _N.mean(ctprob_mvs, axis=2)
    mnsRPS = _N.mean(prob_mvs_RPS, axis=2)
    mnsDSURPS = _N.mean(prob_mvs_DSURPS, axis=2)    
    
    
    #sum_cv[pid-1] = sds/(1-_N.abs(0.5-mns))
    sum_sd_DSUWTL[pid-1] = sdsDSUWTL
    sum_mn[pid-1] = mns
    sum_mn_DSURPS[pid-1] = mnsDSURPS
    sum_sd_RPSWTL[pid-1] = sdsRPSWTL
    sum_sd_DSURPS[pid-1] = sdsDSURPS    
    score[pid-1] = _N.sum(_hnd_dat[:, 2])# / _hnd_dat.shape[0]

    #  DSUWTL_corrs[pid-1]
    #  DSUWTL_corrs = _N.empty((36, len(participants)))
    DSUWTL_corrs[:, pid-1]      = _aift.corr_btwn_probCRcomps(prob_mvs)
    DSURPS_corrs[:, pid-1]      = _aift.corr_btwn_probCRcomps(prob_mvs_DSURPS)
    RPSWTL_corrs[:, pid-1]      = _aift.corr_btwn_probCRcomps(prob_mvs_RPS)


    netwins[pid-1] = _N.sum(_hnd_dat[:, 2])
    wins = _N.where(_hnd_dat[:, 2] == 1)[0]
    losses = _N.where(_hnd_dat[:, 2] == -1)[0]
    perform[pid -1] = len(wins) / (len(wins) + len(losses))

    stayLs = []
    strt   = 0
    L      = 1
    for ti in range(TO-1):
        if _hnd_dat[ti, 0] == _hnd_dat[ti+1, 0]:
            L += 1
        else:
            stayLs.append(L)
            L = 1
    mn_stayL[pid-1] = _N.std(stayLs)
    maxs = maxs_DSUWTL

    cntrsDSUWTL[pid-1, 0], cntrsDSUWTL[pid-1, 1] = _aift.cntrmvs_DSUWTL(prob_mvs, TO)
    cntrsDSURPS[pid-1, 0], cntrsDSURPS[pid-1, 1] = _aift.cntrmvs_DSURPS(prob_mvs_DSURPS, TO)
    cntrsRPSWTL[pid-1, 0], cntrsRPSWTL[pid-1, 1] = _aift.cntrmvs_RPSWTL(prob_mvs_DSURPS, TO)

    _aift.action_result(_hnd_dat, TO, pid, u_or_d_res, dn_res, up_res, stay_res, u_or_d_tie, stay_tie)

    iwis = _N.diff(_N.where(_hnd_dat[:, 2] == 1)[0])
    iwis_cv[pid-1] = _N.std(iwis) / _N.mean(iwis)
    itis = _N.diff(_N.where(_hnd_dat[:, 2] == 0)[0])
    itis_cv[pid-1] = _N.std(itis) / _N.mean(itis)
    ilis = _N.diff(_N.where(_hnd_dat[:, 2] == -1)[0])
    ilis_cv[pid-1] = _N.std(ilis) / _N.mean(ilis)

#############  AI WEIGHTS
FEAT1, FEAT2, FEAT3, FEAT4, FEAT5, FEAT6, AIent1, AIent2, AIent3, AIent4, mn_diff_top2, sd_diff_top2, mnFt1, mnFt2, mnFt3, mnFt4, mnFt5, mnFt6, mnFt7, mnFt8, sdFt1, sdFt2, sdFt3, sdFt4, sdFt5, sdFt6, sdFt7, sdFt8 = _aift.perceptron_features(all_AI_weights, all_AI_preds, partIDs)


USDdiff0 = _N.std(marginalCRs, axis=2)[:, 0]   #  how different are USD in LOSE condition
USDdiff1 = _N.std(marginalCRs, axis=2)[:, 1]   #  how different are USD in LOSE condition
USDdiff2 = _N.std(marginalCRs, axis=2)[:, 2]   #  how different are USD in LOSE condition

USDdiff3 = _N.std(marginalCRs, axis=1)[:, 0]   #  how different are USD in LOSE condition
USDdiff4 = _N.std(marginalCRs, axis=1)[:, 1]   #  how different are USD in LOSE condition
USDdiff5 = _N.std(marginalCRs, axis=1)[:, 2]   #  how different are USD in LOSE condition


#entropyWL    = entropyW + entropyL
#sds20_m_sds22  = sds20 - sds22
#sds01_m_sds11  = sds01 - 0.2*sds11
#sds01_m_sds12  = sds01 - sds12
#for pid in range(len(partIDs)):
#    AIent2[pid] = entropy3(stg2[pid], 8)

################

# AIfts1allcomps = _N.std(_N.sum(sumAIw, axis=3), axis=1)
# AIfts1 = AIfts1allcomps[:, 0]
# AIfts2 = AIfts1allcomps[:, 1]
# AIfts3 = AIfts1allcomps[:, 2]

cntrsDSUWTL /= _N.sum(cntrsDSUWTL, axis=1).reshape(len(partIDs), 1)
cntrmvs_DSUWTL = cntrsDSUWTL[:, 1] - cntrsDSUWTL[:, 0]
cntrsDSURPS /= _N.sum(cntrsDSURPS, axis=1).reshape(len(partIDs), 1)
cntrmvs_DSURPS = cntrsDSURPS[:, 1] - cntrsDSURPS[:, 0]
cntrsRPSWTL /= _N.sum(cntrsRPSWTL, axis=1).reshape(len(partIDs), 1)
cntrmvs_RPSWTL = cntrsRPSWTL[:, 1] - cntrsRPSWTL[:, 0]

cntrmvsDIFF = cntrmvs_DSUWTL - cntrmvs_DSURPS
cntrmvsSUM  = cntrmvs_DSUWTL + cntrmvs_RPSWTL

#  sum_sd_RPSWTL
#  cntrsDSUWTL
#  mnFt1
#features_
#sumsdRPS0 = sum_sd_RPS[:, 2, 0]
#sumsdRPS1 = sum_sd_RPS[:, 2, 1]
#sumsdRPS2 = sum_sd_RPS[:, 2, 2]   #  sds02  sumsdRPS0

# diff_sd_RPS_DSU = _N.sum(_N.sum(sum_sd_RPS - sum_sd, axis=2), axis=1)
# diff_sds1 = sum_sd[:, 2, 0] - sum_sd_RPS[:, 2, 0]
# diff_sds2 = sum_sd[:, 2, 2] - sum_sd_RPS[:, 2, 2]
#  More sim:  If large
# features_cab2 = ["isis", "isis_cv", "isis_corr", "isis_lv",
#                 "entropyD", "entropyS", "entropyU",
#                 "entropyB", "entropyM",                  
#                 "entropyT2", "entropyW2", "entropyL2",
#                  "entropyT", "entropyW", "entropyL", "entropyWL",
#                  "entropyRPS123", "entropyRPS312", 
#                 #"mn00", "mn01", "mn02",
#                 #"mn10", "mn11", "mn12",
#                 #"mn20", "mn21", "mn22",                
#                 #"sd_M", "sd_BW", "sd_BW2", "sd_BT", "sd_BL", "sd_MW", "sd_MT", "sd_ML",
#                 "pc_M1", "pc_M2", "pc_M3", "pfrm_change69", "USDdiff0", "USDdiff1", "USDdiff2", "USDdiff3", "USDdiff4", "USDdiff5",
#                 #"pc0220s", "pc0110s", "pc0010s",
#                 #"up_cvs", "dn_cvs", "st_cvs",
#                 #"R_cvs", "S_cvs", "P_cvs",                
#                 "sumsdRPS0", "sumsdRPS1", "sumsdRPS2",
#                 "pc0010s", "pc2122s", "pc0110s", "pc0220s",
#                  "diff_sd_RPS_DSU",
#                  "DSURPSpc0010s", "DSURPSpc0012s","DSURPSpc0110s",
#                  "DSURPSpc0221s", "DSURPSpc1011s",
#                  "diff_sds1", "diff_sds2",
#                  "cntrmvs", "sd_diff_top2"]

features_cab2    = ["sum_sd_DSUWTL", "sum_sd_DSURPS", "sum_sd_RPSWTL",
                    "DSUWTL_corrs", "DSURPS_corrs", "RPSWTL_corrs",
                    "cntrmvsDIFF", "cntrmvsSUM"]

features_cab1 = []
#features_cab1 = []#"sds00", "sds01", "sds02",
                 #"sds10", "sds11", "sds12",
                 #"sds20", "sds21", "sds22"]#, "sds20_m_sds22", "sds01_m_sds11",
#                 "sds01_m_sds12"]

features_AI  = ["mnFt1", "mnFt2", "mnFt3", "mnFt4", "mnFt5", "mnFt6", "mnFt7", "mnFt8", "sdFt1", "sdFt2", "sdFt3", "sdFt4", "sdFt5", "sdFt6", "sdFt7", "sdFt8"]

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
                "R_aft_win", "R_aft_tie", "R_aft_los",
                "S_aft_win", "S_aft_tie", "S_aft_los",
                "P_aft_win", "P_aft_tie", "P_aft_los"]                

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
dmp_dat["smth"] = smth
dmp_dat["netwins"] = netwins
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


dmpout = open("predictAQ28dat/AQ28_vs_RPS_%(v)d_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "wt" : win_type, "w" : win, "s" : smth}, "wb")
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

for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
    exec("tar = %s" % star)
    pcW, pvW = rm_outliersCC_neighbors(tar[ths], time_aft_win[ths])
    pcT, pvT = rm_outliersCC_neighbors(tar[ths], time_aft_tie[ths])
    pcL, pvL = rm_outliersCC_neighbors(tar[ths], time_aft_los[ths])
    # pcW, pvW = _ss.pearsonr(tar[ths], time_aft_win[ths])
    # pcT, pvT = _ss.pearsonr(tar[ths], time_aft_tie[ths])
    # pcL, pvL = _ss.pearsonr(tar[ths], time_aft_los[ths])
    fig = _plt.figure(figsize=(8, 3))
    _plt.suptitle(star)
    fig.add_subplot(1, 3, 1)
    _plt.scatter(time_aft_win[ths], tar[ths])
    fig.add_subplot(1, 3, 2)
    _plt.scatter(time_aft_tie[ths], tar[ths])
    fig.add_subplot(1, 3, 3)
    _plt.scatter(time_aft_los[ths], tar[ths])
    print(star)
    print("%(pc).3f  %(pv).3f" % {"pc" : pcW, "pv" : pvW})
    print("%(pc).3f  %(pv).3f" % {"pc" : pcT, "pv" : pvT})
    print("%(pc).3f  %(pv).3f" % {"pc" : pcL, "pv" : pvL})    


entropyDSUWTL = entropyDSUWTL_D + entropyDSUWTL_S + entropyDSUWTL_U
entropyRPSWTL = entropyRPSWTL_R + entropyRPSWTL_S + entropyRPSWTL_P
entropyDSURPS = entropyDSURPS_D + entropyDSURPS_S + entropyDSURPS_U


shuffle_scrs = False

sths = _N.array(ths)
if shuffle_scrs:
    _N.random.shuffle(sths)

print("+++++++++++corr between entropy")
for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
    exec("tar = %s" % star)
    #pc, pv = _ss.pearsonr((s1-s3)[ths], tar[sths])
    pc, pv = _ss.pearsonr((entropyDSUWTL - entropyDSURPS)[ths], tar[sths])
    #if _N.abs(pc) > 0.15:
    print("%(st)s   %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "st" : star})

print("+++++++++++corr between entropy")
for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
    exec("tar = %s" % star)
    pc, pv = _ss.pearsonr((entropyDSUWTL_T - entropyDSUWTL_L)[ths], tar[sths])
    if _N.abs(pc) > 0.15:
        print("AQ28scrs   %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv})
        
print("+++++++++++corr between CR prob components")
for mdl in ["DSUWTL", "RPSWTL", "DSURPS"]:
    exec("model = %s_corrs" % mdl)
    print("----------------   %s" % mdl)
    for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        exec("tar = %s" % star)
        print("!!!!!  %s" % star)
        for ic in range(36):
            pc, pv = _ss.pearsonr(model[ic, ths], tar[sths])
            if _N.abs(pc) > 0.15:
                print("ic %(ic)d    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ic" : ic})
                
print("..........  sd of components")
#sum
for mdl in ["DSUWTL", "RPSWTL", "DSURPS"]:
    exec("model = sum_sd_%s" % mdl)
    print("----------------   %s" % mdl)
    for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        exec("tar = %s" % star)
        print("!!!!!  %s" % star)
        for ic1 in range(3):
            for ic2 in range(3):        
                pc, pv = _ss.pearsonr(model[ths, ic1, ic2], tar[sths])
                if _N.abs(pc) > 0.15:
                    print("ic %(ic)d    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ic" : ic})

""" 
print("..........  sent1 of components")
#sum
for mdl in ["DSUWTL", "RPSWTL", "DSURPS"]:
    exec("model = sum_ent_%s" % mdl)
    print("----------------   %s" % mdl)
    for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        exec("tar = %s" % star)
        print("!!!!!  %s" % star)
        for ic1 in range(3):
            for ic2 in range(3):        
                pc, pv = _ss.pearsonr(model[ths, ic1, ic2], tar[sths])
                if _N.abs(pc) > 0.15:
                    print("ic %(ic)d    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ic" : ic})


print("..........   AI entropy")
#sum
for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
    exec("tar = %s" % star)
    print("!!!!!  %s" % star)
    pc, pv = _ss.pearsonr(AIent1[ths], tar[sths])
    #if _N.abs(pc) > 0.15:
    print("ic %(ic)d    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ic" : ic})


        
                    
for ud in range(6):
    exec("usd = USDdiff%d" % ud)
    for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        exec("tar = %s" % star)
        print("!!!!!  %s" % star)
        pc, pv = _ss.pearsonr(usd[ths], tar[sths])
        if _N.abs(pc) > 0.15:
            print("ic %(ic)d    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ic" : ic})
    
print("interval statistics")
for sud in ["isis", "isis_corr", "isis_cv", "isis_lv"]:
    print("int stat   %s" % sud)
    exec("ist_ud = %s" % sud)
    for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        exec("tar = %s" % star)
        print("!!!!!  %s" % star)
        pc, pv = _ss.pearsonr(ist_ud[ths], tar[sths])
        #if _N.abs(pc) > 0.15:
        print("%(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv})
    
ths = _N.where((AQ28scrs > 35))[0]            
# """

ths = _N.where((AQ28scrs > 35))[0]
sths = _N.array(ths)
#_N.random.shuffle(sths)

for sud in ["FEAT1", "FEAT2", "FEAT3", "FEAT4", "FEAT5", "AIent1", "AIent2", "AIent3", "AIent4", "mn_diff_top2", "sd_diff_top2"]:
    exec("ist_ud = %s" % sud)
    for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        exec("tar = %s" % star)
        print("!!!!!  %s" % star)
        pc, pv = _ss.pearsonr(ist_ud[ths], tar[sths])
        if _N.abs(pc) > 0.18:
            print("%(ft)s    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ft" : sud})
            fig = _plt.figure()
            _plt.suptitle("%(ft)s  %(tar)s    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ft" : sud, "tar" : star})
            _plt.scatter(ist_ud[ths], tar[sths])

for sud in ["mnFt1", "mnFt2", "mnFt3", "mnFt4", "mnFt5", "mnFt6", "mnFt7", "mnFt8", "sdFt1", "sdFt2", "sdFt3", "sdFt4", "sdFt5", "sdFt6", "sdFt7", "sdFt8"]:
    for cmp in range(3):
        sudcmp = "%(ft)s[:, %(c)d]" % {"ft" : sud, "c" : cmp}
        exec("ist_ud = %s" % sudcmp)
        for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
            exec("tar = %s" % star)
            print("!!!!!  %s" % star)
            pc, pv = _ss.pearsonr(ist_ud[ths], tar[sths])
            if _N.abs(pc) > 0.18:
                print("%(ft)s    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ft" : sud})
                fig = _plt.figure()
                _plt.suptitle("%(ft)s  %(tar)s    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ft" : sud, "tar" : star})
                _plt.scatter(ist_ud[ths], tar[sths])
