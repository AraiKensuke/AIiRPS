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

__DSUWTL__ = 0
__RPSWTL__ = 1
__DSURPS__ = 2
__ALL__    = 3

mode       = __ALL__
#mode       = __DSUWTL__
#mode       = __RPSWTL__
#mode       = __DSURPS__

__1st__ = 0
__2nd__ = 1

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1

#  sum_sd
#  entropyL
#  isi_cv, isis_corr

def show(p1, p2, shf):
    fig = _plt.figure()
    avg = _N.mean(rc_trg_avg[p1:p2, :, shf], axis=0)
    for ipid in range(p1, p2):
        _plt.plot(rc_trg_avg[ipid, :, shf], color="grey")
    _plt.plot(avg, color="black")        

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
##  Then I expect wins following UPs and DOWNs to also be correlated to AQ28
look_at_AQ = True
data   = "TMB2"

#visit = 2
#visits= [1, 2]   #  if I want 1 of [1, 2], set this one to [1, 2]
visit = 1
visits= [1, ]   #  if I want 1 of [1, 2], set this one to [1, 2]
    
if data == "TMB2":
    dates = _rt.date_range(start='7/13/2021', end='12/30/2021')
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=15000, minIGI=20, maxIGI=30000, MinWinLossRat=0.35, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=800, max_meanIGI=8000, minIGI=200, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    ####  use this for reliability
    #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_AND_FALSE_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=8000, minIGI=50, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

A1 = []
show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

win_type = 2   #  window is of fixed number of games
#win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
label          = win_type*100+win*10+smth
TO = 300
SHF_NUM = 1

partIDs, incmp_dat = only_complete_data(partIDs, TO, label, SHF_NUM)
strtTr=0
TO -= strtTr

#fig= _plt.figure(figsize=(14, 14))

SHUFFLES = 200
nMimics = _N.empty(len(partIDs), dtype=_N.int)
t0 = -5
t1 = 10
trigger_temp = _N.empty(t1-t0)
cut = 1
all_avgs = _N.empty((len(partIDs), SHUFFLES+1, t1-t0))
netwins  = _N.empty(len(partIDs), dtype=_N.int)
gk = _Am.gauKer(2)
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

pfrm_change36 = _N.empty(len(partIDs))
pfrm_change69 = _N.empty(len(partIDs))
pfrm_change912= _N.empty(len(partIDs))


all_maxs  = []

aboves = []
belows = []

AQ28scrs  = _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))

all_prob_mvs = []
all_prob_pcs = []
istrtend     = 0
strtend      = _N.zeros(len(partIDs)+1, dtype=_N.int)

incomplete_data = []
gkISI = _Am.gauKer(2)
gkISI /= _N.sum(gkISI)

RPS_ratios = _N.empty((len(partIDs), 3))
RPS_ratiosMet = _N.empty(len(partIDs))

#  DISPLAYED AS R,S,P
#  look for RR RS RP
#  look for SR SS SP
#  look for PR PS PP


L30  = 30

rc_trg_avg = _N.empty((len(partIDs), 15, SHUFFLES+1))
rc_trg_avg_RPS = _N.empty((len(partIDs), 15, SHUFFLES+1))
rc_trg_avg_DSURPS = _N.empty((len(partIDs), 15, SHUFFLES+1))

chg = _N.empty(len(partIDs))

n_maxes   = _N.zeros((len(partIDs), SHUFFLES+1), dtype=_N.int)

# mdl, SHUFFLES, cond, act
stds        = _N.zeros((len(partIDs), 3, SHUFFLES+1, 3, 3, ))
# mdl, 1st hlf, 2nd hlf, SHUFFLES cond, act
stds12      = _N.zeros((len(partIDs), 3, 2, SHUFFLES+1, 3, 3))

#stds      = _N.zeros((len(partIDs), 3, SHUFFLES+1))
#stdsDSUWTL      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))
#stdsRPSWTL      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))
#stdsDSURPS      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))

marginalCRs = _N.empty((len(partIDs), SHUFFLES+1, 3, 3))
for partID in partIDs:
    pid += 1
    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit}))

    AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})

    _prob_mvs = dmp["cond_probs"][:, :, strtTr:]
    _prob_mvsRPS = dmp["cond_probsRPS"][:, strtTr:]
    _prob_mvsDSURPS = dmp["cond_probsDSURPS"][:, strtTr:]
    prob_mvs  = _prob_mvs[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPS  = _prob_mvsRPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSURPS  = _prob_mvsDSURPS[:, :, 0:TO - win]  #  is bigger than hand by win size

    #stds_all_mdls[0] = _N.std(prob_mvs, axis=2)
    
    for SHF_NUM in range(SHUFFLES+1):
        _prob_mvs = dmp["cond_probs"][SHF_NUM][:, strtTr:]
        _prob_mvsRPS = dmp["cond_probsRPS"][SHF_NUM][:, strtTr:]
        _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]    

        _prob_mvs_STSW = dmp["cond_probsSTSW"][SHF_NUM][:, strtTr:]    
        _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
        #end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])
        

        hdcol = 0

        inds =_N.arange(_hnd_dat.shape[0])
        hnd_dat_all[pid-1] = _hnd_dat[0:TO]

        cv_sum = 0
        dhd = _N.empty(TO)
        dhd[0:TO-1] = _N.diff(_hnd_dat[0:TO, 3])
        dhd[TO-1] = dhd[TO-2]
        #dhdr = dhd.reshape((20, 15))
        #rsp_tms_cv[pid-1] = _N.mean(_N.std(dhdr, axis=1) / _N.mean(dhdr, axis=1))


        #rsp_tms_cv[pid-1] = _N.std(_hnd_dat[:, 3]) / _N.mean(_hnd_dat[:, 3])
        #marginalCRs[pid-1] = _emp.marginalCR(_hnd_dat)
        prob_mvs  = _prob_mvs[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsRPS  = _prob_mvsRPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsDSURPS  = _prob_mvsDSURPS[:, 0:TO - win]  #  is bigger than hand by win size        
        prob_mvs_STSW  = _prob_mvs_STSW[:, 0:TO - win]  #  is bigger than hand by win size    
        prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
        prob_mvs_RPS = prob_mvsRPS.reshape((3, 3, prob_mvsRPS.shape[1]))
        prob_mvs_DSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))

        marginalCRs[pid-1, SHF_NUM] = _emp.marginalCR(_hnd_dat)
        N = prob_mvs.shape[2]
        dbehv    = _crut.get_dbehv_combined([prob_mvs_DSURPS, prob_mvs_RPS, prob_mvs], gkISI, equalize=True)

        maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2)#  3 from label71

        n_maxes[pid-1, SHF_NUM] = len(maxs)
        #print("%(sh)d   %(lm)d" % {"sh" : SHF_NUM, "lm" : len(maxs)})
        #mn_stayL[pid-1] = _N.std(stayLs)
        for sh in range(1):
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
                    try:
                        avgs[im-1, :] = hnd_dat[maxs[im]+t0:maxs[im]+t1, 2]
                    except ValueError:
                        print("*****  %(1)d  %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
                        print(avgs[im-1, :].shape)
                        print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2])


            all_avgs[pid-1, sh] = _N.mean(avgs, axis=0)
            #fig.add_subplot(5, 5, pid)
            #_plt.plot(_N.mean(avgs, axis=0))

        #srtd   = _N.sort(all_avgs[pid-1, 1:], axis=0)
        #signal_5_95[pid-1, 1] = srtd[int(0.05*SHUFFLES)]
        #signal_5_95[pid-1, 2] = srtd[int(0.95*SHUFFLES)]
        signal_5_95[pid-1, 0] = all_avgs[pid-1, 0]
        signal_5_95[pid-1, 3] = (signal_5_95[pid-1, 0] - signal_5_95[pid-1, 1]) / (signal_5_95[pid-1, 2] - signal_5_95[pid-1, 1])

        #pfrm_change36[pid-1] = _N.max(signal_5_95[pid-1, 0, 3:6]) - _N.min(signal_5_95[pid-1, 0, 3:6])

        sInds = _N.argsort(signal_5_95[pid-1, 0, 6:9])
        #sInds = _N.argsort(signal_5_95[pid-1, 0, 5:10])
        if sInds[2] - sInds[0] > 0:
            m69 = 1
        else:
            m69 = -1

        imax69 = _N.argmax(signal_5_95[pid-1, 0, 6:9])+6
        imin69 = _N.argmin(signal_5_95[pid-1, 0, 6:9])+6
        if SHF_NUM == 0:
            chg[pid-1] = _N.mean(signal_5_95[pid-1, 0, 7:11]) - _N.mean(signal_5_95[pid-1, 0, 4:8])
            #pfrm_change69[pid-1] = signal_5_95[pid-1, 0, imax69] - signal_5_95[pid-1, 0, imin69]
            pfrm_change69[pid-1] = _N.mean(signal_5_95[pid-1, 0, 7:11]) - _N.mean(signal_5_95[pid-1, 0, 5:9])

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

        #prob_mvs = lm["all_prob_mvsA"][ip]

        cntr = 0
        n_cntr = 0
        maxp_chg_times_wtl = []

        rc_trg_avg[pid-1, :, SHF_NUM] = signal_5_95[pid-1, 0]

popmn_rc_trg_avg = _N.mean(rc_trg_avg, axis=0)        


#  p(UP | W)
#  a big change means lots of UP | W
fig  = _plt.figure(figsize=(8, 4))
for sh in range(SHUFFLES):
    _plt.plot(popmn_rc_trg_avg[:, sh+1], color="grey", lw=1)
_plt.plot(popmn_rc_trg_avg[:, 0], color="black", lw=3)
_plt.xticks(_N.arange(15), _N.arange(-7, 8), fontsize=15)
_plt.yticks(fontsize=15)
_plt.axvline(x=7, ls=":", color="grey")
_plt.xlabel("lags from rule change (# games)", fontsize=18)
_plt.ylabel("p(WIN, lag) - p(LOS, lag)", fontsize=18)
fig.subplots_adjust(bottom=0.15, left=0.15)
_plt.xlim(0, 14)
#_plt.ylim(-0.1, 0.1)
_plt.savefig("Rulechange")

