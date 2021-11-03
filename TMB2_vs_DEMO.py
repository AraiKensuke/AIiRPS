#!/usr/bin/python

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

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

        
data   = "TMB2"
if data == "TMB1":
    dates = _rt.date_range(start='7/13/2021', end='11/10/2021')
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_ONLY_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=400, maxIGI=30000, MinWinLossRat=0.1, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=4)
if data == "TMB2":
    dates = _rt.date_range(start='7/13/2021', end='11/10/2021')    
    partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=800, maxIGI=30000, MinWinLossRat=0.3, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)


pid = 0

netwins  = _N.empty((4, len(partIDs)))

ages      = _N.empty(len(partIDs))
gens      = _N.empty(len(partIDs))
Engs      = _N.empty(len(partIDs))

pid = 0
for partID in partIDs:
    pid += 1
    age, gen, Eng = _rt.Demo("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/DQ1.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data})
    ages[pid-1] = age
    gens[pid-1] = gen
    Engs[pid-1] = Eng

    
for blk in range(1):
    pid = 0
    for partID in partIDs:
        pid += 1
        _hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB2", block=(blk+1))

        netwins[blk, pid-1] = _N.sum(_hnd_dat[:, 2])

lblsz=15
tksz=13


for blk in range(1):
    fig = _plt.figure(figsize=(4, 9))
    _plt.suptitle(cnstrs[partIDs[0]][blk])
    ###################    
    fig.add_subplot(3, 1, 1)
    _plt.xlabel("age", fontsize=lblsz)
    _plt.ylabel("netwins", fontsize=lblsz)
    _plt.xticks(fontsize=tksz)
    _plt.yticks(fontsize=tksz)
    answd = _N.where(ages > 0)[0]
    _plt.scatter(ages[answd] + 0.1*_N.random.randn(len(answd)), netwins[blk][answd], color="black", s=6)
    pc, pv = _ss.pearsonr(ages[answd], netwins[blk][answd])
    _plt.title("%(pc).2f   pv<%(pv).1e" % {"pc" : pc, "pv" : pv})
    ###################        
    fig.add_subplot(3, 1, 2)
    _plt.xlabel("gender", fontsize=lblsz)
    _plt.ylabel("netwins", fontsize=lblsz)
    _plt.xticks([0, 1, 2], ["M", "F", "NB"], fontsize=tksz)
    _plt.yticks(fontsize=tksz)
    answd = _N.where(gens > -1)[0]
    _plt.scatter(gens[answd] + 0.1*_N.random.randn(len(answd)), netwins[blk][answd], color="black", s=6)
    pc, pv = _ss.pearsonr(gens[answd], netwins[blk][answd])
    _plt.title("%(pc).2f   pv<%(pv).1e" % {"pc" : pc, "pv" : pv})
    ###################            
    fig.add_subplot(3, 1, 3)
    _plt.xlabel("Engs", fontsize=lblsz)
    _plt.ylabel("netwins", fontsize=lblsz)
    _plt.xticks([0, 1], ["No", "Yes"], fontsize=tksz)
    _plt.yticks(fontsize=tksz)
    answd = _N.where(Engs > -1)[0]
    _plt.scatter(Engs[answd] + 0.1*_N.random.randn(len(answd)), netwins[blk][answd], color="black", s=6)
    pc, pv = _ss.pearsonr(Engs[answd], netwins[blk][answd])
    _plt.title("%(pc).2f   pv<%(pv).1e" % {"pc" : pc, "pv" : pv})
    
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    _plt.savefig("DEMO_vs_performance_blk_%d" % blk)
