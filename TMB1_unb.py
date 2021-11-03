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

#dates = _rt.date_range(start='7/13/2021', end='8/17/2021')
dates = _rt.date_range(start='8/18/2021', end='11/30/2021')

nicknames = {"FixedSequence(__moRSP__, " +
             "[1, 1, 2, 1, 3, 1, 1, 1, 1, 3, " +
             "1, 1, 2, 1, 1, 1, 1, 1, 1, 3, " + 
             "1, 1, 2, 1, 1, 3, 1, 2, 1, 1, " + 
             "2, 1, 1, 1, 3, 1, 1, 1, 1, 1]);" : "Biased_Random",
             #
             "FixedSequencen(__moRSP__, " +
              "[3, 1, 2, 3, 2, 1, 2, 3, 3, 1, " +
             "1, 1, 2, 1, 3, 3, 2, 1, 2, 3, " + 
             "3, 1, 2, 1, 2, 1, 3, 2, 2, 3, " + 
             "2, 1, 3, 3, 2, 2, 3, 1, 3, 1]);" : "Unbiased_Random",
             #
             "WTL(__moRSP__, " +
             "[0.05, 0.85, 0.1], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3], false);"
             : "Exploitable_Win",
             #
             "Mimic(__moRSP__, 0, 0.2);" : "Mimic"}

#a = _N.array([3, 1, 2, 3, 2, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 3, 2, 1, 2, 3, 3, 1, 2, 1, 2, 1, 3, 2, 2, 3, 2, 1, 3, 3, 2, 2, 3, 1, 3, 1])

#A = _N.array([3, 1, 2, 3, 2, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 3, 2, 1, 2, 3, 3, 1, 2, 1, 2, 1, 3, 2, 2, 3, 2, 1, 3, 3, 2, 2, 3, 1, 3, 1])

#partIDs, dats, cnstrs = _rt.filterRPSdats("TMB1", dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=100, maxIGI=30000, MinWinLossRat=0.1, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=4)

partIDs, dats, cnstrs = _rt.filterRPSdats("TMB1", dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=300, MinWinLossRat=0.1, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=4)


#  first, let's just list out the types of constructors each

lblsz=15
tksz=14
titsz=17

nDataSets = len(partIDs)

dat_by_constr = {}
cnstrs_list   = []

s_bia_rand = "FixedSequence(__moRSP__, [1, 1, 2, 1, 3, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1]);"
s_unb_rand = "FixedSequence(__moRSP__, [3, 1, 2, 3, 2, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 3, 2, 1, 2, 3, 3, 1, 2, 1, 2, 1, 3, 2, 2, 3, 2, 1, 3, 3, 2, 2, 3, 1, 3, 1]);"
s_wtl_rand = "WTL(__moRSP__, [0.05, 0.85, 0.1], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3], false);"

pid = 0

all_bia_dat = _N.empty((nDataSets, 40, 4), dtype=_N.int)
all_unbr_dat = _N.empty((nDataSets, 40, 4), dtype=_N.int)
all_wtl_dat  = _N.empty((nDataSets, 40, 4), dtype=_N.int)


netwin_aft_AI_win_wtl  = _N.empty(nDataSets, dtype=_N.int)
netwin_aft_AI_tie_wtl  = _N.empty(nDataSets, dtype=_N.int)
netwin_aft_AI_los_wtl  = _N.empty(nDataSets, dtype=_N.int)

for partID in partIDs:
    pid += 1
    for blk in range(4):
        _hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB1", block=(blk+1))
        if cnstr == s_unb_rand:
            all_unbr_dat[pid-1] = _hnd_dat
        if cnstr == s_wtl_rand:
            all_wtl_dat[pid-1] = _hnd_dat
        if cnstr == s_bia_rand:
            all_bia_dat[pid-1] = _hnd_dat

    AI_wins = _N.where(all_wtl_dat[pid-1, :, 2] == -1)[0]    #  these are 
    to   = len(AI_wins)-1 if (AI_wins[-1] == 39) else len(AI_wins)
    HP_win_aft_AI_wins = _N.where(all_wtl_dat[pid-1, AI_wins[0:to]+1, 2] == 1)[0]
    HP_los_aft_AI_wins = _N.where(all_wtl_dat[pid-1, AI_wins[0:to]+1, 2] == -1)[0]    
    #print("%(w)d    %(l)d" % {"w" : len(HP_win_aft_AI_wins), "l" : len(HP_los_aft_AI_wins)})
    print("%d" % (len(HP_win_aft_AI_wins) - len(HP_los_aft_AI_wins)))
    netwin_aft_AI_win_wtl[pid-1]  =  len(HP_win_aft_AI_wins) - len(HP_los_aft_AI_wins)

    AI_loss = _N.where(all_wtl_dat[pid-1, :, 2] == 1)[0]    #  these are 
    to   = len(AI_loss)-1 if (AI_loss[-1] == 39) else len(AI_loss)
    HP_win_aft_AI_loss = _N.where(all_wtl_dat[pid-1, AI_loss[0:to]+1, 2] == 1)[0]
    HP_los_aft_AI_loss = _N.where(all_wtl_dat[pid-1, AI_loss[0:to]+1, 2] == -1)[0]    
    #print("%(w)d    %(l)d" % {"w" : len(HP_win_aft_AI_wins), "l" : len(HP_los_aft_AI_wins)})
    print("%d" % (len(HP_win_aft_AI_loss) - len(HP_los_aft_AI_loss)))
    netwin_aft_AI_los_wtl[pid-1]  =  len(HP_win_aft_AI_loss) - len(HP_los_aft_AI_loss)

    AI_ties = _N.where(all_wtl_dat[pid-1, :, 2] == 0)[0]    #  these are 
    to   = len(AI_ties)-1 if (AI_ties[-1] == 39) else len(AI_ties)
    HP_win_aft_AI_ties = _N.where(all_wtl_dat[pid-1, AI_ties[0:to]+1, 2] == 1)[0]
    HP_los_aft_AI_ties = _N.where(all_wtl_dat[pid-1, AI_ties[0:to]+1, 2] == -1)[0]    
    #print("%(w)d    %(l)d" % {"w" : len(HP_win_aft_AI_wins), "l" : len(HP_los_aft_AI_wins)})
    print("%d" % (len(HP_win_aft_AI_ties) - len(HP_los_aft_AI_ties)))
    netwin_aft_AI_tie_wtl[pid-1]  =  len(HP_win_aft_AI_ties) - len(HP_los_aft_AI_ties)
    
