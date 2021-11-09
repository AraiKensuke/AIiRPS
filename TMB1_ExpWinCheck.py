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

partIDs, dats, cnstrs = _rt.filterRPSdats("TMB1", dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=400, MinWinLossRat=0., has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=4, ngames=40)


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
s_mim_rand = "Mimic(__moRSP__, 0, 0.2);"

pid = 0

all_bia_dat = _N.empty((nDataSets, 40, 4), dtype=_N.int)
all_unbr_dat = _N.empty((nDataSets, 40, 4), dtype=_N.int)
all_wtl_dat  = _N.empty((nDataSets, 40, 4), dtype=_N.int)
all_mim_dat  = _N.empty((nDataSets, 40, 4), dtype=_N.int)


frmgm = 10
netwin_aft_AI_win_wtl  = _N.empty(nDataSets, dtype=_N.int)
netwin_aft_AI_tie_wtl  = _N.empty(nDataSets, dtype=_N.int)
netwin_aft_AI_los_wtl  = _N.empty(nDataSets, dtype=_N.int)
hlf12_wtl  = _N.empty(nDataSets, dtype=_N.int)
hlf12_bia  = _N.empty(nDataSets, dtype=_N.int)
hlf12_unbr  = _N.empty(nDataSets, dtype=_N.int)
hlf12_mim  = _N.empty(nDataSets, dtype=_N.int)
netwin_wtl  = _N.empty(nDataSets, dtype=_N.int)
netwin_bia  = _N.empty(nDataSets, dtype=_N.int)
netwin_unbr  = _N.empty(nDataSets, dtype=_N.int)
netwin_mim  = _N.empty(nDataSets, dtype=_N.int)

for partID in partIDs:
    pid += 1
    for blk in range(4):
        _hnd_dat, start_time, end_time, UA, cnstr, inpmeth, none1, none2            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB1", block=(blk+1))
        if cnstr == s_unb_rand:
            all_unbr_dat[pid-1] = _hnd_dat
        if cnstr == s_wtl_rand:
            all_wtl_dat[pid-1] = _hnd_dat
        if cnstr == s_bia_rand:
            all_bia_dat[pid-1] = _hnd_dat
        if cnstr == s_mim_rand:
            all_mim_dat[pid-1] = _hnd_dat

frm        = 32
netwin_wtl = _N.sum(all_wtl_dat[:, frm:, 2], axis=1)
netwin_bia = _N.sum(all_bia_dat[:, frm:, 2], axis=1)
netwin_unb = _N.sum(all_unbr_dat[:, frm:, 2], axis=1)
netwin_mim = _N.sum(all_mim_dat[:, frm:, 2], axis=1)

N    = len(partIDs)
dats = [netwin_wtl, netwin_mim, netwin_bia, netwin_unb]
sdats = ["ExW", "Mim", "BR", "UNB"]
fig = _plt.figure(figsize=(8, 8))

pcpvs = _N.empty((6, 2))

ij    = -1
for i in range(4):
    dat_i = dats[i]
    for j in range(i+1, 4):
        ij += 1
        dat_j = dats[j]
        pc, pv = _ss.pearsonr(dat_i, dat_j)
        pcpvs[ij, 0] = pc
        pcpvs[ij, 1] = pv

        _plt.subplot2grid((3, 3), (i, j-1))
        _plt.scatter(dat_i + 0.2 * _N.random.randn(N), dat_j + 0.2 * _N.random.randn(N), color="black", s=4)
        _plt.title(r"$r = %(pc).2f$, $p < %(pv).3f$" % {"pc" : pc, "pv" : pv})        
        _plt.xlim(-8, 8)
        _plt.ylim(-8, 8)
        _plt.xticks([-6, -3, 0, 3, 6])
        _plt.yticks([-6, -3, 0, 3, 6])        
        _plt.xlabel(sdats[i])
        _plt.ylabel(sdats[j])
fig.subplots_adjust(wspace=0.4, hspace=0.4)
_plt.savefig("TMB1_win_cov")


# sdat = ["Unbiased ran", "exploitable win", "biased ran", "mimic"]

# di   = -1
# for dat in [all_unbr_dat, all_wtl_dat, all_bia_dat, all_mim_dat]:
#     di += 1
#     fig = _plt.figure(figsize=(13, 14))
#     _plt.suptitle(sdat[di])
#     L = 30
#     cmsm = _N.cumsum(dat[:, :, 2], axis=1)

#     col = -1

#     for i in range(len(partIDs)):
#         if (i % L == 0):
#             col += 1
#             row = -1
#         row += 1
#         _plt.plot(range(col*50, col*50+40), cmsm[i] + 30*row, color="black")
#         _plt.plot([col*50, col*50+40], [0 + 30*row, 0 + 30*row], color="red")

#     _plt.savefig("cumsum_%s" % sdat[di])

# mat_dict = {}
# mat_dict["unbiasedran"] = all_unbr_dat
# mat_dict["biasedran"] = all_bia_dat
# mat_dict["exploitablewin"] = all_wtl_dat
# mat_dict["mimic"] = all_mim_dat
# _scio.savemat("tmb1_hnddat", mat_dict)
