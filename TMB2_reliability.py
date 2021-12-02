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

def standardize(y):
    ys = y - _N.mean(y)
    ys /= _N.std(ys)
    return ys

def depickle(s):
    import pickle
    with open(s, "rb") as f:
        lm = pickle.load(f)
    return lm

def unskew(dat):
    sk = _N.empty(15)
    im = -1
    ms = _N.linspace(0.01, 1.1, 15)
    for m in ms:
        im += 1
        sk[im] = _ss.skew(_N.exp(dat / (m*_N.mean(dat))))
    min_im = _N.where(_N.abs(sk) == _N.min(_N.abs(sk)))[0][0]
    return _N.exp(dat / (ms[min_im]*_N.mean(dat)))
 
id = 0

lm1 = depickle("predictAQ28dat/AQ28_vs_RPS_1.dmp")
lm2 = depickle("predictAQ28dat/AQ28_vs_RPS_2.dmp")

features_cab1 = lm1["features_cab1"]
features_cab2 = lm1["features_cab2"]
features_AI   = lm1["features_AI"]
features_stat = lm1["features_stat"]

ifeatinds_soc_skils = _N.loadtxt("use_features_soc_skils", dtype=_N.int)
ifeatinds_imag = _N.loadtxt("use_features_imag", dtype=_N.int)
ifeatinds_rout = _N.loadtxt("use_features_rout", dtype=_N.int)
ifeatinds_switch = _N.loadtxt("use_features_switch", dtype=_N.int)
ifeatinds_AQ28scrs = _N.loadtxt("use_features_AQ28scrs", dtype=_N.int)

indlist = _N.sort(_N.unique(ifeatinds_soc_skils.tolist() + ifeatinds_imag.tolist() + ifeatinds_rout.tolist() + ifeatinds_switch.tolist() + ifeatinds_AQ28scrs.tolist()))

cmp_againsts = features_cab1 + features_cab2 + features_AI + features_stat

for ca in cmp_againsts:
    for v in range(1, 3):
        exec("temp = lm%(v)d[\"%(ca)s\"]" % {"ca" : ca, "v" : v})
        exec("%(ca)s%(v)d = lm%(v)d[\"%(ca)s\"]" % {"ca" : ca, "v" : v})    
        if ca[0:7] == "entropy":
            exec("temp = unskew(temp)" % {"ca" : ca, "v" : v})
        print(ca)
        exec("%(ca)s%(v)d = standardize(temp)" % {"ca" : ca, "v" : v})


pcpvs = _N.empty((len(cmp_againsts), 2))

all_im = 0
for scalm in ["features_cab1", "features_cab2", "features_AI", "features_stat", ]:
    calm = lm1[scalm]
    im = 0
    all_im_cat = 0    

    for ca in calm:
        if im % 30 == 0:
            fig = _plt.figure(figsize=(13, 13))
            _plt.suptitle(scalm)
            im = 0
        exec("mark_v1 = %s1" % ca)
        exec("mark_v2 = %s2" % ca)    
        im += 1
        all_im += 1
        all_im_cat += 1        
        ax = fig.add_subplot(5, 6, im)

        if len(_N.where(indlist == all_im-1)[0]) > 0:
            ax.set_facecolor("#CCCCCC")
        _plt.scatter(mark_v1, mark_v2, color="black", s=3)
        pc, pv = _ss.pearsonr(mark_v1, mark_v2)
        #pc, pv = _ss.spearmanr(mark_v1, mark_v2)
        _plt.xlabel("round 1")
        _plt.ylabel("round 2")

        pcpvs[all_im-1, 0] = pc
        pcpvs[all_im-1, 1] = pv
        marker = _N.array(mark_v1.tolist() + mark_v2.tolist())
        minM = _N.min(marker)
        maxM = _N.max(marker)
        A = maxM - minM
        _plt.plot([minM - 0.1*A, maxM + 0.1*A], [minM - 0.1*A, maxM + 0.1*A])
        _plt.title("%(pc).3f  %(nm)s" % {"pc" : pc, "nm" : calm[all_im_cat-1]})

    fig.subplots_adjust(hspace=0.9, bottom=0.08, left=0.08, right=0.95)
    _plt.savefig("stability_%s" % scalm)

fig = _plt.figure(figsize=(7, 3))
_plt.scatter(_N.arange(pcpvs.shape[0]), pcpvs[:, 0], color="black", s=7)
_plt.scatter(indlist, pcpvs[indlist, 0], color="red", s=20)
_plt.xlabel("feature index")
_plt.ylabel("CC round 1 and 2")
_plt.axvline(x=(len(features_cab1)-0.5))
_plt.axvline(x=(len(features_cab1+features_cab2)-0.5))
_plt.axvline(x=(len(features_cab1+features_cab2+features_AI)-0.5))
_plt.axhline(y=0, ls="--")
_plt.ylim(-1, 1)
fig.subplots_adjust(bottom=0.15)
_plt.savefig("stability_summary")
"""
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

"""
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
    
