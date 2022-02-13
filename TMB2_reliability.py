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

id = 0

win_type = 2   #  window is of fixed number of games
#win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
#win     = 4
smth    = 1
#smth    = 3

win_type = 2  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1

outdir = "Results_%(wt)d%(w)d%(s)d" % {"wt" : win_type, "w" : win, "s" : smth}

lm1 = depickle("predictAQ28dat/AQ28_vs_RPS_1_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})
lm2 = depickle("predictAQ28dat/AQ28_vs_RPS_2_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})

features_cab1 = lm1["features_cab1"]
features_cab2 = lm1["features_cab2"]
features_AI   = lm1["features_AI"]
features_stat = lm1["features_stat"]

# ifeatinds_soc_skils = _N.loadtxt("use_features_soc_skils", dtype=_N.int)
# ifeatinds_imag = _N.loadtxt("use_features_imag", dtype=_N.int)
# ifeatinds_rout = _N.loadtxt("use_features_rout", dtype=_N.int)
# ifeatinds_switch = _N.loadtxt("use_features_switch", dtype=_N.int)
# ifeatinds_AQ28scrs = _N.loadtxt("use_features_AQ28scrs", dtype=_N.int)

#indlist = _N.sort(_N.unique(ifeatinds_soc_skils.tolist() + ifeatinds_imag.tolist() + ifeatinds_rout.tolist() + ifeatinds_switch.tolist() + ifeatinds_AQ28scrs.tolist()))

cmp_againsts = lm1["cmp_againsts_name"]#features_cab1 + features_cab2 + features_AI + features_stat
indlist = _N.arange(len(cmp_againsts))

pcpvs = _N.empty((len(cmp_againsts), 2))

all_im = 0

all_features = _N.empty((len(cmp_againsts), len(lm1["AQ28scrs"]), 2))
for scalm in ["cmp_againsts_name"]:#["features_cab1", "features_cab2", "features_AI", "features_stat", ]:
    calm = lm1[scalm]
    #print(calm)
    im = 0
    all_im_cat = 0    

    for ca in calm:
        if im % 30 == 0:
            fig = _plt.figure(figsize=(13, 13))
            fig.subplots_adjust(hspace=0.95, bottom=0.08, left=0.08, right=0.95)
            
            _plt.suptitle(scalm, fontsize=10)
            im = 0
        exec("mark_v1 = lm1[\"%s\"]" % ca)
        exec("mark_v2 = lm2[\"%s\"]" % ca)        

        all_features[all_im, :, 0] = mark_v1
        all_features[all_im, :, 1] = mark_v2
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
        _plt.title("%(pc).2f  %(nm)s" % {"pc" : pc, "nm" : calm[all_im_cat-1]}, fontsize=9)

    _plt.savefig("stability_%s" % scalm)
    _plt.close()


#unreliable = _N.where(pcpvs[:, 0] < 0)[0]
#for unr in unreliable:
#    print(cmp_againsts[unr])

##############  look at the features used for prediction
pcthresh=0.08
for starget in ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs"]:
    print("--------   target   %s" % starget)
    fp = open("%(od)s/reliable_feats%(tar)s_%(th).2f" % {"tar" : starget, "od" : outdir, "th" : pcthresh}, "r")
    lines = fp.readlines()
    pcs = _N.empty(len(lines))
    iln = -1
    fig  = _plt.figure()
    for ln in lines:
        iln += 1
        ind = cmp_againsts.index(ln.rstrip())
        print("%(ca)s  %(pc).2f" % {"pc" : pcpvs[ind, 0], "ca" : ln.rstrip()})
        pcs[iln] = pcpvs[ind, 0]
    _plt.plot(pcs)
    _plt.ylim(-1, 1)
    _plt.axhline(y=0, ls=":")
    print("mean:   %.2f" % _N.mean(pcs))

    
# fig = _plt.figure(figsize=(7, 3))
# _plt.scatter(_N.arange(pcpvs.shape[0]), pcpvs[:, 0], color="black", s=7)
# _plt.scatter(indlist, pcpvs[indlist, 0], color="red", s=20)
# _plt.xlabel("feature index")
# _plt.ylabel("CC round 1 and 2")
# _plt.axvline(x=(len(features_cab1)-0.5))
# _plt.axvline(x=(len(features_cab1+features_cab2)-0.5))
# _plt.axvline(x=(len(features_cab1+features_cab2+features_AI)-0.5))
# _plt.axhline(y=0, ls="--")
# _plt.ylim(-1, 1)
# fig.subplots_adjust(bottom=0.15)
# _plt.savefig("stability_summary")
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
    
# SHUFFS = 500
# sign_shuf=20
# pm_one   = _N.array([-1, 1])
# pcpvs_all = _N.empty((SHUFFS, 2))
# pick = 2
# for i in range(SHUFFS):
#     rand_inds = _N.random.choice(_N.arange(80), pick, replace=False)
#     pcs = _N.empty((sign_shuf, 2))

#     for ss in range(sign_shuf):
#         wgts = _N.random.choice(pm_one, pick)
#         wgtsr = wgts.reshape((pick, 1, 1))
#         v = _N.sum(all_features[rand_inds]*wgtsr, axis=0)
#         pc, pv = _ss.pearsonr(v[:, 0], v[:, 1])
#         pcs[ss, 0] = pc
#         pcs[ss, 1] = pv
#     args = pcs[:, 0].argsort()
#     pc = pcs[args[-1], 0]
#     pv = pcs[args[-1], 1]    
#     print("%(pc).2f  %(pv).1e" % {"pc" : pc, "pv" : pv})
#     pcpvs_all[i, 0] = pc
#     pcpvs_all[i, 1] = pv    


# SHUFFS = 500
# pcpvs_all = _N.empty((SHUFFS, 2))
# for ish in range(SHUFFS):
#     rand_inds = _N.random.choice(_N.arange(80), pick, replace=False)
#     pc0, pv0 = _ss.pearsonr(all_features[rand_inds[0], :, 0], all_features[rand_inds[1], :, 0])
#     pc1, pv1 = _ss.pearsonr(all_features[rand_inds[0], :, 1], all_features[rand_inds[1], :, 1])
#     pcpvs_all[ish, 0] = pc0
#     pcpvs_all[ish, 1] = pc1

# soc_skils
# imag
# rout
# switch
# AQ28scrs

# fig = _plt.figure(figsize=(4, 10))
# ti = 0
# for tar in ["soc_skils", "imag", "rout", "switch", "AQ28scrs"]:
#     ti += 1
#     ax = fig.add_subplot(5, 1, ti)
#     ax.set_aspect("equal")
    
#     lm = depickle("LRfit%s.dmp" % tar)
#     ifeats = lm["features_thresh3_fld4"]
#     iwgts  = lm["weights_thresh3_fld4"]
    
#     wgtsr = iwgts.reshape((len(iwgts), 1, 1))
#     v = _N.sum(all_features[ifeats]*wgtsr, axis=0)
#     xymin = _N.min(v)
#     xymax = _N.max(v)
#     amp   = xymax - xymin
#     _plt.xlim(xymin-0.1*amp, xymax+0.1*amp)
#     _plt.ylim(xymin-0.1*amp, xymax+0.1*amp)
#     _plt.plot([xymin, xymax], [xymin, xymax])
#     pc, pv = _ss.pearsonr(v[:, 0], v[:, 1])

#     _plt.title(pc)
#     _plt.scatter(v[:, 0], v[:, 1])
