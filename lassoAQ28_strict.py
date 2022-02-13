import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
from sklearn import linear_model
import sklearn.linear_model as _skl
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn.preprocessing as _skp
import pickle
import cv_funcs as cvf
import os

#------------------ SOCIAL SKILLS     [0, 1, 2, 3, 4, 5, 6]
#0   1C  "I prefer to do things with others rather than on my own.",
#1   11C "I find social situations easy.",
#2   13A "I would rather go to a library than to a party.",
#3   15C "I find myself drawn more strongly to people than to things.",
#4   22A "I find it hard to make new friends.",
#5   44C "I enjoy social occasions.",
#6   47C "I enjoy meeting new people.",

#------------------ ROUTINE    [0, 1, 2, 3]
#0   2A  "I prefer to do things the same way over and over again.",
#1   25C "It does not upset me if my daily routine is disturbed.",
#2  34C"I enjoy doing things spontaneously.",
#3  46A"New situations make me anxious.",

#------------------ SWITCHING  [0, 1, 2, 3]
#0  4A "I frequently get strongly absorbed in one thing.",
#1  10C"I can easily keep track of several different people's conversations.",
#2  32C"I find it easy to do more than one thing at once.",
#3  37C"If there is an interruption, I can switch back very quickly.",

#------------------ IMAG            [0, 1, 2, 3, 4, 5, 6, 7]
#0  3C "Trying to imagine something, I find it easy to create a picture in my mind.",
#1  8C "Reading a story, I can easily imagine what the characters might look like.",
#2  14C"I find making up stories easy.",
#3  20A"Reading a story, I find it difficult to work out the character's intentions.",
#4  36C"I find it easy to work out what someone is thinking or feeling.",
#5  42A"I find it difficult to imagine what it would be like to be someone else.",
#6  45A"I find it difficult to work out people's intentions.",
#7  50C"I find it easy to play games with children that involve pretending.

#------------------ FACT NUMB AND PATT",     [0, 1, 2, 3, 4]
#0  6A "I usually notice car number plates or similar strings of information.",
#1  9A "I am fascinated by dates.",
#2  19A"I am fascinated by numbers.",
#3  23A"I notice patterns in things all the time.",
#4  41A"I like to collect information about categories of things."

def str_float_array(arr, sfmt):
    s = "["
    for i in range(arr.shape[0]-1):
        s += (sfmt % arr[i]) + ", "
    s += (sfmt % arr[i]) + "]"       
    return s

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
    sk = _N.empty(61)
    im = -1
    if _ss.skew(dat) < 0:
        dat *= -1
    dat -= _N.mean(dat)   #  0-mean
    dat /= _N.std(dat)
    dat -= _N.min(dat)
    amp = _N.max(dat) - _N.min(dat)
    dat += 0.1*amp

    return _N.log(dat)

# def unskew(dat):
#     sk = _N.empty(15)
#     im = -1
#     ms = _N.linspace(0.01, 1.1, 15)
#     for m in ms:
#         im += 1
#         sk[im] = _ss.skew(_N.exp(dat / (m*_N.mean(dat))))
#     min_im = _N.where(_N.abs(sk) == _N.min(_N.abs(sk)))[0][0]
#     return _N.exp(dat / (ms[min_im]*_N.mean(dat)))

#win_type = 2   #  window is of fixed number of games
win_type = 2  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1

outdir = "Results_%(wt)d%(w)d%(s)d" % {"wt" : win_type, "w" : win, "s" : smth}
if not os.access(outdir, os.F_OK):
    os.mkdir(outdir)

lm = depickle("predictAQ28dat/AQ28_vs_RPS_1_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})

AQ28scores = ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]
AQ28scores_ab = ["AQ28", "SS", "IM", "RT", "SW", "FP"]

#  7+8+4+4+5 =
#soc_skils_use = _N.array([0, 1, 2, 3, 4, 5, 6])
soc_skils_use = _N.array([0, 1, 2, 3, 4, 5, 6])
#soc_skils_use = _N.array([0, 1, 2, 3, 4, 5,])  #  this does better
#soc_skils_use = _N.array([0, 1, 2, 3, 4, 6,])  #  this does better
#soc_skils_use = _N.array([0, 1, 2, 3, 5, 6,])   # poor
#soc_skils_use = _N.array([0, 1, 2, 4, 5, 6,])    #  not quite as good
#soc_skils_use = _N.array([0, 1, 3, 4, 5, 6,])    #  not quite as good
#imag_use      = _N.array([0, 1, 2, 3, 4, 5, 6, 7])
imag_use      = _N.array([0, 1, 2, 3, 4, 5, 6, 7])
rout_use      = _N.array([0, 1, 2, 3])
switch_use    = _N.array([0, 1, 2, 3])
fact_pat_use  = _N.array([0, 1, 2, 3, 4])

              
features_cab1 = lm["features_cab1"]
features_cab2 = lm["features_cab2"]
features_AI   = lm["features_AI"]
features_stat = lm["features_stat"]
_cmp_againsts = features_stat + features_cab2 + features_AI + features_stat
cmp_againsts = lm["cmp_againsts_name"]#features_cab1 + features_cab2 + features

#cmp_againsts  = []
######  unskew and standardize the features to use.
for ca in cmp_againsts:
    # if (ca[0:4] == "mnFt") or (ca[0:4] == "sdFt"):
    #     for i in range(3):
    #         exec("temp = lm[\"%(ca)s\"][:, %(cmp)d]" % {"ca" : ca, "cmp" : i})
    #         exec("%(ca)s_%(cmp)d = lm[\"%(ca)s\"]" % {"ca" : ca, "cmp" : i})
    #         #exec("temp = unskew(temp)")
    #         exec("%(ca)s_%(cmp)d_s = standardize(temp)" % {"ca" : ca, "cmp" :i})
    #         print("%(ca)s_%(cmp)d_s" % {"ca" : ca, "cmp" : i})
    #         cmp_againsts.append("%(ca)s_%(cmp)d" % {"ca" : ca, "cmp" : i})
    # elif (ca[0:6] == "sum_sd"):
    #     for i in range(3):
    #         for j in range(3):            
    #             exec("temp = lm[\"%(ca)s\"][:, %(i)d, %(j)d]" % {"ca" : ca, "i" : i, "j" : j})
    #             exec("%(ca)s_%(i)d%(j)d = lm[\"%(ca)s\"]" % {"ca" : ca, "i" : i, "j" : j})
    #             #exec("temp = unskew(temp)")
    #             exec("%(ca)s_%(i)d%(j)d_s = standardize(temp)" % {"ca" : ca, "i" :i, "j" : j})
    #             print("%(ca)s_%(i)d%(j)d_s" % {"ca" : ca, "i" : i, "j" : j})
    #             cmp_againsts.append("%(ca)s_%(i)d%(j)d" % {"ca" : ca, "i" : i, "j" : j})
    # elif (ca[7:12] == "corrs"):
    #     for i in range(36):
    #         exec("temp = lm[\"%(ca)s\"][%(i)d, :]" % {"ca" : ca, "i" : i})
    #         exec("%(ca)s_%(i)d = lm[\"%(ca)s\"]" % {"ca" : ca, "i" : i})
    #         #exec("temp = unskew(temp)")
    #         exec("%(ca)s_%(i)d_s = standardize(temp)" % {"ca" : ca, "i" :i})
    #         print("%(ca)s_%(i)d_s" % {"ca" : ca, "i" : i})
    #         cmp_againsts.append("%(ca)s_%(i)d" % {"ca" : ca, "i" : i})
    # else:
    exec("temp = lm[\"%(ca)s\"]" % {"ca" : ca})
    exec("%(ca)s = lm[\"%(ca)s\"]" % {"ca" : ca})
    if ca[0:7] == "entropy":
        exec("temp = unskew(temp)")
    exec("%(ca)s_s = standardize(temp)" % {"ca" : ca})
    #print("%(ca)s" % {"ca" : ca})
    #cmp_againsts.append("%(ca)s" % {"ca" : ca})

filtdat = lm["filtdat"]
filtdat_sh = _N.array(filtdat)
_N.random.shuffle(filtdat_sh)
for scrs in AQ28scores:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : scrs})
    exec("%(f)s_sh = _N.array(%(f)s)" % {"f" : scrs})    
    exec("%(f)s_sh[filtdat] = %(f)s[filtdat_sh]" % {"f" : scrs})    

print("Using %(fd)d of %(all)d participants" % {"fd" : len(filtdat), "all" : AQ28scrs.shape[0]})


##  using custom list of answers
for scrs in AQ28scores[1:]:   #  raw answers data
    exec("ans_%(f)s = lm[\"ans_%(f)s\"]" % {"f" : scrs})
    exec("%(f)s     = _N.sum(ans_%(f)s[:, %(f)s_use], axis=1)" % {"f" : scrs})

#AQ28scrs = soc_skils + imag + switch + rout + fact_pat
    
#imag = _N.sum(ans_imag[:, imag_use], axis=1)
#switch = _N.sum(ans_switch[:, switch_use], axis=1)
#soc_skils = _N.sum(ans_soc_skils[:, soc_skils_use], axis=1)
    
####################  USE ALL DATA
####################  OUR DATA HAS 1 very strong outlier - 1 person with 
####################  AQ-28 score of 30 or so.  Next closest person has 
####################  score of 42 (or so).  
####################  using these 
#filtdat = _N.where((AQ28scrs > 35) & (rout > 4))[0]
#filtdat = _N.where((AQ28scrs > 35))[0]

#starget = ""



for shuffle in [False]:#, True]:
    for pcthresh in [0.08, 0.1, 0.12]:
        sshf     = ""
        if shuffle:
            sshf = "_sh"
        for starget in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        #for starget in ["AQ28scrs"]:
            X_all_feats            = _N.empty((len(filtdat), len(cmp_againsts)))

            cand_feats = []

            if shuffle:
                exec("target = %s_sh" % starget)
            else:
                exec("target = %s" % starget)                
            y    = target[filtdat]

            iaf = -1
            for af in cmp_againsts:
                iaf += 1
                exec("feat = %s_s" % af)
                X_all_feats[:, iaf] = feat[filtdat]

            for i in range(X_all_feats.shape[1]):
                pc, pv = _ss.pearsonr(y, X_all_feats[:, i])
                if _N.abs(pc) > pcthresh:
                    cand_feats.append(i)
                #    print("%(f)s   %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "f" : cmp_againsts[i]})
            v_cand_feats = _N.array(cand_feats)

            # REPS = 20
            # scrs = _N.empty(REPS*4)

            outer_flds = 4
            inner_flds = 4
            outer_REPS = 50
            inner_REPS = 20

            X_cand_feats = _N.array(X_all_feats[:, v_cand_feats])
            # rkf = RepeatedKFold(n_splits=outer_flds, n_repeats=REPS)#, random_state=0)
            # rkfINNER = RepeatedKFold(n_splits=inner_flds, n_repeats=4)#, random_state=0)

            # iii = -1

            # ichosen = _N.zeros(len(cmp_againsts), dtype=_N.int)
            # reg_coefs = _N.empty((outer_flds*REPS, len(cmp_againsts)))
            # for train, test in rkf.split(datinds):
            #     iii += 1
            #     ####  first, pick alpha using LassoCV
            #     train_data_inds = _N.arange(len(train))
            #     splits = rkfINNER.split(train_data_inds)
            #     reg = LassoCV(cv=splits, max_iter=100000).fit(X_all_feats[train], y[train])

            #     #reg = LassoCV(cv=4, max_iter=100000).fit(X_all_feats[train], y[train])

            #     maxWeight = _N.max(_N.abs(reg.coef_))
            #     reg_coefs[iii] = reg.coef_

            weights, feat_inv_cv, inner_scores, tar_prd_pcs, pcpvs = cvf.pickFeaturesTwoCVs(X_cand_feats, y, outer_flds=outer_flds, inner_flds=inner_flds, outer_REPS=outer_REPS, innter_REPS=inner_REPS)

            print("------------   %s" % starget)
            print(feat_inv_cv)

            fig = _plt.figure(figsize=(8, 11))
            #fig.add_subplot(3, 1, 1)
            _plt.subplot2grid((3, 6), (0, 0), colspan=5)
            nonzero_weights = _N.zeros(weights.shape[1], dtype=_N.int)

            for icv in range(weights.shape[0]):
                nonzero = _N.where(weights[icv] != 0)[0]
                nonzero_weights[nonzero] += 1

            reliable_thr = int(0.9*(outer_flds * outer_REPS))
            i_reliable_cands = _N.where(nonzero_weights >= reliable_thr)[0]
            i_unreliable_cands = _N.where(nonzero_weights < reliable_thr)[0]

            for icv in range(weights.shape[0]):
                _plt.scatter(i_reliable_cands + _N.random.randn(len(i_reliable_cands)) * 0.1, weights[icv, i_reliable_cands], s=3, color="black")
                _plt.scatter(i_unreliable_cands + _N.random.randn(len(i_unreliable_cands)) * 0.1, weights[icv, i_unreliable_cands], s=3, color="grey")            

            abv_0 = len(_N.where(inner_scores > 0)[0])
            _plt.xlabel("feature #")
            _plt.ylabel("weights")

            _plt.subplot2grid((3, 6), (0, 5), colspan=1)
            _plt.scatter(_N.random.randn(outer_REPS*outer_flds), inner_scores, s=3, color="black")
            _plt.axhline(y=0, ls=":", color="grey")
            _plt.title("CD")
            _plt.ylabel("coeff determination")
            _plt.xlim(-8, 8)
            _plt.ylim(-1, 1)

            _plt.subplot2grid((3, 6), (1, 0), colspan=6)
            _plt.hist(nonzero_weights / (outer_flds * outer_REPS), bins=_N.linspace(-0.005, 1.005, 102))

            _plt.suptitle("tar: %(tar)s    feat_pcth: %(pcth).2f   CD abv 0: %(az)d/%(R)d   median: %(md).2f" % {"az" : abv_0, "md" : _N.median(inner_scores), "pcth" : pcthresh, "tar" : starget, "R" : (outer_REPS*outer_flds)})

            #  nonzero_weights is # of times for each feature LassoCV picked it (index of v_cand_feats)
            reliable_feats = v_cand_feats[i_reliable_cands]

            _plt.subplot2grid((3, 6), (2, 0), colspan=6)
            _plt.title("reliable features")
            _plt.ylim(0, 1000)
            _plt.xlim(0, 1000)
            _plt.xticks([])
            _plt.yticks([])

            more2print = False
            col        = 0
            fp = open("%(od)s/reliable_feats%(tar)s_%(th).2f" % {"tar" : starget, "od" : outdir, "th" : pcthresh}, "w")
            for ir in range(len(reliable_feats)):
                fp.write("%s\n" % cmp_againsts[reliable_feats[ir]])
                more2print = True
                if ir % 10 == 0:
                    s = ""
                s += "  %(times).2f   %(feat)s\n" % {"feat" : cmp_againsts[reliable_feats[ir]], "times" : (nonzero_weights[i_reliable_cands[ir]]) / (outer_REPS * outer_flds)}
                if ir % 10 == 9:
                    _plt.text(col*500+10, 10, s)
                    col += 1
                    more2print = False
            fp.close()
            if more2print:
                _plt.text(col*500+10, 10, s)

            fig.subplots_adjust(wspace=0.8)

            _plt.savefig("%(od)s/lassoAQ28_%(tar)s_%(pcth).2f%(sh)s.png" % {"tar" : starget, "pcth" : pcthresh, "sh" : sshf, "od" : outdir})



