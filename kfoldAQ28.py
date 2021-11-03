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


def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

#lm = depickle("AQ28_vs_RPS_300.dmp")
lm = depickle("AQ28_vs_RPS.dmp")

AQ28scores = ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]
AQ28scores_ab = ["AQ28", "SS", "IM", "RT", "SW", "FP"]

outcomes = ["netwins",
            "win_aft_win", "tie_aft_win", "los_aft_win",            
            "win_aft_tie", "tie_aft_tie", "los_aft_tie",
            "win_aft_los", "tie_aft_los", "los_aft_los",
            "u_or_d_tie", "u_or_d_res", "stay_tie", "stay_res"]
outcome_ab = ["NW",
              "WW", "TW", "LW",
              "WT", "TT", "LT",
              "WL", "TL", "LL",
              "T", "W"]  #  0.5 3.5 6.5 9.5

rule_change_features = ["pfrm_change36", "pfrm_change69", "pfrm_change912", "sum_sd", "moresim"]

features = ["entropyT2", "entropyW2", "entropyL2", "entropyU", "entropyS", "entropyD",
            "isis_corr", "isis_lv", "entropyL", "entropyT", "entropyW"]
#            "pfrm_change36", "pfrm_change69", "pfrm_change912"]
for feat in features:
    #exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : feat})
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : feat})
    exec("%(f)s = %(f)s / _N.std(%(f)s)" % {"f" : feat})
    exec("%(f)s = %(f)s - _N.mean(%(f)s)" % {"f" : feat})
for outc in outcomes:
    #exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : outc})
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : outc})
    exec("%(f)s = %(f)s / _N.std(%(f)s)" % {"f" : outc})
    exec("%(f)s = %(f)s - _N.mean(%(f)s)" % {"f" : outc})
for scrs in AQ28scores:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : scrs})
    exec("%(f)s = %(f)s / _N.std(%(f)s)" % {"f" : scrs})
    exec("%(f)s = %(f)s - _N.mean(%(f)s)" % {"f" : scrs})
    
for rcf in rule_change_features:
    #exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : rcf})
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : rcf})
    exec("%(f)s = %(f)s / _N.std(%(f)s)" % {"f" : rcf})
    exec("%(f)s = %(f)s - _N.mean(%(f)s)" % {"f" : rcf})
#Xs_train = Xs_train / _N.std(Xs_train, axis=0)
#        Xs_train = Xs_train - _N.mean(Xs_train, axis=0)
        
#        ys_train= (ys_train - _N.mean(ys_train)) / _N.std(ys_train)
    

#use_feats = ["CondOutc"]#, "CondOutc", "Both"]
use_feats = ["Both"]
iuf = -1
resids = _N.zeros((3, 16, 4))

targets    = ["SB:SW", "SB:IM", "SB:SS", "AQ28"]
#targets    = ["Switch", "Imag", "Soc_skils", "AQ28"]
#targets    = ["Soc_skils"]
#targets    = ["Imag"]
#targets    = ["Switch"]
reps       = 40
#AQ28scrs = soc_skils + imag + switch

allscores  = _N.empty((4, reps))

ths = _N.arange(AQ28scrs.shape[0], dtype=_N.int)
trainSz = 60
partitions = 100
all_trainingInds = _N.empty((partitions, trainSz), dtype=_N.int)
#  For each shuffle,

ti = -1
#fig = _plt.figure(figsize=(3, 5))
fig = _plt.figure(figsize=(6, 2.5))
for target in targets:

    ti += 1
    #ax = fig.add_subplot(2, 2,ti+1)
    ax = fig.add_subplot(1, 4, ti+1)
    pcspvs = _N.zeros((partitions, 3, 2))
    #  compare for me AQ28  - just RC, just CO, BOTH
    #  for sTarget in targets:

    iuf = -1
    for uf in use_feats:
        iuf += 1
        ths = _N.arange(AQ28scrs.shape[0])

        # ######################################
        Xs_trainSw = _N.empty((ths.shape[0], 4))
        Xs_trainSw[:, 3] = isis_corr[ths]
        Xs_trainSw[:, 0] = pfrm_change69[ths]
        Xs_trainSw[:, 1] = win_aft_tie[ths]
        #Xs_trainSw[:, 2] = u_or_d_tie[ths]
        Xs_trainSw[:, 2] = entropyW2[ths]
        #######################  ADDING        
        # Xs_trainSw = _N.empty((ths.shape[0], 2))
        # #Xs_trainSw[:, 4] = isis_corr[ths]
        # Xs_trainSw[:, 0] = pfrm_change69[ths]
        # Xs_trainSw[:, 1] = win_aft_tie[ths] + entropyW2[ths] - u_or_d_tie[ths] - isis_corr[ths]
        # Xs_trainSw[:, 1] = win_aft_tie[ths]
        # Xs_trainSw[:, 2] = u_or_d_tie[ths]
        # Xs_trainSw[:, 3] = entropyW2[ths]
        
        #Xs_trainSw[:, 5] = stay_tie[ths]
        ys_trainSw = switch[ths]
        # ######################################
        # ######################################        
        Xs_trainI = _N.empty((ths.shape[0], 4))
        #Xs_trainI[:, 0] = entropyT2[ths]
        Xs_trainI[:, 1] = pfrm_change69[ths]
        Xs_trainI[:, 0] = entropyU[ths]
        #Xs_trainI[:, 2] = u_or_d_tie[ths]
        Xs_trainI[:, 3] = win_aft_tie[ths] 
        Xs_trainI[:, 2] = stay_tie[ths]
        #Xs_trainI[:, 4] = sum_sd[ths]# + u_or_d_tie[ths]
        #######################  ADDING
        # Xs_trainI = _N.empty((ths.shape[0], 2))
        # Xs_trainI[:, 1] = win_aft_tie[ths]
        # Xs_trainI[:, 0] = entropyU[ths]-stay_tie[ths]-pfrm_change69[ths]

        ys_trainI = imag[ths]
        #############################
        #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
        Xs_trainSS = _N.empty((ths.shape[0], 1))
        #Xs_trainSS[:, 0] = entropyUD[ths]
        #Xs_trainSS[:, 0] = entropyT2[ths]
        #Xs_trainSS[:, 1] = sum_sd[ths]
        #Xs_trainSS[:, 2] = pfrm_change69[ths] #  +0.16
        #Xs_trainSS[:, 1] = moresim[ths]       #  +0.21
        #Xs_trainSS[:, 4] = tie_aft_tie[ths]
        #Xs_trainSS[:, 0] = win_aft_tie[ths]   #  +0.17
        Xs_trainSS[:, 0] = win_aft_tie[ths] + moresim[ths] + pfrm_change69[ths] - sum_sd[ths] + entropyT2[ths]
        ys_trainSS = soc_skils[ths]
        # #############################
        #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
        Xs_trainA = _N.empty((ths.shape[0], 5))
        Xs_trainA[:, 0] = entropyT2[ths]
        #Xs_trainA[:, 0] = entropyU[ths] - pfrm_change69[ths] - win_aft_tie[ths]
        #Xs_trainA[:, 1] = los_aft_tie[ths]
        Xs_trainA[:, 1] = pfrm_change69[ths]
        Xs_trainA[:, 2] = win_aft_tie[ths]
        #Xs_trainA[:, 3] = los_aft_tie[ths]
        Xs_trainA[:, 3] = stay_tie[ths]
        Xs_trainA[:, 4] = u_or_d_res[ths]        
        #Xs_trainA[:, 5] = sum_sd[ths]
        #Xs_trainA[:, 5] = moresim[ths]
        ys_trainA = AQ28scrs[ths]

        if target == "AQ28":
            Xs_train = _N.array(Xs_trainA)
            ys_train = _N.array(ys_trainA)
        elif target == "SB:IM":                
            Xs_train = _N.array(Xs_trainI)
            ys_train = _N.array(ys_trainI)
        elif target == "SB:SW":                
            Xs_train = _N.array(Xs_trainSw)
            ys_train = _N.array(ys_trainSw)
        elif target == "SB:SS":                
            Xs_train = _N.array(Xs_trainSS)
            ys_train = _N.array(ys_trainSS)

        clf = _skl.LinearRegression()
        #clf = _skl.Ridge()
        #rr  = TheilSenRegressor()
        #clf  = linear_model.Lasso()
        nrep = 100
        #for rs in range(nrep):
            #cv = ShuffleSplit(n_splits=5, test_size=0.35, random_state=rs)
        #kf = KFold(n_splits=3, )#, test_size=0.35, random_state=rs)
        rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)#, random_state=0)
        #scores = cross_val_score(clf, Xs_train, ys_train, cv=cv)
        scores = cross_val_score(clf, Xs_train, ys_train, cv=rkf)

    _plt.title(target, fontsize=13)
    _plt.scatter(_N.zeros(len(scores))+0.1*_N.random.randn(len(scores)), scores, s=3, color="black")
    _plt.plot([-0.5, 0.2], [_N.mean(scores), _N.mean(scores)], color="red", lw=2)
    _plt.plot([-0.2, 0.5], [_N.median(scores), _N.median(scores)], color="lightgreen", lw=2)
    if ti == 3:
        ax.spines["left"].set_linewidth(4)
        ax.spines["right"].set_linewidth(4)
        ax.spines["top"].set_linewidth(4)
        ax.spines["bottom"].set_linewidth(4)
    
    _plt.xlim(-1.2, 0.8)
    _plt.ylim(-0.5, 0.5)
    _plt.axhline(y=0, ls=":")
    _plt.xticks([])
    if ti == 0:
        _plt.yticks([-0.3, 0, 0.3], fontsize=13)
    else:
        _plt.yticks([], fontsize=13)        
    #if (ti == 0) or (ti == 2):
    #    _plt.ylabel(r"$r^2$", fontsize=15, rotation=0)
    if ti == 0:
        _plt.ylabel(r"$r^2$", fontsize=15, rotation=0)
    
    _plt.text(-1.1, 0.1, "mn:\n%(mn).2f\nmd:\n%(md).2f" % {"mn" : _N.mean(scores), "md" : _N.median(scores)})
#fig.subplots_adjust(hspace=0.22, left=0.31, wspace=0.8, right=0.96)
fig.subplots_adjust(left=0.15, wspace=0.15, right=0.96)
_plt.savefig("CoeffDeterm", transparent=True)

# _ss.pearsonr(entropyL -  entropyW, switch)
