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
import itertools
from sklearn.decomposition import PCA


def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

lm = depickle("AQ28_vs_RPS_300.dmp")

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
            "isis_corr", "isis_lv"]


all_features = outcomes + rule_change_features + features
X            = _N.empty((lm["AQ28scrs"].shape[0], len(all_features)))
#            "pfrm_change36", "pfrm_change69", "pfrm_change912"]
iX           = -1
for feat in features:
    iX += 1 
    #exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : feat})
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : feat})
    exec("X[:, %(ix)d] = %(f)s" % {"ix" : iX, "f" : feat})
for outc in outcomes:
    iX += 1 
    #exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : outc})
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : outc})
    exec("X[:, %(ix)d] = %(f)s" % {"ix" : iX, "f" : outc})    
for scrs in AQ28scores:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : scrs})
for rcf in rule_change_features:
    iX += 1
    #exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : rcf})
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : rcf})
    exec("X[:, %(ix)d] = %(f)s" % {"ix" : iX, "f" : rcf})    

#use_feats = ["CondOutc"]#, "CondOutc", "Both"]
use_feats = ["Both"]
iuf = -1
resids = _N.zeros((3, 16, 4))

targets    = ["SB:IM"]#, "SB:IM", "SB:SS", "AQ28"]
#targets    = ["Switch", "Imag", "Soc_skils", "AQ28"]
#targets    = ["Soc_skils"]
#targets    = ["Imag"]
#targets    = ["Switch"]
reps       = 40
#AQ28scrs = soc_skils + imag + switch
X /= _N.std(X, axis=0)
X -= _N.mean(X, axis=0)

#allscores  = _N.empty((4, reps))

# ths = _N.arange(AQ28scrs.shape[0], dtype=_N.int)
# trainSz = 60
# partitions = 100
# all_trainingInds = _N.empty((partitions, trainSz), dtype=_N.int)
# #  For each shuffle,

pca = PCA()
pca.fit(X)
            
proj = _N.einsum("ni,mi->nm", pca.components_, X)
print(pca.explained_variance_ratio_)
maxC = _N.where(_N.cumsum(pca.explained_variance_ratio_) > min_var_expld)[0][0]

# ti = -1
# for target in targets:
#     ti += 1
#     pcspvs = _N.zeros((partitions, 3, 2))
#     #  compare for me AQ28  - just RC, just CO, BOTH
#     #  for sTarget in targets:

#     iuf = -1
#     for uf in use_feats:
#         iuf += 1
#         ths = _N.arange(AQ28scrs.shape[0])

#         # ######################################
#         Xs_trainSw = _N.empty((ths.shape[0], 5))
#         Xs_trainSw[:, 4] = isis_corr[ths]
#         Xs_trainSw[:, 0] = pfrm_change69[ths]
#         Xs_trainSw[:, 1] = win_aft_tie[ths]
#         Xs_trainSw[:, 2] = u_or_d_tie[ths]
#         Xs_trainSw[:, 3] = entropyW2[ths]
#         Xs_trainSw[:, 3] = stay_tie[ths]
#         Xs_trainSw[:, 3] = tie_aft_tie[ths]
#         ys_trainSw = switch[ths]
#         # ######################################
#         Xs_trainI = _N.empty((ths.shape[0], 6))
#         Xs_trainI[:, 0] = entropyU[ths]
#         Xs_trainI[:, 1] = pfrm_change69[ths]        
#         Xs_trainI[:, 2] = u_or_d_tie[ths]
#         Xs_trainI[:, 3] = win_aft_tie[ths]
#         Xs_trainI[:, 4] = stay_tie[ths]
#         Xs_trainI[:, 5] = los_aft_tie[ths]        
#         ys_trainI = imag[ths]

#         if target == "SB:SW":                
#             Xs_train = _N.array(Xs_trainSw)
#             ys_train = _N.array(ys_trainSw)
#         elif target == "SB:IM":                
#             Xs_train = _N.array(Xs_trainI)
#             ys_train = _N.array(ys_trainI)
#         Xs_train = Xs_train / _N.std(Xs_train, axis=0)
#         Xs_train = Xs_train - _N.mean(Xs_train, axis=0)
        
#         ys_train= (ys_train - _N.mean(ys_train)) / _N.std(ys_train)

#         clf = _skl.LinearRegression()

#         nrep = 100

#         biggest  = -1900
#         allVars = _N.arange(8)        
#         #for n in range(3, 4):
#         #    perms = itertools.permutations(allVars, n)
#             # for it in perms:
#             #     use_this = Xs_train[:, _N.array(it)]
#             #     rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)
#             #     scores = cross_val_score(clf, use_this, ys_train, cv=rkf)
#             #     print(_N.mean(scores))


#         print("correlations.................")            
#         for ch in range(6):
#             pc, pv = _ss.pearsonr(Xs_train[:, ch], ys_trainI)
#             print("%(ch)d    %(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv, "ch" : ch})
#         print("----------------------------------")
#         print("0, 3, 4")
#         use_this1 = Xs_train[:, _N.array([0])]
#         use_this2 = Xs_train[:, _N.array([3])]
#         use_this3 = Xs_train[:, _N.array([4])]        
#         use_this4 = Xs_train[:, _N.array([0, 3])]
#         use_this5 = Xs_train[:, _N.array([3, 4])]
#         use_this6 = Xs_train[:, _N.array([0, 3, 4])]        

#         for use_this in [use_this1, use_this2, use_this3, use_this4, use_this5, use_this6]:
#             rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)
#             scores = cross_val_score(clf, use_this, ys_train, cv=rkf)
#             print(_N.mean(scores))
#         print("###################################")        
            
#         print("2, 5")
#         use_this1 = Xs_train[:, _N.array([2])]
#         use_this2 = Xs_train[:, _N.array([5])]
#         use_this3 = Xs_train[:, _N.array([2, 5])]

#         for use_this in [use_this1, use_this2, use_this3]:
#             rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)
#             scores = cross_val_score(clf, use_this, ys_train, cv=rkf)
#             print(_N.mean(scores))
#         print("------------------------------------")            
                
#         use_this1 = Xs_train[:, _N.array([0, 2, 3, 4, 5])]
#         print("0, 2, 3, 4, 5")

#         for use_this in [use_this1, ]:
#             rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)
#             scores = cross_val_score(clf, use_this, ys_train, cv=rkf)
#             print(_N.mean(scores))

#         print("------------------------------------")
#         print("0, 2")
#         use_this1 = Xs_train[:, _N.array([0, 2])]


#         for use_this in [use_this1, ]:
#             rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)
#             scores = cross_val_score(clf, use_this, ys_train, cv=rkf)
#             print(_N.mean(scores))

#         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         #clean02  = Xs_train[:, 0] + Xs_train[:, 2] - Xs_train[:, 1]
#         clean02  = Xs_train[:, 2] - Xs_train[:, 1]
#         clean02  = clean02.reshape((Xs_train.shape[0], 1))
#         for use_this in [clean02 ]:
#             rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)
#             scores = cross_val_score(clf, use_this, ys_train, cv=rkf)
#             print(_N.mean(scores))

#         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")            
#         clean03  = _N.empty((Xs_train.shape[0], 2))
#         clean03[:, 0]  = Xs_train[:, 3] + Xs_train[:, 4]
#         clean03[:, 1]  = Xs_train[:, 0] + Xs_train[:, 2] - Xs_train[:, 1] + Xs_train[:, 5] 
            
#         for use_this in [clean03 ]:
#             rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)
#             scores = cross_val_score(clf, use_this, ys_train, cv=rkf)
#             print(_N.mean(scores))
