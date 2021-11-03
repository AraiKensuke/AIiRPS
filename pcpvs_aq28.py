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


def predict_new(Xs, y, fitdat):
    N  = y.shape[0]
    testdat = _N.setdiff1d(_N.arange(N), fitdat)
    Xs_fit = Xs[fitdat]       #  
    y_fit  = y[fitdat]        #  target
    model = _skl.LinearRegression()
    model.fit(Xs_fit, y_fit)
    
    model_in = model.intercept_
    model_as = model.coef_
    mean_y_fit = _N.mean(y[fitdat])
    Xs_test = Xs[testdat]
    y_test  = y[testdat]
    y_fit_pred = _N.dot(Xs_fit, model_as) + model_in
    fit_resid = _N.sum((y_fit_pred - y_fit)**2) / len(y_fit)
    fit_resid0 = _N.sum((mean_y_fit - y_fit)**2) / len(y_fit)
    
    y_pred = _N.dot(Xs_test, model_as) + model_in
    test_resid = _N.sum((y_pred - y_test)**2) / len(y_pred)
    test_resid0 = _N.sum((mean_y_fit - y_test)**2) / len(y_pred)
    return Xs_test, y_test, y_pred#, fit_resid, fit_resid0, test_resid, test_resid0

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

lm = depickle("AQ28_vs_RPS.dmp")

AQ28scores = ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs"]
AQ28scores_ab = ["SB:SS", "SB:IM", "SB:RT", "SB:SW", "NumPats", "AQ28"]

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
#     _plt.xticks(_N.arange(10), ["NetW", "WaT", "TaT", "LaT",
#                 "WaW", "TaW", "LaW",
#                 "WaL", "TaL", "LaL"])

features = ["entropyT2", "entropyW2", "entropyL2", "entropyUD", "entropyS",
            "isis_corr", "isis_cv"]
#            "pfrm_change36", "pfrm_change69", "pfrm_change912"]
for feat in features:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : feat})
for outc in outcomes:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : outc})
for scrs in AQ28scores:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : scrs})
for rcf in rule_change_features:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : rcf})

nks = len(lm["pcpvs"])

pcpvs = _N.empty((nks, 2, 6))

ik = -1
for key in lm["pcpvs"].keys():
    ik += 1
    #  ss, imag, rout, switch, fact_pat, AQ28
    pcpvs[ik, 0] = lm["pcpvs"][key][0]
    pcpvs[ik, 1] = lm["pcpvs"][key][1]

"""
fig = _plt.figure(figsize=(6.2, 5))

for i in range(6):
    ax = fig.add_subplot(3, 2, i+1)
    _plt.title(AQ28scores_ab[i], fontsize=17)
    nsigs = _N.where(pcpvs[:, 1, i] >= 0.05)[0]
    
    _plt.scatter(nsigs, pcpvs[nsigs, 0, i], s=30, color="grey", marker=".")
    sigs = _N.where(pcpvs[:, 1, i] < 0.05)[0]
    _plt.scatter(sigs, pcpvs[sigs, 0, i], s=110, color="blue", marker=".")
    _plt.ylim(-0.36, 0.36)
    _plt.axhline(y=0, ls="--", color="black")
    _plt.axhline(y=-0.2, ls=":", color="black")
    _plt.axhline(y=0.2, ls=":", color="black")
    _plt.yticks([-0.2, 0, 0.2], fontsize=13)
    _plt.xticks(fontsize=13)
    if i % 2 == 0:
        _plt.ylabel("Corr coeff.", fontsize=15)

    if i == 5:
        ax.spines["left"].set_linewidth(4)
        ax.spines["right"].set_linewidth(4)
        ax.spines["top"].set_linewidth(4)
        ax.spines["bottom"].set_linewidth(4)

    if (i == 4) or (i == 5):
        _plt.xlabel("Feature #", fontsize=15)
              
fig.subplots_adjust(hspace=0.55, left=0.14, right=0.98, bottom=0.12, top=0.93)
_plt.savefig("summary_corrs")
"""
#########  displayed sideways
fig = _plt.figure(figsize=(9.5, 2.4))

for i in range(6):
    ax = fig.add_subplot(1, 6, i+1)
    _plt.title(AQ28scores_ab[i], fontsize=17)
    nsigs = _N.where(pcpvs[:, 1, i] >= 0.05)[0]
    
    _plt.scatter(nsigs, pcpvs[nsigs, 0, i], s=30, color="grey", marker=".")
    sigs = _N.where(pcpvs[:, 1, i] < 0.05)[0]
    _plt.scatter(sigs, pcpvs[sigs, 0, i], s=110, color="blue", marker=".")
    _plt.ylim(-0.41, 0.41)
    _plt.axhline(y=0, ls="--", color="black")
    _plt.axhline(y=-0.2, ls=":", color="black")
    _plt.axhline(y=0.2, ls=":", color="black")
    if i == 0:
        _plt.yticks([-0.4, -0.2, 0, 0.2, 0.4], fontsize=13)
    else:
        _plt.yticks([], fontsize=13)
    _plt.xticks(fontsize=13)
    if i == 0:
        _plt.ylabel("Corr coeff.", fontsize=15)

    if i == 5:
        ax.spines["left"].set_linewidth(4)
        ax.spines["right"].set_linewidth(4)
        ax.spines["top"].set_linewidth(4)
        ax.spines["bottom"].set_linewidth(4)
    _plt.xlabel("Feature #", fontsize=15)
              
fig.subplots_adjust(wspace=0.2, left=0.1, right=0.98, bottom=0.23, top=0.87)
_plt.savefig("summary_corrs")

