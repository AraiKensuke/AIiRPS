import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
from sklearn import linear_model
import sklearn.linear_model as _skl

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

AQ28scores = ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]
AQ28scores_ab = ["AQ28", "SS", "IM", "RT", "SW", "FP"]

outcomes = ["netwins",
            "win_aft_win", "tie_aft_win", "los_aft_win",            
            "win_aft_tie", "tie_aft_tie", "los_aft_tie",
            "win_aft_los", "tie_aft_los", "los_aft_los",
            "u_or_d_tie", "u_or_d_res"]
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
    exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : feat})
for outc in outcomes:
    exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : outc})
for scrs in AQ28scores:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : scrs})
for rcf in rule_change_features:
    exec("%(f)s = lm[\"AQ28_%(f)s\"][:, 1]" % {"f" : rcf})

labsz=15
tksz=13
for of in range(2):   #  rule change OR conditional outcome features
    if of == 0:
        fig = _plt.figure(figsize=(5.5, 11))
    elif of == 1:
        fig = _plt.figure(figsize=(3, 11))
    fi  = -1
    for sfeat in features:   #  behavioral features
        fi += 1
        fig.add_subplot(7, 1, fi+1)
        feat = eval(sfeat)
        colors = ["#CCCCCC"] * len(outcomes)
        io = -1

        outcomes_or_AQ = outcomes if of == 0 else AQ28scores
        pcpvs = _N.empty((len(outcomes_or_AQ), 2))
        
        for sooA in outcomes_or_AQ:
            ooA = eval(sooA)
            io += 1
            pcpvs[io] = _ss.pearsonr(feat, ooA)
            if pcpvs[io, 1] < 0.05:
                colors[io] = "black"
        _plt.bar(range(len(outcomes_or_AQ)), pcpvs[:, 0], color=colors)


        #_plt.xticks(range(len(outcomes)), outcome_ab)
        if of == 0:
            _plt.ylim(-0.6, 0.6)
            yticks = _N.array([-0.5, -0.25, 0, 0.25, 0.5])            
            _plt.axvline(x=0.5, ls="--")
            _plt.axvline(x=3.5, ls="--")
            _plt.axvline(x=6.5, ls="--")
            _plt.axvline(x=9.5, ls="--")
            _plt.xticks(range(len(outcomes_or_AQ)), outcome_ab)
            _plt.yticks(yticks)
            for yt in yticks:
                _plt.axhline(y=yt, ls=":", color="grey")
        else:
            _plt.ylim(-0.32, 0.32)
            yticks = _N.array([-0.3, -0.15, 0, 0.15, 0.3])
            _plt.yticks(yticks)
            for yt in yticks:
                _plt.axhline(y=yt, ls=":", color="grey")
            _plt.axvline(x=0.5, ls="--")
            _plt.xticks(range(len(outcomes_or_AQ)), AQ28scores_ab)            
        _plt.title(sfeat)
    fig.subplots_adjust(bottom=0.07, top=0.94, hspace=0.6, left=0.15, right=0.96)
    _plt.savefig("summaries_%d" % of, transparent=True)

use_feats = ["RCfeat", "CondOutc", "Both"]
#use_feats = ["Both"]
iuf = -1
resids = _N.zeros((3, 16, 4))

targets    = ["Switch", "Imag", "Soc_skils", "AQ28"]

ths = _N.arange(AQ28scrs.shape[0], dtype=_N.int)
trainSz = 60
partitions = 100
all_trainingInds = _N.empty((partitions, trainSz), dtype=_N.int)
#  For each shuffle,
for tr in range(partitions):
    #  want to compare 
    all_trainingInds[tr] =  _N.random.choice(_N.arange(len(ths)), size=trainSz, replace=False)

for target in targets:
    pcspvs = _N.zeros((partitions, 3, 2))
    #  compare for me AQ28  - just RC, just CO, BOTH
    #  for sTarget in targets:

    for it in range(partitions):
        trainingInds = all_trainingInds[it]
        iuf = -1
        for uf in use_feats:
            iuf += 1
            ths = _N.arange(AQ28scrs.shape[0])
            if uf == "RCfeat":
                # ######################################
                Xs_trainSw = _N.empty((ths.shape[0], 3))
                Xs_trainSw[:, 0] = win_aft_tie[ths]
                Xs_trainSw[:, 1] = u_or_d_tie[ths]
                Xs_trainSw[:, 2] = tie_aft_tie[ths]
                ys_trainSw = switch[ths]
                # ######################################
                Xs_trainI = _N.empty((ths.shape[0], 2))
                Xs_trainI[:, 0] = u_or_d_tie[ths]
                Xs_trainI[:, 1] = win_aft_tie[ths]
                ys_trainI = imag[ths]
                # ######################################                
                Xs_trainSS = _N.empty((ths.shape[0], 2))
                Xs_trainSS[:, 0] = tie_aft_tie[ths]
                Xs_trainSS[:, 1] = win_aft_tie[ths]
                ys_trainSS = soc_skils[ths]
                
                #############################
                #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
                Xs_trainA = _N.empty((ths.shape[0], 3))
                Xs_trainA[:, 0] = win_aft_tie[ths]
                Xs_trainA[:, 1] = tie_aft_tie[ths]
                Xs_trainA[:, 2] = u_or_d_tie[ths]
                #Xs_trainA[:, 3] = cmp_vs[ths]
                ys_trainA = AQ28scrs[ths]
            elif uf == "CondOutc":
                # ######################################
                Xs_trainSw = _N.empty((ths.shape[0], 2))
                Xs_trainSw[:, 0] = isis_corr[ths]
                Xs_trainSw[:, 1] = pfrm_change69[ths]
                ys_trainSw = switch[ths]
                # ######################################
                Xs_trainI = _N.empty((ths.shape[0], 3))
                Xs_trainI[:, 0] = entropyT2[ths]
                Xs_trainI[:, 1] = pfrm_change69[ths]
                Xs_trainI[:, 2] = entropyUD[ths]
                ys_trainI = imag[ths]
                #############################
                #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
                Xs_trainSS = _N.empty((ths.shape[0], 4))
                Xs_trainSS[:, 0] = entropyUD[ths]
                Xs_trainSS[:, 1] = entropyT2[ths]
                Xs_trainSS[:, 2] = pfrm_change69[ths]
                Xs_trainSS[:, 3] = moresim[ths]
                ys_trainSS = soc_skils[ths]
                # #############################
                #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
                Xs_trainA = _N.empty((ths.shape[0], 3))
                Xs_trainA[:, 0] = entropyT2[ths]
                Xs_trainA[:, 1] = entropyUD[ths]
                Xs_trainA[:, 2] = pfrm_change69[ths]
                ys_trainA = AQ28scrs[ths]
            elif uf == "Both":
                # ######################################
                Xs_trainSw = _N.empty((ths.shape[0], 5))
                Xs_trainSw[:, 0] = isis_corr[ths]
                Xs_trainSw[:, 1] = pfrm_change69[ths]
                Xs_trainSw[:, 2] = win_aft_tie[ths]
                Xs_trainSw[:, 3] = u_or_d_tie[ths]
                Xs_trainSw[:, 4] = tie_aft_tie[ths]
                ys_trainSw = switch[ths]
                # ######################################
                Xs_trainI = _N.empty((ths.shape[0], 5))
                Xs_trainI[:, 0] = entropyT2[ths]
                Xs_trainI[:, 1] = pfrm_change69[ths]
                Xs_trainI[:, 2] = entropyUD[ths]
                Xs_trainI[:, 3] = u_or_d_tie[ths]
                Xs_trainI[:, 4] = win_aft_tie[ths]
                ys_trainI = imag[ths]
                #############################
                #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
                Xs_trainSS = _N.empty((ths.shape[0], 6))
                Xs_trainSS[:, 0] = entropyUD[ths]
                Xs_trainSS[:, 1] = entropyT2[ths]
                #Xs_trainSS[:, 1] = cmp_vs[ths]
                Xs_trainSS[:, 2] = pfrm_change69[ths]
                Xs_trainSS[:, 3] = moresim[ths]
                Xs_trainSS[:, 4] = tie_aft_tie[ths]
                Xs_trainSS[:, 5] = win_aft_tie[ths]
                ys_trainSS = soc_skils[ths]
                # #############################
                #cmp_vs = _N.log(_N.sum(sum_sd[:, :, 0], axis=1) + _N.sum(sum_sd[:, :, 2], axis=1))
                Xs_trainA = _N.empty((ths.shape[0], 6))
                Xs_trainA[:, 0] = entropyT2[ths]
                Xs_trainA[:, 1] = entropyUD[ths]
                Xs_trainA[:, 2] = pfrm_change69[ths]
                Xs_trainA[:, 3] = win_aft_tie[ths]
                Xs_trainA[:, 4] = tie_aft_tie[ths]
                Xs_trainA[:, 5] = u_or_d_tie[ths]
                #Xs_trainA[:, 3] = cmp_vs[ths]
                ys_trainA = AQ28scrs[ths]
                
            if target == "AQ28":
                Xs_train = Xs_trainA
                ys_train = ys_trainA
            elif target == "Imag":                
                Xs_train = Xs_trainI
                ys_train = ys_trainI
            elif target == "Switch":                
                Xs_train = Xs_trainSw
                ys_train = ys_trainSw
            elif target == "Soc_skils":                
                Xs_train = Xs_trainSS
                ys_train = ys_trainSS

            Xs_test, y_test, y_pred = predict_new(Xs_train, ys_train, trainingInds)
            pc, pv = _ss.pearsonr(y_test, y_pred)
            pcspvs[it, iuf, 0] = pc
            pcspvs[it, iuf, 1] = pv            
            #print("%(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv})
            A = _N.vstack([y_test, _N.ones(y_test.shape[0])]).T
            m, c = _N.linalg.lstsq(A, y_pred, rcond=-1)[0]

    #fig = _plt.figure(figsize=(8, 4))
    fig = _plt.figure(figsize=(2.2, 4))
    #_plt.suptitle("tar=%(tar)s  trsz=%(trn)d tstsz=%(tst)d" % {"tar" : target, "trn" : trainSz, "tst" : (len(ths)-trainSz)})
    _plt.suptitle("pred. tar=%(tar)s" % {"tar" : target, "trn" : trainSz, "tst" : (len(ths)-trainSz)})
    #ax = _plt.subplot2grid((1, 5), (0, 0))
    #ax = fig.add_subplot(1, 2, 1)
    for ip in range(partitions):
        color="orange"
        if (pcspvs[ip, 0, 0] < pcspvs[ip, 2, 0]) and (pcspvs[ip, 1, 0] < pcspvs[ip, 2, 0]):
            color="black"
        _plt.plot([0, 1, 2], [pcspvs[ip, 0, 0], pcspvs[ip, 1, 0], pcspvs[ip, 2, 0]], color=color, zorder=0)
    _plt.ylim(0, 0.6)
    _plt.xticks([0, 1, 2], ["RC", "ST", "RC+ST"], fontsize=tksz)
    _plt.yticks(fontsize=tksz)    
    iuf = -1
    for iuf in range(3):
        y = _N.mean(pcspvs[:, iuf, 0])
        #_plt.plot([iuf-0.2, iuf+0.2], [y, y], lw=3, color="blue")
        _plt.scatter([iuf], [y], marker=".", s=400, color="blue", zorder=1)
    _plt.ylabel("CCoeff, pred vs obsvd score", fontsize=labsz)
    _plt.xlabel("covariates used", fontsize=labsz)
    fig.subplots_adjust(left=0.31, bottom=0.17)
    
    _plt.savefig("pred_%s" % target)
    

    # ax = _plt.subplot2grid((1, 5), (0, 1), colspan=4)        
    # #ax = fig.add_subplot(1, 2, 2)
    # for ip in range(partitions):
    #     iuf = -1
    #     for uf in use_feats:
    #         iuf += 1
    #         signf  = _N.where(pcspvs[:, iuf, 1] < 0.05)[0]
    #         nsignf = _N.where(pcspvs[:, iuf, 1] >= 0.05)[0]
    #         ms = 5 if iuf == 2 else 1
            
    #         if len(signf) > 0:
    #             _plt.scatter(signf, pcspvs[signf, iuf, 0], color="black", s=ms)
    #         if len(nsignf) > 0:
    #             _plt.scatter(nsignf, pcspvs[nsignf, iuf, 0], color="#CDCDCD", s=ms)
    # _plt.ylim(0, 0.6)                
        
