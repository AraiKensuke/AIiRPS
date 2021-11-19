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
    sk = _N.empty(15)
    im = -1
    ms = _N.linspace(0.01, 1.1, 15)
    for m in ms:
        im += 1
        sk[im] = _ss.skew(_N.exp(dat / (m*_N.mean(dat))))
    min_im = _N.where(_N.abs(sk) == _N.min(_N.abs(sk)))[0][0]
    return _N.exp(dat / (ms[min_im]*_N.mean(dat)))
           
lm = depickle("predictAQ28dat/AQ28_vs_RPS_1.dmp")

AQ28scores = ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]
AQ28scores_ab = ["AQ28", "SS", "IM", "RT", "SW", "FP"]

features_cab = lm["features_cab"]
features_stat = lm["features_stat"]
cmp_againsts = features_cab + features_stat

######  unskew and standardize the features to use.
for ca in cmp_againsts:
    exec("temp = lm[\"%(ca)s\"]" % {"ca" : ca})
    exec("%(ca)s = lm[\"%(ca)s\"]" % {"ca" : ca})    
    if ca[0:7] == "entropy":
        exec("temp = unskew(temp)" % {"ca" : ca})
    print(ca)
    exec("%(ca)s_s = standardize(temp)" % {"ca" : ca})
for scrs in AQ28scores:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : scrs})

####################  USE ALL DATA
filtdat = _N.arange(AQ28scrs.shape[0])
####################  OUR DATA HAS 1 very strong outlier - 1 person with 
####################  AQ-28 score of 30 or so.  Next closest person has 
####################  score of 42 (or so).  
####################  using these 
#filtdat = _N.where((AQ28scrs > 35) & (rout > 4))[0]
filtdat = _N.where((AQ28scrs > 35))[0]
print("Using %(fd)d of %(all)d participants" % {"fd" : len(filtdat), "all" : AQ28scrs.shape[0]})

X_all_feats            = _N.empty((len(filtdat), len(cmp_againsts)))

starget = "switch"
exec("target = %s" % starget)

y    = target[filtdat]
iaf = -1
for af in cmp_againsts:
    iaf += 1
    exec("feat = %s_s" % af)
    X_all_feats[:, iaf] = feat[filtdat]

REPS = 30
scrs = _N.empty(REPS*4)
coeffs= _N.empty((REPS*4, len(cmp_againsts)))

n_splits = 5
datinds = _N.arange(len(filtdat))
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=REPS)#, random_state=0)
iii = -1
these = {}
ichosen = _N.zeros(len(cmp_againsts), dtype=_N.int)
reg_coefs = _N.empty((n_splits*REPS, len(cmp_againsts)))
for train, test in rkf.split(datinds):
    iii += 1
    ####  first, pick alpha using LassoCV
    reg = LassoCV(cv=4, max_iter=100000).fit(X_all_feats[train], y[train])

    maxWeight = _N.max(_N.abs(reg.coef_))
    reg_coefs[iii] = reg.coef_
    chosen = _N.where(_N.abs(reg.coef_) > maxWeight*0.05)[0]
    print(reg.alpha_)
    #print(_N.array(cmp_againsts)[chosen])
    for ic in range(len(chosen)):
        try:
            these[cmp_againsts[chosen[ic]]] += 1
        except KeyError:
            these[cmp_againsts[chosen[ic]]] = 1
        ichosen[chosen[ic]] += 1

##  WHICH FEATURES TO USE
fig = _plt.figure(figsize=(8, 11))
_plt.suptitle("target = %s" % starget)
ict = 0
arr_cmp_againsts = _N.array(cmp_againsts)
for cv_thresh in [1.4, 1.25, 1.1, 1]:
    ict += 1
    fig.add_subplot(4, 1, ict)
    ths_feats = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0)) < cv_thresh)[0]
    n_ths_feats = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0)) >= cv_thresh)[0]
    all_inds = _N.arange(len(cmp_againsts))
    all_ys   = _N.abs(_N.mean(reg_coefs, axis=0)) / _N.std(reg_coefs, axis=0)
    _plt.scatter(all_inds[n_ths_feats], all_ys[n_ths_feats], color="grey", s=3)
    _plt.scatter(all_inds[ths_feats], all_ys[ths_feats], color="black", s=13)
    _plt.xticks(all_inds[ths_feats], arr_cmp_againsts[ths_feats], rotation=70)
fig.subplots_adjust(hspace=1.1, left=0.05, right=0.98, top=0.94)
_plt.savefig("LASSO_features_4_%s" % starget)

ichosen140 = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0) ) < 1.4)[0]        
ichosen125 = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0) ) < 1.25)[0]
ichosen11 = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0) ) < 1.1)[0]
ichosen1 = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0) ) < 1)[0]

chosen_features140 = _N.array(cmp_againsts)[ichosen140]
chosen_features125 = _N.array(cmp_againsts)[ichosen125]
chosen_features11 = _N.array(cmp_againsts)[ichosen11]
chosen_features1 = _N.array(cmp_againsts)[ichosen1]
clf = _skl.LinearRegression()
nrep = 20

fig = _plt.figure(figsize=(10, 10))
iu  = 0
_plt.suptitle("target=%s  (grey=LinearRegr, black=RidgeRegr)" % starget)
for use_features in [ichosen1, ichosen11, ichosen125, ichosen140]:
    iu += 1
    fig.add_subplot(2, 2, iu)
    #use_features = _N.where(ichosen > int((iii+1)*use_thresh))[0]
    Xs_train = _N.empty((X_all_feats.shape[0], len(use_features)))

    fi = -1
    for i_feat_indx in use_features:
        fi += 1
        Xs_train[:, fi] = X_all_feats[:, i_feat_indx]

    for ns in [3, 4, 5]:
        xs = 0.22*_N.random.randn(ns*nrep)
        coefsLR = _N.empty((nrep*ns, len(use_features)))
        obs_v_preds = _N.empty((nrep*ns, 2))
        
        scoresLR = _N.empty(nrep*ns)
        rkf = RepeatedKFold(n_splits=ns, n_repeats=nrep)#, random_state=0)
        iii = -1
        for train, test in rkf.split(datinds):
            iii += 1
            clf_f = clf.fit(Xs_train[train], y[train])
            scoresLR[iii] = clf_f.score(Xs_train[test], y[test])
            coefsLR[iii] = clf_f.coef_
            obs_v_preds[iii, 0] = y[test]
            obs_v_preds[iii, 0] = clf_f.predict(Xs_train[test])
            
        # scoresLR  = cross_val_score(clf, Xs_train, y, cv=rkf)

        print("LR     ns %(ns)d    %(mn).4f  %(md).4f" % {"mn" : _N.mean(scoresLR), "md" : _N.median(scoresLR), "ns" : ns})
        abs_cvs_of_coeffs = _N.abs(_N.std(coefsLR, axis=0) / _N.mean(coefsLR, axis=0))
        outstrLR = str_float_array(abs_cvs_of_coeffs, "%.2f")

        mnLR = _N.mean(scoresLR)
        mdLR = _N.median(scoresLR)        
        _plt.scatter(xs+3*ns, scoresLR, color="grey", s=3)
        _plt.plot([3*ns-0.6, 3*ns+0.6], [mnLR, mnLR], color="blue", lw=3)
        _plt.plot([3*ns-0.6, 3*ns+0.6], [mdLR, mdLR], color="red", lw=3)
    _plt.xticks([9.5, 12.5, 15.5], [3, 4, 5], fontsize=14)
    _plt.yticks(fontsize=14)    
    _plt.xlabel("# folds", fontsize=16)
    _plt.ylabel(r"$r^2$", fontsize=16, rotation=0)
    _plt.ylim(-0.5, 0.5)
    _plt.axhline(y=0, ls=":")
fig.subplots_adjust(hspace=0.4, wspace=0.4)
_plt.savefig("scores_%s" % starget)
