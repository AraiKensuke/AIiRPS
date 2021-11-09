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
           
lm = depickle("predictAQ28dat/AQ28_vs_RPS.dmp")

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
    exec("%(ca)s_s = standardize(temp)" % {"ca" : ca})
for scrs in AQ28scores:
    exec("%(f)s = lm[\"%(f)s\"]" % {"f" : scrs})

####################  USE ALL DATA
filtdat = _N.arange(AQ28scrs.shape[0])
####################  OUR DATA HAS 1 very strong outlier - 1 person with 
####################  AQ-28 score of 30 or so.  Next closest person has 
####################  score of 42 (or so).  
####################  using these 
filtdat = _N.where((AQ28scrs > 35) & (rout > 4))[0]
print("Using %(fd)d of %(all)d participants" % {"fd" : len(filtdat), "all" : AQ28scrs.shape[0]})

X_all_feats            = _N.empty((len(filtdat), len(cmp_againsts)))
y    = AQ28scrs[filtdat]
iaf = -1
for af in cmp_againsts:
    iaf += 1
    exec("feat = %s_s" % af)
    X_all_feats[:, iaf] = feat[filtdat]

REPS = 30
scrs = _N.empty(REPS*4)
coeffs= _N.empty((REPS*4, len(cmp_againsts)))

datinds = _N.arange(len(filtdat))
rkf = RepeatedKFold(n_splits=4, n_repeats=REPS)#, random_state=0)

iii = -1

these = {}
ichosen = _N.zeros(len(cmp_againsts), dtype=_N.int)
for train, test in rkf.split(datinds):
    iii += 1
    ####  first, pick alpha using LassoCV
    reg = LassoCV(cv=5, max_iter=100000).fit(X_all_feats[train], y[train])
    chosen = _N.where(_N.abs(reg.coef_) > 0.0001)[0]
    print(reg.alpha_)
    #print(_N.array(cmp_againsts)[chosen])
    for ic in range(len(chosen)):
        try:
            these[cmp_againsts[chosen[ic]]] += 1
            
        except KeyError:
            these[cmp_againsts[chosen[ic]]] = 1
        ichosen[chosen[ic]] += 1

clf = _skl.LinearRegression()
nrep = 100

for use_thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    use_features = _N.where(ichosen > int((iii+1)*use_thresh))[0]
    Xs_train = _N.empty((X.shape[0], len(use_features)))

    fi = -1
    for i_feat_indx in use_features:
        fi += 1
        Xs_train[:, fi] = X_all_feats[:, i_feat_indx]

    print("ut thresh %.2f" % use_thresh)
    for ns in [3, 4, 5]:
        rkf = RepeatedKFold(n_splits=4, n_repeats=nrep)#, random_state=0)
        scores = cross_val_score(clf, Xs_train, y, cv=rkf)
        print("     ns %(ns)d    %(mn).4f  %(md).4f" % {"mn" : _N.mean(scores), "md" : _N.median(scores), "ns" : ns})
            
