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
cmp_againsts = features_cab1 + features_cab2 + features_AI + features_stat

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

##  using custom list of answers
for scrs in AQ28scores[1:]:   #  raw answers data
    exec("ans_%(f)s = lm[\"ans_%(f)s\"]" % {"f" : scrs})
    exec("%(f)s     = _N.sum(ans_%(f)s[:, %(f)s_use], axis=1)" % {"f" : scrs})

#AQ28scrs = soc_skils + imag + switch + rout + fact_pat
    
#imag = _N.sum(ans_imag[:, imag_use], axis=1)
#switch = _N.sum(ans_switch[:, switch_use], axis=1)
#soc_skils = _N.sum(ans_soc_skils[:, soc_skils_use], axis=1)
    
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

starget = "imag"
exec("target = %s" % starget)

y    = target[filtdat]
#_N.random.shuffle(y)
iaf = -1
for af in cmp_againsts:
    iaf += 1
    exec("feat = %s_s" % af)
    X_all_feats[:, iaf] = feat[filtdat]

REPS = 20
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
for cv_thresh in [1.4, 1.2, 1]:
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
ichosen120 = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0) ) < 1.2)[0]
ichosen1 = _N.where(_N.std(reg_coefs, axis=0) / _N.abs(_N.mean(reg_coefs, axis=0) ) < 1)[0]

#  rout
"""
ichosen140  = _N.array([ 1, 18, 20, 30, 31, 32, 37, 39, 63, 68])
ichosen120  = _N.array([ 1, 20, 30, 31, 32, 37, 39, 63])
ichosen1  = _N.array([ 1, 20, 30, 31, 32, 37, 39, 63])

ichosen1 = _N.array([ 6, 22, 23, 29, 33, 49, 55, 58])
ichosen120= _N.array([ 2,  6, 22, 23, 29, 33, 49, 55, 58, 69])
ichosen140= _N.array([ 2,  6, 21, 22, 23, 29, 33, 43, 49, 55, 58, 69])

ichosen140  = _N.array([34, 35, 36, 37, 38])
ichosen120  = _N.array([35, 36, 37, 38])
ichosen1  = _N.array([35, 37, 38])
"""
chosen_s = ["1", "1.2", "1.4"]
chosen_features140 = _N.array(cmp_againsts)[ichosen140]
chosen_features120 = _N.array(cmp_againsts)[ichosen120]
chosen_features1 = _N.array(cmp_againsts)[ichosen1]
clf = _skl.LinearRegression()
#clf = _skl.TheilSenRegressor()
nrep = 50

fig = _plt.figure(figsize=(9, 3))
iu  = 0

scores_thresh = []   #  each thresh, 3 different folds
_plt.suptitle("target=\"%s\"  (grey=LinearRegr, black=RidgeRegr)" % starget)
for use_features in [ichosen1, ichosen120, ichosen140]:
    iu += 1
    scores_folds = []
    ax = fig.add_subplot(1, 3, iu)
    _plt.title(chosen_s[iu-1])
    #use_features = _N.where(ichosen > int((iii+1)*use_thresh))[0]
    Xs_train = _N.empty((X_all_feats.shape[0], len(use_features)))

    fi = -1
    if len(use_features) > 0:
        for i_feat_indx in use_features:
            fi += 1
            Xs_train[:, fi] = X_all_feats[:, i_feat_indx]

        for ns in [3, 4, 5]:
            xs = 0.22*_N.random.randn(ns*nrep)
            coefsLR = _N.empty((nrep*ns, len(use_features)))
            #test_sz = ns*(len(filtdat)//ns)-(ns-1)*(len(filtdat)//ns)
            test_sz = len(filtdat)//ns + 1 if len(filtdat) % ns != 0 else len(filtdat)//ns
            print("test_sz   %d" % test_sz)
            obs_v_preds = _N.zeros((nrep*ns, test_sz, 2))

            scoresLR = _N.empty(nrep*ns)
            rkf = RepeatedKFold(n_splits=ns, n_repeats=nrep)#, random_state=0)
            iii = -1

            for train, test in rkf.split(datinds):
                iii += 1
                clf_f = clf.fit(Xs_train[train], y[train])
                scoresLR[iii] = clf_f.score(Xs_train[test], y[test])
                #coefsLR[iii] = clf_f.coef_
                obs_v_preds[iii, 0:len(test), 0] = y[test]
                obs_v_preds[iii, 0:len(test), 1] = clf_f.predict(Xs_train[test])

            # scoresLR  = cross_val_score(clf, Xs_train, y, cv=rkf)

            scores_folds.append([scoresLR, coefsLR, obs_v_preds])

            print("LR     ns %(ns)d    %(mn).4f  %(md).4f" % {"mn" : _N.mean(scoresLR), "md" : _N.median(scoresLR), "ns" : ns})
            #abs_cvs_of_coeffs = _N.abs(_N.std(coefsLR, axis=0) / _N.mean(coefsLR, axis=0))
            #outstrLR = str_float_array(abs_cvs_of_coeffs, "%.2f")

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
    scores_thresh.append(scores_folds)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
_plt.savefig("scores_%s" % starget)
_N.savetxt("use_features_%s" % starget, use_features, fmt="%d")


fi = -1

lbsz=13
s_gdns = [r"worst $r^2$", r"median $r^2$", r"best $r^2$"]
for use_features in [ichosen1, ichosen120, ichosen140]:
    fi += 1
    if len(use_features) > 0:
        ins = -1

        fig = _plt.figure(figsize=(9, 12))
        _plt.suptitle("target %(t)s   threshold %(th)s" % {"t" : starget, "th" : chosen_s[fi]})
        for ns in [3, 4, 5]:
            scores      = scores_thresh[fi][ns-3][0]
            obs_v_preds = scores_thresh[fi][ns-3][2][:, 0:-1]
            byScores = scores.argsort()

            xymin  = _N.min(obs_v_preds)
            xymax  = _N.max(obs_v_preds)

            test_sz = len(filtdat)//ns + 1 if len(filtdat) % ns != 0 else len(filtdat)//ns
            
            ax = fig.add_subplot(4, 3, (ns-3)+1)
            _plt.title("%d folds" % ns)
            xs = 0.22*_N.random.randn(scores.shape[0])
            igd0 = 0
            igd1 = (len(byScores)//2)
            igd2 = (len(byScores)//2)*2 - 1
            _plt.scatter(xs[0:igd2], scores[0:igd2], color="#CCCCCC", s=3)
            _plt.scatter(xs[byScores[igd0]], scores[byScores[igd0]], color="black", s=10)
            _plt.scatter(xs[byScores[igd1]], scores[byScores[igd1]], color="black", s=10)            
            _plt.scatter(xs[byScores[igd2]], scores[byScores[igd2]], color="black", s=10)
            mnLR = _N.mean(scores)
            mdLR = _N.median(scores)        
            _plt.plot([1.5, 3.5], [mnLR, mnLR], color="blue", lw=3)
            _plt.plot([1.5, 3.5], [mdLR, mdLR], color="red", lw=3)
            _plt.axhline(y=0, ls="--", color="grey")
            _plt.xlim(-4, 4)
            _plt.ylim(-0.5, 0.5)
            _plt.xticks([])
            _plt.ylabel(r"$r^2$", rotation=0, fontsize=lbsz)
            
            
            for gdns in range(3): #  1 4 7    2 5 8
                igd = (len(byScores)//2)*gdns
                igd = igd if gdns < 2 else igd-1
                idisp_gdns = 2 - gdns
                ax = fig.add_subplot(4, 3, 3*(idisp_gdns+1)+(ns-3)+1)
                ax.set_aspect("equal")
                _plt.xlim(xymin-1, xymax+1)
                _plt.ylim(xymin-1, xymax+1)
                y_test = obs_v_preds[byScores[igd], 0:test_sz-1, 0]
                
                y_pred = obs_v_preds[byScores[igd], 0:test_sz-1, 1]
                A = _N.vstack([y_test, _N.ones(y_test.shape[0])]).T
                m, c = _N.linalg.lstsq(A, y_pred, rcond=-1)[0]

                src, srv = _ss.spearmanr(y_test, y_pred)
                pc, pv = _ss.pearsonr(y_test, y_pred)                
                _plt.scatter(y_test+0.2*_N.random.randn(len(y_test)), y_pred, s=3, color="black")
                _plt.plot([xymin-1, xymax+1], [xymin-1, xymax+1], ls=":")
                _plt.plot([xymin-1, xymax+1], [(xymin-1)*m + c, (xymax+1)*m + c], ls="-", color="red")
                _plt.title(s_gdns[gdns])
                _plt.text(0.2*(xymax-xymin), 0.8*(xymax-xymin) + xymin, "%(sp).2f  %(pc).2f" % {"sp" : src, "pc" : pc})
                _plt.xlabel("obsverved score", fontsize=lbsz)
                _plt.ylabel("predicted score", fontsize=lbsz)
        fig.subplots_adjust(hspace=0.62, wspace=0.55, bottom=0.07, left=0.07, right=0.99)                
        _plt.savefig("predict_%(t)s_%(th)s.png" % {"t" : starget, "th" : chosen_s[fi]})
