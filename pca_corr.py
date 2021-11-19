import scipy.optimize as opt
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
from scipy.optimize import basinhopping

#def LC_pca_comp(w, target=None, pcaX=None):
def LC_pca_comp(w, pcaX, target):
    pc, pv = _ss.pearsonr(target, w[0]*pcaX[0]+w[1]*pcaX[1]+w[2]*pcaX[2]+w[3]*pcaX[3]+w[4]*pcaX[4]+w[5]*pcaX[5]+w[6]*pcaX[6]+w[7]*pcaX[7]+w[8]*pcaX[8])
    #print(pcaX)
    return 1 - pc

def corrcoeff(w, pcaX, target):
    #  target is size 184
    #  pcaX   is isze 184 x 9
    linsum = w[0]*pcaX[0]+w[1]*pcaX[1]+w[2]*pcaX[2]+w[3]*pcaX[3]+w[4]*pcaX[4]+w[5]*pcaX[5]+w[6]*pcaX[6]+w[7]*pcaX[7]+w[8]*pcaX[8]
    mn_linsum = 
    top = (target - _N.mean(target)) * 
    
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

features_cab = lm["features_cab"]
features_stat = lm["features_stat"]
cmp_againsts = features_cab + features_stat

iaf = -1
######  unskew and standardize the features to use.
for ca in cmp_againsts:
    exec("temp = lm[\"%(ca)s\"]" % {"ca" : ca})
    exec("%(ca)s = lm[\"%(ca)s\"]" % {"ca" : ca})    
    if ca[0:7] == "entropy":
        exec("temp = unskew(temp)" % {"ca" : ca})
    print(ca)
    exec("%(ca)s_s = standardize(temp)" % {"ca" : ca})

Xs            = _N.empty((184, len(cmp_againsts)))

for af in cmp_againsts:
    iaf += 1
    exec("feat = %s_s" % af)    
    Xs[:, iaf] = feat

pca = PCA()
pca.fit(Xs)
proj = _N.einsum("ni,mi->nm", pca.components_, Xs)
#maxC = _N.where(_N.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]
#data = Xs[:, 0:maxC]

#minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, "pcaX" : proj, "target" : lm["AQ28scrs"]}
#minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, "args" : (proj, lm["AQ28scrs"])}

"""
SHUFFLES = 10
inds = _N.zeros((SHUFFLES+1, 184), dtype=_N.int)
inds0 = _N.arange(184, dtype=_N.int)

inds[0] = inds0
for shf in range(1, SHUFFLES+1):
    _N.random.shuffle(inds0)
    inds[shf] = inds0

for starg in ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs"]:
    print(starg)    
    for shf in range(1+SHUFFLES):
        print("----------   %d" % shf)

        y = _N.array(lm[starg][inds[shf]])
            
        minimizer_kwargs = {"args" : (proj, y)}
        w0  = _N.ones(9)
        ret = basinhopping(LC_pca_comp, w0, minimizer_kwargs=minimizer_kwargs,
                           niter=1000)
        pc, pv = _ss.pearsonr(_N.sum(ret.x.reshape(9, 1) * proj[0:9], axis=0), y)
        print("%(pc).3f   %(pv).1e" % {"pc" : pc, "pv" : pv})
"""

SHUFFLES = 10
inds = _N.zeros((SHUFFLES+1, 184), dtype=_N.int)
inds0 = _N.arange(184, dtype=_N.int)

inds[0] = inds0
for shf in range(1, SHUFFLES+1):
    _N.random.shuffle(inds0)
    inds[shf] = inds0

for starg in ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs"]:
    print(starg)    
    for shf in range(1+SHUFFLES):
        print("----------   %d" % shf)

        y = _N.array(lm[starg][inds[shf]])

        opt.minimize_scalar(LC_pca_comp, 
        minimizer_kwargs = {"args" : (proj, y)}
        w0  = _N.ones(9)

        pc, pv = _ss.pearsonr(_N.sum(ret.x.reshape(9, 1) * proj[0:9], axis=0), y)
        print("%(pc).3f   %(pv).1e" % {"pc" : pc, "pv" : pv})


# # inds = \
# #     _N.array([125,  88,  90, 123,  35, 134, 102, 139, 182, 154, 161, 168, 133,
# #               140,  18, 105,  46, 130, 151,  78, 104, 153,  85,  24,   5,  21,
# #               171,  32,  75, 181,  54,  82,  27,  56,  69, 128,  59,  11, 101,
# #               92,  12,  84,  49, 107, 142,  28,  23,  26,  30, 155,  63, 116,
# #               129, 160, 163,   8,  44, 174, 179,  55, 172,  67,  14,  38,  31,
# #               183, 159, 138,  86,  25, 177, 132,  68,  95, 120,  36,   3,  17,
# #               58,  72, 108, 148,  29, 136, 141,  73,  76, 152,  42, 166, 157,
# #               119,  10, 109, 180, 137, 169,  40, 124,  77, 176,  41,  34,  22,
# #               4, 135, 103,  97, 122,  13, 156,  48,  96, 158,  15,  64, 144,
# #               127, 106, 113,  93,  74, 170, 178,  79,  16,  94, 121, 145,  43,
# #               98,  87,  65,  60,  52,  51,   2, 115, 117, 147,  70, 175, 162,
# #               110,   9, 111, 118,  20, 100, 165, 150,  71,  45,   7,  89, 131,
# #               39,   1,  19,  57,  91,  99,  83, 164, 149, 112,  33,  53, 146,
# #               37, 167, 143,  61,  50,  81, 114, 126,  80,  66,  47,   6,  62,
# #               173,   0])
# # #_ss.pearsonr(proj[0]-1.1*proj[1]-1.6*proj[2]+0.8*proj[3], lm["soc_skils"])
# # _ss.pearsonr(proj[0]+0.6*proj[2]+1.7*proj[3]-0.5*proj[4]-1.3*proj[5]-2.5*proj[6]+3*proj[7], lm["soc_skils"][inds])
# # _ss.pearsonr(proj[0]-1.1*proj[1]-1.6*proj[2]+0.8*proj[3]+3.3*proj[5]+0.4*proj[6]-3*proj[7], lm["soc_skils"])
# # #_ss.pearsonr(proj[0]-0.7*proj[1]+0.1*proj[2]-0.2*proj[3], lm["imag"])
# # #_ss.pearsonr(proj[0]-0.4*proj[1]-0.3*proj[4], lm["AQ28scrs"])
# # #_ss.pearsonr(-0.5*proj[4] + proj[5]+1.5*proj[6]-0.9*proj[3], lm["switch"])
