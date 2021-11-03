import sklearn.linear_model as _skl
import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

N  = 120
Xs   = _N.random.randn(N, 2)
Xs[:, 0] *= 0.7
Xs[:, 0] -= 1
Xs[:, 1] += 0.4

a    = _N.array([0.5, -0.4])      #  coupling coefficients of Xs and y

e_explanatory = _N.zeros((N, 2))
e_explanatory[:, 0] = 0.03*_N.std(Xs[:, 0])*_N.random.randn(N)
e_explanatory[:, 1] = 0.05*_N.std(Xs[:, 1])*_N.random.randn(N)

Xs_e = Xs + e_explanatory         #  let's assume there's some noise on our observation of Xs.  This is not the value of predictor used to generate y, but its the Xs that we measure

e  = 1.1  #  noise 1
nz = e*_N.random.randn(N)
y      = _N.dot(Xs, a) + nz       #  GENERATE y given Xs

####   STANDARDIZE
zXs   = (Xs_e / _N.std(Xs_e, axis=0))#
zXs  -= _N.mean(zXs, axis=0)
zy   = (y / _N.std(y))#
zy   -= _N.mean(zy)

#  observed is Xs

print("original predictors, original target")
pc0, pv0 = _ss.pearsonr(Xs[:, 0], y)
pc1, pv1 = _ss.pearsonr(Xs[:, 1], y)
pcS, pvS = _ss.pearsonr(Xs[:, 0] - Xs[:, 1], y)
print("%(pc0).3f, %(pv0).2e     %(pc1).3f,%(pv1).2e    sum %(pcS).3f,%(pvS).2e" % {"pc0" : pc0, "pv0" : pv0, "pc1" : pc1, "pv1" : pv1, "pcS" : pcS, "pvS" : pvS})

print("whitened predictors, original target")
pc0, pv0 = _ss.pearsonr(zXs[:, 0], zy)
pc1, pv1 = _ss.pearsonr(zXs[:, 1], zy)
pcS, pvS = _ss.pearsonr(zXs[:, 0] - zXs[:, 1], zy)
print("%(pc0).3f, %(pv0).2e     %(pc1).3f,%(pv1).2e    sum %(pcS).3f,%(pvS).2e" % {"pc0" : pc0, "pv0" : pv0, "pc1" : pc1, "pv1" : pv1, "pcS" : pcS, "pvS" : pvS})

clf = _skl.LinearRegression()
nrep = 100
rkf = RepeatedKFold(n_splits=3, n_repeats=nrep)#, random_state=0)
scores1 = cross_val_score(clf, zXs, zy, cv=rkf)

zXs_sum = _N.empty((N, 1))
zXs_sum[:, 0] = zXs[:, 1] - zXs[:, 0]
scores2 = cross_val_score(clf, zXs_sum, zy, cv=rkf)
