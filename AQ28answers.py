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
from matplotlib import cm

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

def depickle(s):
    import pickle
    with open(s, "rb") as f:
        lm = pickle.load(f)
    return lm

lm = depickle("predictAQ28dat/AQ28_vs_RPS_1.dmp")

AQ28scores = ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]
for scrs in AQ28scores[1:]:
    exec("ans_%(f)s = lm[\"ans_%(f)s\"]" % {"f" : scrs})

soc_skils_use = _N.array([0, 1, 2, 3, 4, 5, 6])
imag_use      = _N.array([0, 1, 2, 3, 4, 5, 6, 7])
rout_use      = _N.array([0, 1, 2, 3])
switch_use    = _N.array([0, 1, 2, 3])
fact_pat_use  = _N.array([0, 1, 2, 3, 4])


fig = _plt.figure(figsize=(9, 5))
si = 0
for starget in AQ28scores[1:]:
    exec("inds = %s_use" % starget)
    exec("target = ans_%s" % starget)    
    si += 1
    fig.add_subplot(2, 3, si)
    mat_CCs = _N.zeros((len(inds), len(inds)))
    for i in range(len(inds)):
        for j in range(i+1, len(inds)):
            mat_CCs[i, j], pv = _ss.pearsonr(target[:, i], target[:, j])
            mat_CCs[j, i] = mat_CCs[i, j]

    _plt.imshow(mat_CCs, cmap=cm.gray, vmin=0, vmax=1)
    _plt.colorbar()
    _plt.title(starget)

