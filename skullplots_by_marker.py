import numpy as _N
import AIiRPS.utils.read_taisen as _rt
import matplotlib.pyplot as _plt
import scipy.signal as _ss
import scipy.stats as _sstats
from scipy.signal import savgol_filter
import scipy.io as _scio
import pickle
import os
import GCoh.preprocess_ver as _ppv
import rpsms
import glob
import AIiRPS.constants as _cnst
from AIiRPS.models import empirical as _em
from filter import gauKer
from AIiRPS.utils.dir_util import getResultFN
from mpl_toolkits.axes_grid1 import ImageGrid
import AIiRPS.models.CRutils as _emp

_ME_WTL = 0
_ME_RPS = 1
_AI_WTL = 2   #  DON'T USE   (re-mapped _ME_WTL)
_AI_RPS = 3   #  DON'T USE

_W      = 0
_T      = 1
_L      = 2
_R      = 0
_P      = 1
_S      = 2

gk = gauKer(1)
gk /= _N.sum(gk)

TO_GAME = 270
fns     = ["WTL", "HUMRPS", "AIRPS"]

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

#   This file puts the rpsm file together with the latent state output into one 

####  RPS
#   hnd_dat
#   latent state btp_wtl, chg_wtl, btp_rps, chg_rps  (and time stamp for each only)
#   derivative of latent states  (and time stamp for each one)
#   
####  EEG
#   spectrogram   
#   GCoh          

win     = 4
smth    = 1 
label          = win*10+smth
        

#partIDs=["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09", "20200601_0748-03", "20210529_1923-44", "20210529_1419-14", "20210606_1237-17", "20210609_1517-23", "20201122_1108-25", "20201121_1959-30", "20201121_2131-38", "20200410_2203-19", "20200410_2248-43", "20200415_2034-12", "20200418_2148-58"]

#  some participants represented many times
#  8, 15, 16
#partIDs = ["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1747-07", "20210609_1517-23", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1331-06", "20200812_1252-50", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09", "20200601_0748-03", "20210529_1923-44", "20210529_1419-14", "20210606_1237-17", "20201122_1108-25", "20201121_1959-30", "20201121_2131-38"]   #  20210606_1237_17   has very large cvs, but removing this data we still see strong correlation  CC from 0.55 to 0.48, pv from 5e-3 to 2e-2
#partIDs = ["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1331-06", "20200812_1252-50", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]   #  20210606_1237_17   has very large cvs, but removing this data we still see strong correlation  CC from 0.55 to 0.48, pv from 5e-3 to 2e-2
partIDs = ["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1331-06", "20200812_1252-50", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]   #  20210606_1237_17   has very large cvs, but removing this data we still see strong correlation  CC from 0.55 to 0.48, pv from 5e-3 to 2e-2
#  "20210526_1416-25",   --->>  had bad channels
#partIDs.pop(16)
#partIDs.pop(15)
#partIDs.pop(8)
# partIDs=["20210607_1434-03",
#          "20210607_1434-20",
#          "20210607_1434-52",
#          "20210607_1435-13",
#          "20210607_1435-36",
#          "20210607_1435-56",
#          "20210607_1436-20",
#          "20210607_1436-36",
#          "20210607_1502-42",
#          "20210607_1503-21",
#          "20210607_1503-36",
#          "20210607_1503-52",
#          "20210607_1504-07",
#          "20210607_1504-49",
#          "20210607_1505-09",
#          "20210607_1505-39",
#          "20210607_1951-49"]

"""
###  only 1 game from each participant
partIDs = ["20210609_1230-28", "20210609_1248-16",
           "20210609_1321-35", "20210609_1747-07",
           "20210609_1517-23", #  WPI
           "20210526_1318-12", "20210526_1358-27",
           "20210526_1416-25", "20210526_1503-39",
           #  Neurable
           "20200109_1504-32",   # Ken
           "20200601_0748-03", "20210529_1923-44",
           "20210529_1419-14",   #  relatives
           "20210606_1237-17",   #  megumi
           "20201122_1108-25"]   #  sam
"""

#  The successful player tends to 
stds = _N.zeros((len(partIDs), 3, 3))    #  of loss only
stds12 = _N.zeros((len(partIDs), 3, 3, 2))    #  1st/2nd half
cvs  = _N.zeros((len(partIDs), 3, 3))
skews  = _N.zeros((len(partIDs), 3, 3))
IEI_wtl_act  = _N.zeros((len(partIDs), 3, 3))
means = _N.zeros((len(partIDs), 3, 3))
cumwins = _N.zeros(len(partIDs))

pc_sum = _N.empty(len(partIDs))
isis    = _N.empty(len(partIDs))
isis_sd    = _N.empty(len(partIDs))
isis_cv    = _N.empty(len(partIDs))
isis_corr    = _N.empty(len(partIDs))

corr_UD    = _N.empty((len(partIDs), 3))

score  = _N.empty(len(partIDs))
sum_sd = _N.empty((len(partIDs), 3, 3))
sum_cv = _N.empty((len(partIDs), 3, 3))
entropyDSU = _N.empty((len(partIDs), 3))
entropyWTL = _N.empty((len(partIDs), 3))
entropyD = _N.empty(len(partIDs))
entropyS = _N.empty(len(partIDs))
entropyU = _N.empty(len(partIDs))
entropyW = _N.empty(len(partIDs))
entropyT = _N.empty(len(partIDs))
entropyL = _N.empty(len(partIDs))

cpss = []
ip =0

pid = -1
for partID in partIDs:
    pid += 1

    _hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=None, expt="EEG1")
    dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/WTL.dmp" % {"rpsm" : partID, "lb" : label}))

    _prob_mvs = dmp["cond_probs"]
    prob_mvs  = _prob_mvs[:, 0:_hnd_dat.shape[0] - win]  #  is bigger than hand by win size
    prob_mvs = prob_mvs.reshape((3, 3, prob_mvs.shape[1]))
    dbehv = _emp.get_dbehv(prob_mvs, gk)
    maxs = _N.where((dbehv[0:-1] >= 0) & (dbehv[1:] < 0))[0] + (win//2) #  3 from label71

    PCS=10
    #  prob_mvs[:, 0] - for each time point, the DOWN probabilities following 3 different conditions
    #  prob_mvs[0]    - for each time point, the DOWN probabilities following 3 different conditions    

    entsDSU = _N.array([_emp.entropy3(prob_mvs[:, 0].T, PCS), _emp.entropy3(prob_mvs[:, 1].T, PCS), _emp.entropy3(prob_mvs[:, 2].T, PCS)])    
    entropyD[pid-1] = entsDSU[0]
    entropyS[pid-1] = entsDSU[1]
    entropyU[pid-1] = entsDSU[2]

    isi   = _N.diff(maxs)
    pc, pv = _sstats.pearsonr(isi[0:-1], isi[1:])
    isis_corr[pid-1] = pc
    isis_sd[pid-1] = _N.std(isi)
    isis[pid-1] = _N.mean(isi)        
    isis_cv[pid-1] = isis_sd[pid-1] / isis[pid-1]
    #all_maxs.append(isi)
    W = len(_N.where(_hnd_dat[:, 2] == 1)[0])
    L = len(_N.where(_hnd_dat[:, 2] == -1)[0])
    cumwins[pid-1] = W / (W+L)
    

        
# fig = _plt.figure(figsize=(11.2, 3.))
# tksz  = 15
# lblsz = 17
# sw = _N.array([0, 1, 2])

# wtl_stds = _N.mean(cvs, axis=2)
# wtl_stds = _N.mean(stds, axis=2)

# #wtl_stds = _N.mean(means, axis=2) 
# #wtl_stds = stds[:, :, 1] / means[:, :, 1]
# IEI_wtl = _N.mean(IEI_wtl_act, axis=2)

# swtls = ["WIN", "TIE", "LOSE"]

# for wtl in range(3):
#     fig.add_subplot(1, 3, wtl+1)
#     #_plt.scatter(_N.mean(stds[:, wtl, :], axis=1), cumwins, marker=".", color="black")
    
#     _plt.scatter(wtl_stds[:, wtl], cumwins, marker=".", color="black", s=90)
#     #_plt.scatter(IEI_wtl[:, wtl], cumwins, marker=".", color="black")
#     #_plt.scatter(wtl_stds[smallest_win, wtl], cumwins[smallest_win], marker="x", s=20, color="blue")
#     #_plt.scatter(wtl_stds[largest_win, wtl], cumwins[largest_win], marker="x", s=20, color="red")    
#     _plt.axhline(y=0, ls=":", color="red")
#     _plt.xlim(0.235, 0.355)
#     pc, pv = _sstats.pearsonr(wtl_stds[:, wtl], cumwins)
#     #pc, pv = _sstats.pearsonr(IEI_wtl[:, wtl], cumwins)
#     #_plt.xlabel(r"std[mean $p_k$($\cdot$ | %s)]" % swtls[wtl], fontsize=lblsz)
#     _plt.xlabel(r"modulation strength", fontsize=lblsz)

#     _plt.ylabel("Net wins", fontsize=lblsz)
#     _plt.xticks([0.25, 0.3, 0.35], fontsize=tksz)
#     _plt.yticks(fontsize=tksz)    
#     print("wtl %(w)d %(a)d   %(pc).3f %(pv).3f" % {"w" : wtl, "a" : act, "pc" : pc, "pv" : pv})

#     x =  0.24 if (wtl < 2) else 0.295
#     _plt.text(x, -53, "CC=%(pc).2f\npv<%(pv).1e" % {"w" : wtl, "a" : act, "pc" : pc, "pv" : pv}, fontsize=(tksz-2))

# fig.subplots_adjust(bottom=0.2, wspace=0.42)

# _plt.savefig("CR_dynamics")

########
biomarker        = cumwins#entropyD + entropyU# + entropyS
srt_by_biomarker = biomarker.argsort()


# i = -1
# for si in srt_by_lose_inds:
#     i += 1
#     print(partIDs[si])
#     os.system("cp ../AIiRPS_Results/%(id)s/v12/%(id)s_512_128_skull_mncoh_pattern_0_32_48_v12.png rnk_%(rnk)d%(id)s.png" % {"id" : rpsms.rpsm_partID_as_key[partIDs[si]], "rnk" : (i+1)})



grid_fig = _plt.figure(figsize=(15., 2.))
grid = ImageGrid(grid_fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 15),  # creates 2x2 grid of axes
                 axes_pad=0.05,  # pad between axes in inch.
                 )

ims = []
f1  = 32
f2  = 48
for si in srt_by_biomarker:
    ims.append(_plt.imread("../AIiRPS_Results/%(id)s/v12/%(id)s_512_128_skull_mncoh_pattern_0_%(1)d_%(2)d_v12.png" % {"id" : rpsms.rpsm_partID_as_key[partIDs[si]], "1" : f1, "2" : f2}))

i = 0

for ax, im in zip(grid, ims):
    i+=1
    # Iterating over the grid returns the Axes.
    ax.imshow(im)
    ax.set_title("%d" % i, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
_plt.savefig("skulls_%(1)d_%(2)d" % {"1" : f1, "2" : f2})
#plt.show()
fig = _plt.figure(figsize=(15, 2.5))
fig.add_subplot(1, 1, 1)
_plt.plot(biomarker[srt_by_biomarker], color="black", marker=".", ms=10)
_plt.xticks(_N.arange(len(partIDs)), _N.arange(len(partIDs))+1)
_plt.xlim(-0.5, len(partIDs)-0.5)
_plt.xlabel("subject #")
_plt.ylabel(r"win rate $\frac{W}{W+L}$")
fig.subplots_adjust(bottom=0.2)
_plt.savefig("skulls_biomarker_%(1)d_%(2)d" % {"1" : f1, "2" : f2})

# fig = _plt.figure(figsize=(14, 1.5))
# _plt.plot(wtl_stds[srt_by_lose_inds, 2], marker=".", ms=14, color="black")
# _plt.xticks(_N.arange(15), _N.arange(1, 16), fontsize=14)
# _plt.yticks(fontsize=14)
# _plt.xlabel("subject #", fontsize=16)
# _plt.ylabel("modulation\nstrength", fontsize=16)

# _plt.xlim(-0.2, 14.2)
# _plt.ylim(0.24, 0.35)
# fig.subplots_adjust(bottom=0.35)
# _plt.savefig("modulation")


