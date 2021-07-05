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

from AIiRPS.utils.dir_util import getResultFN

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

label = 71

def pkg_all_data(partID):
    """
    d_reprtd_start:  delay RPSM game start relative to DSi - as calculated by reported JS start time and DSi start time.
    dt_sys_clks:  how much system times are off.

    d_reprtd_start = RPSM(start) - DSi(start)
    RPSM  say 12:00:00, DSi says 11:59:30      +30 
    if same_time_RPSM is 12:00:00 and same_time_EEG is 11:59:35,
    then d_sys_clks = 25
    11:59:35 (RPSM) and 11:59:30  (DSi)
als
    d_start        = d_reprtd_start - d_sys_clks

    if d_start > 0, RPSM started AFTER EEG
    if d_start < 0, RPSM started BEFORE EEG
    """
    
    print("partID   %s" % partID)
    dat_pkg         = {}
    dsi_fn          = rpsms.rpsm_partID_as_key[partID]

    #same_time_RPSM  = params4partID[partID][0]   #  from calibration.html 
    #same_time_EEG   = params4partID[partID][1]   #  from calibration.html 

    rpsm_fn         = "rpsm_%s.dat" % partID

    _hnd_dat, start_time, end_time            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True)

    reprtd_start_RPSM_str  = start_time#rpsm_key[8:15]     #  from key name
    #if dsi_fn is not None:
    #     reprtd_start_EEG_str   = dsi_fn[10:18]

    #day = partID[0:8]
    #time= partID[9:]

    #eeg_dir       = "~/Sites/taisen/DATA/EEG1/%(day)s/%(dt)s/" % {"day" : day, "dt" : partID}

    trials          = _hnd_dat.shape[0]
    print("!!!!!  trials %d" % trials)

    #  the weight each one should have is just proportional to the # of events 
    #  of the condition (WTL) observed.  

    tr0     = 0 
    tr1     = trials

    hnd_dat         = _N.array(_hnd_dat[tr0:tr1])#0:trials-trim_trials])

    dat_pkg["lat"]               = []
    dat_pkg["behv"]              = []
    dat_pkg["behv_ts"]           = []

    #savgol_win                   = params4partID[partID][5]
    #%(dat)s,%(rel)s,%(cov)s%(ran)s
    sum_chosen_behv_sig          = None#_N.zeros((len(behv_list), Tm1))
    sigcov_behv_sig              = None
    sigcov_behv_fsig              = None
    behv_list                    = _N.array([0, 1, 2], _N.int)

    for bi in range(0, 1):
        if behv_list[bi] == _cnst._WTL:
            sig_cov = _N.array([_W, _T, _L])
        else:
            sig_cov = _N.array([_R, _P, _S])
        sig_cov   = behv_list[bi]
        behv_file = fns[bi]
        print(getResultFN("%(rpsm)s/%(lb)d/%(fl)s.dmp" % {"rpsm" : partID, "fl" : behv_file, "lb" : label}))
        dmp       = depickle(getResultFN("%(rpsm)s/%(lb)d/%(fl)s.dmp" % {"rpsm" : partID, "fl" : behv_file, "lb" : label}))

        cond_probs = dmp["cond_probs"]
        cond_probs = cond_probs.reshape((3, 3, cond_probs.shape[1]))
    return cond_probs, _N.sum(_hnd_dat[:, 2])
        

#fig = _plt.figure(figsize=(8, 8))
ip =0
#  THESE KEYS ARE RPS GAME DATA NAMES
#for rpsm_key in ["20Jan08-1703-13"]:
#for rpsm_key in ["20Jan09-1504-32"]:

c1 = 1
c2 = 2
pcs = []
pc1s = []
pc2s = []
col = 2
y1s = []
y2s = []

#partIDs=["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09", "20200601_0748-03", "20210529_1923-44", "20210529_1419-14", "20210606_1237-17", "20210609_1517-23", "20201122_1108-25", "20201121_1959-30", "20201121_2131-38", "20200410_2203-19", "20200410_2248-43", "20200415_2034-12", "20200418_2148-58"]
partIDs = ["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1747-07", "20210609_1517-23", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1331-06", "20200812_1252-50", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09", "20200601_0748-03", "20210529_1923-44", "20210529_1419-14", "20210606_1237-17", "20201122_1108-25", "20201121_1959-30", "20201121_2131-38"]

stds = _N.zeros((len(partIDs), 3))
cvs  = _N.zeros((len(partIDs), 3))
means = _N.zeros((len(partIDs), 3))
cumwins = _N.zeros(len(partIDs))

cpss = []
for partID in partIDs:
    ip += 1
    cps, cumwins[ip-1] = pkg_all_data(partID)  #  first nperseg/2 points are constant

    #cps[0]   #  conditioned on Win
    #cps[1]   #  conditioned on Tie
    #cps[2]   #  conditioned on Lose

    cpss.append(cps[2, 1])

    y1 = cps[c1, col]
    y2 = cps[c2, col]
    y1s.extend(y1[10:-20])
    y2s.extend(y2[10:-20])              
    by1 = _N.zeros(cps.shape[2])
    by2 = _N.zeros(cps.shape[2])
    by1[_N.where(y1 > 0.5)[0]] = 1
    by2[_N.where(y2 > 0.5)[0]] = 1    
    
    L = y1.shape[0]
    pc1, pv1 = _sstats.pearsonr(y1[0:L//2], y2[0:L//2])
    pc2, pv2 = _sstats.pearsonr(y1[L//2:], y2[L//2:])
    pc, pv = _sstats.pearsonr(y1, y2)        
    pcs.append(pc)
    pc1s.append(pc1)
    pc2s.append(pc2)

    cvs[ip-1, 0] = _N.std(cps[0, 1]) / _N.mean(cps[0, 1])
    cvs[ip-1, 1] = _N.std(cps[1, 1]) / _N.mean(cps[1, 1])
    cvs[ip-1, 2] = _N.std(cps[2, 1])     / _N.mean(cps[2, 1])
    stds[ip-1, 0] = _N.std(cps[0, 1])
    stds[ip-1, 1] = _N.std(cps[1, 1])
    stds[ip-1, 2] = _N.std(cps[2, 1])
    
    means[ip-1, 0] = _N.mean(cps[0, 1])
    means[ip-1, 1] = _N.mean(cps[1, 1])
    means[ip-1, 2] = _N.mean(cps[2, 1])

lblsz=16
tksz=14
fig = _plt.figure(figsize=(5, 4))
_plt.scatter(cvs[:, 2], cumwins, marker=".", color="black")
#_plt.
pc, pv = _sstats.pearsonr(cvs[:, 2], cumwins)
_plt.text(0.82, -60, "CC=%(pc).2f   p-val=%(pv).1e" % {"pc" : pc, "pv" : pv}, fontsize=tksz)
_plt.xlabel("CV p(STAY|LOSE)", fontsize=lblsz)
_plt.ylabel("Net Win", fontsize=lblsz)

_plt.xticks(fontsize=tksz)
_plt.yticks(fontsize=tksz)
fig.subplots_adjust(bottom=0.15, left=0.2)
_plt.savefig("CVofCR_vs_netwin")


# scov = ["W", "T", "L"]
# sact = ["DN", "ST", "UP"]
# L = len(y1s)
# ay1s = _N.array(y1s)
# ay2s = _N.array(y2s)


# majo = _N.where(cvs[:, 0] > cvs[:, 2])[0]
# mino = _N.where(cvs[:, 0] < cvs[:, 2])[0]
# fig = _plt.figure(figsize=(6, 5))
# for cat in range(2):
#     ax = _plt.subplot2grid((7, 2), (0, cat), rowspan=1)
#     if cat == 0:
#         _plt.scatter(cumwins[majo], _N.zeros(len(majo)))
#         _plt.title("BIGGER POST WIN FLUC")
#     if cat == 1:
#         _plt.scatter(cumwins[mino], _N.zeros(len(mino)))
#         _plt.title("BIGGER POST LOS FLUC")        
#     _plt.xlim(-30, 30)
#     _plt.xlabel("NET WINS")
#     ax = _plt.subplot2grid((7, 2), (2, cat), rowspan=5)
#     #ax.set_facecolor("#CCCCCC")
#     m_cvs = _N.mean(cvs, axis=0)
#     m_cvs /= m_cvs[1]
#     for i in range(cvs.shape[0]):
#         n_cvs = cvs[i] / cvs[i, 1]
#         if n_cvs[0] > n_cvs[2]:
#             clr="black"
#             lw=2
#             thiscat = 0
#             expl = "fluc(p(ST|W))>fluc(p(ST|L))"
#             if (n_cvs[2] > n_cvs[1]) and (n_cvs[0] > n_cvs[1]):
#                 clr="red"
#             elif (n_cvs[2] < n_cvs[1]) and (n_cvs[0] < n_cvs[1]):
#                 clr="blue"
#         if n_cvs[0] < n_cvs[2]:
#             clr="black"            
#             lw=2
#             thiscat = 1
#             expl = "fluc(p(ST|W))<fluc(p(ST|L))"            

#             if (n_cvs[2] > n_cvs[1]) and (n_cvs[0] > n_cvs[1]):
#                 clr="red"
#             elif (n_cvs[2] < n_cvs[1]) and (n_cvs[0] < n_cvs[1]):
#                 clr="blue"
#         if thiscat == cat:
#             _plt.plot([0, 1, 2], n_cvs, color=clr)
#             _plt.xticks([0, 1, 2], ["W", "T", "L"])
#             _plt.yticks([0, 0.5, 1, 1.5, 2])
#             _plt.title(expl)
#     _plt.ylim(0, 2)



# fig.subplots_adjust(wspace=0.3, hspace=0.55, top=0.9, bottom=0.1)
# _plt.suptitle("magnitude of STAY probability fluctuation, coeff var")
# _plt.savefig("fluc_in_CR_for_ST")
