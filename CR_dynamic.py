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

label = 43

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

    _hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=None, expt="EEG1")    


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

    s1 = _N.sum(_hnd_dat[0:TO_GAME, 2])
    s2 = _N.sum(_hnd_dat[:, 2])
    print("%(1)d  %(2)d" % {"1" : s1, "2" : s2})
    return cond_probs, _N.sum(_hnd_dat[0:TO_GAME, 2])
        

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
score  = _N.empty(len(partIDs))

cpss = []
ip =0

for partID in partIDs:
    ip += 1
    cps, cumwins[ip-1] = pkg_all_data(partID)  #  first nperseg/2 points are constant
    CRs  = _em.CRs(partID)

    behv = _N.sum(_N.sum(_N.abs(_N.diff(cps, axis=2)), axis=1), axis=0)
    _dbehv = _N.diff(behv)
    dbehv = _N.convolve(_dbehv, gk, mode="same")
    maxs = _N.where((dbehv[0:-1] >= 0) & (dbehv[1:] < 0))[0] + 2 #  3 from label71
        
    #cps[0]   #  conditioned on Win
    #cps[1]   #  conditioned on Tie
    #cps[2]   #  conditioned on Lose


    perPart = []
    for i in range(1, maxs.shape[0]-2):
        perPart.append(_N.mean(cps[:, :, maxs[i]:maxs[i+1]], axis=2))

    A = _N.array(perPart)
    A_st = A[:, :, 1]
    A_sw = _N.sum(A[:, :, _N.array([0, 2])], axis=2)
    A_stsw = _N.empty((A.shape[0], 3, 2))
    A_stsw[:, :, 0] = A_st
    A_stsw[:, :, 1] = A_sw

    pc01_0, pv01_0 = _sstats.pearsonr(cps[0, 0], cps[1, 0])    
    pc01_1, pv01_1 = _sstats.pearsonr(cps[0, 1], cps[1, 1])
    pc01_2, pv01_2 = _sstats.pearsonr(cps[0, 0], cps[1, 0])        

    pc02_0, pv02_0 = _sstats.pearsonr(cps[0, 0], cps[2, 0])    
    pc02_1, pv02_1 = _sstats.pearsonr(cps[0, 1], cps[2, 1])
    pc02_2, pv02_2 = _sstats.pearsonr(cps[0, 2], cps[2, 2])    

    pc12_0, pv12_0 = _sstats.pearsonr(cps[1, 0], cps[2, 0])
    pc12_1, pv12_1 = _sstats.pearsonr(cps[1, 1], cps[2, 1])
    pc12_2, pv12_2 = _sstats.pearsonr(cps[1, 2], cps[2, 2])        

    pc_sum[ip-1] = pc01_0+pc01_1+pc01_2+pc02_0+pc02_1+pc02_2+pc12_0+pc12_1+pc12_1
    
    # pc01, pv01 = _sstats.pearsonr(A_stsw[:, 0, 0], A_stsw[:, 1, 0])
    # pc02, pv02 = _sstats.pearsonr(A_stsw[:, 0, 0], A_stsw[:, 2, 0])
    # pc12, pv12 = _sstats.pearsonr(A_stsw[:, 1, 0], A_stsw[:, 2, 0])
    # pc_sum[ip-1] = pc01+pc02+pc12    
    cpss.append(cps)

    #fig = _plt.figure(figsize=(12, 12))
    bgclrs = ["#FFCCCC", "#CCFFCC", "#CCCCFF"]
    for wtl in range(3):
        bgclr = bgclrs[wtl]
        for act in range(3):
            stds[ip-1, wtl, act] = _N.std(cps[wtl, act, 20:TO_GAME])
            stds12[ip-1, wtl, act, 0] = _N.std(cps[wtl, act, 10:(TO_GAME-10)//2])
            stds12[ip-1, wtl, act, 1] = _N.std(cps[wtl, act, (TO_GAME-10)//2:])                        
            means[ip-1, wtl, act] = _N.mean(cps[wtl, act, 20:TO_GAME])
            cvs[ip-1, wtl, act] = stds[ip-1, wtl, act] / means[ip-1, wtl, act]
            skews[ip-1, wtl, act] = _sstats.skew(cps[wtl, act, 10:TO_GAME])
            
            if wtl == 0:
                condition = _N.where((CRs >= 0) & (CRs < 3))[0]
                cond_act  = _N.where(CRs[condition] == act)[0]
            elif wtl == 1:            
                condition = _N.where((CRs >= 3) & (CRs < 6))[0]
                cond_act  = _N.where(CRs[condition] == 3+act)[0]
            elif wtl == 2:
                condition = _N.where((CRs >= 6) & (CRs < 9))[0]
                cond_act  = _N.where(CRs[condition] == 6+act)[0]

            # ax = fig.add_subplot(9, 1, wtl*3 + act+1)
            # ax.set_facecolor(bgclr)
            # s01s = _N.zeros(CRs.shape[0])
            # s01s[_N.where(CRs == wtl*3+act)[0]] = 1   #  STAY | LOS
            # _plt.plot(s01s)
            # _plt.plot(cps[wtl, act])
                
            #act_iei  = #_N.diff(condition[cond_act])
            act_iei = _N.diff(cond_act)            
            IEI_wtl_act[ip-1, wtl, act] = _N.std(act_iei) / _N.mean(act_iei)



# CRs == 7     #  LOSE|STAY
# cps[2, 1]

fig = _plt.figure(figsize=(11.2, 3.))
tksz  = 15
lblsz = 17
sw = _N.array([0, 1, 2])

wtl_stds = _N.mean(cvs, axis=2)
wtl_stds = _N.mean(stds, axis=2)

#wtl_stds = _N.mean(means, axis=2) 
#wtl_stds = stds[:, :, 1] / means[:, :, 1]
IEI_wtl = _N.mean(IEI_wtl_act, axis=2)

swtls = ["WIN", "TIE", "LOSE"]

for wtl in range(3):
    fig.add_subplot(1, 3, wtl+1)
    #_plt.scatter(_N.mean(stds[:, wtl, :], axis=1), cumwins, marker=".", color="black")
    
    _plt.scatter(wtl_stds[:, wtl], cumwins, marker=".", color="black", s=90)
    #_plt.scatter(IEI_wtl[:, wtl], cumwins, marker=".", color="black")
    #_plt.scatter(wtl_stds[smallest_win, wtl], cumwins[smallest_win], marker="x", s=20, color="blue")
    #_plt.scatter(wtl_stds[largest_win, wtl], cumwins[largest_win], marker="x", s=20, color="red")    
    _plt.axhline(y=0, ls=":", color="red")
    _plt.xlim(0.235, 0.355)
    pc, pv = _sstats.pearsonr(wtl_stds[:, wtl], cumwins)
    #pc, pv = _sstats.pearsonr(IEI_wtl[:, wtl], cumwins)
    #_plt.xlabel(r"std[mean $p_k$($\cdot$ | %s)]" % swtls[wtl], fontsize=lblsz)
    _plt.xlabel(r"modulation strength", fontsize=lblsz)

    _plt.ylabel("Net wins", fontsize=lblsz)
    _plt.xticks([0.25, 0.3, 0.35], fontsize=tksz)
    _plt.yticks(fontsize=tksz)    
    print("wtl %(w)d %(a)d   %(pc).3f %(pv).3f" % {"w" : wtl, "a" : act, "pc" : pc, "pv" : pv})

    x =  0.24 if (wtl < 2) else 0.295
    _plt.text(x, -53, "CC=%(pc).2f\npv<%(pv).1e" % {"w" : wtl, "a" : act, "pc" : pc, "pv" : pv}, fontsize=(tksz-2))

fig.subplots_adjust(bottom=0.2, wspace=0.42)

_plt.savefig("CR_dynamics")

########
srt_by_lose_inds = pc_sum.argsort()[::-1]#wtl_stds[:, 2].argsort()
#srt_by_lose_inds = cumwins.argsort()
#srt_by_lose_inds = avg_skews[:, 2].argsort()   
#srt_by_lose_inds = _N.sum(wtl_stds[:, 1:], axis=1).argsort()

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
for si in srt_by_lose_inds:
    ims.append(_plt.imread("../AIiRPS_Results/%(id)s/v12/%(id)s_512_128_skull_mncoh_pattern_0_%(1)d_%(2)d_v12.png" % {"id" : rpsms.rpsm_partID_as_key[partIDs[si]], "1" : f1, "2" : f2}))

i = 0

for ax, im in zip(grid, ims):
    i+=1
    # Iterating over the grid returns the Axes.
    ax.imshow(im)
    ax.set_title("subj %d" % i, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
_plt.savefig("skulls_%(1)d_%(2)d" % {"1" : f1, "2" : f2})
#plt.show()    

fig = _plt.figure(figsize=(14, 1.5))
_plt.plot(wtl_stds[srt_by_lose_inds, 2], marker=".", ms=14, color="black")
_plt.xticks(_N.arange(15), _N.arange(1, 16), fontsize=14)
_plt.yticks(fontsize=14)
_plt.xlabel("subject #", fontsize=16)
_plt.ylabel("modulation\nstrength", fontsize=16)

_plt.xlim(-0.2, 14.2)
_plt.ylim(0.24, 0.35)
fig.subplots_adjust(bottom=0.35)
_plt.savefig("modulation")


