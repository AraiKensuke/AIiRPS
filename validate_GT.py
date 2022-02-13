import AIiRPS.models.empirical_ken as empirical
import AIiRPS.utils.misc as _Am
import numpy as _N
import matplotlib.pyplot as _plt
import GCoh.eeg_util as _eu
import AIiRPS.utils.read_taisen as _rt
from AIiRPS.utils.dir_util import getResultFN
import AIiRPS.models.CRutils as _crut
import os
import pickle
import AIiRPS.constants as _AIconst
import scipy.stats as _ss
import glob
import AIiRPS.simulation.simulate_prcptrn as sim_prc

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

def secs_as_string(sec0, sec1):
    l = []
    for s in range(sec0, sec1):
        if s < 10:
            l.append("0%d" % s)
        else:
            l.append("%d"  % s)
    return l
            
win=3

maxT = 300
expt = "SIMHUM1"
trig_infrd = []
trig_GT_rc = []
#for date in ["20110101_0000-00", "20110101_0000-05", "20110101_0000-10", "20110101_0000-20", "20110101_0000-25", "20110101_0000-30", "20110101_0000-35", "20110101_0000-40", "20110101_0000-45", "20110101_0000-50", "20110101_0000-55"]:
for sec in secs_as_string(0, 60):#, "01", "02", "03", "04", "05", "06", "07", "08", "09"]:
#for date in ["20110101_0000-00"]:
    date = "20110101_0000-%s" % sec
    td, start_time, end_time, UA, cnstr, inp_meth, ini_percep, fin_percep = _rt.return_hnd_dat(date, has_useragent=True, has_start_and_end_times=True, has_constructor=True, expt=expt, visit=1)

    ngs, ngsRPS, ngsDSURPS, ngsSTSW, all_tds, TGames  = empirical.empirical_NGS_concat_conds(date, win=win, SHUF=0, flip_human_AI=False, expt=expt, visit=1)

    datdmp = "/Users/arai/Sites/taisen/DATA/%(e)s/20110101/%(d)s/1/block1_AI.dmp" % {"d" : date, "e" : expt}
    lm = depickle(datdmp)


    gk_w = 2
    gk = _Am.gauKer(gk_w)
    gk /= _N.sum(gk)

    SHUFFLES = 0
    if ngs is not None:
        fNGS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]))
        fNGSRPS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]))
        fNGSDSURPS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]))                        
        fNGSSTSW = _N.empty((SHUFFLES+1, ngsSTSW.shape[1], ngsSTSW.shape[2]))            
        t_ms = _N.mean(_N.diff(all_tds[0, :, 3]))
        for sh in range(SHUFFLES+1):
            for i in range(9):
                if gk_w > 0:
                    fNGS[sh, i] = _N.convolve(ngs[sh, i], gk, mode="same")
                    fNGSRPS[sh, i] = _N.convolve(ngsRPS[sh, i], gk, mode="same")
                    fNGSDSURPS[sh, i] = _N.convolve(ngsDSURPS[sh, i], gk, mode="same")                                                
                else:
                    fNGS[sh, i] = ngs[sh, i]
                    fNGSRPS[sh, i] = ngsRPS[sh, i]
                    fNGSDSURPS[sh, i] = ngsDSURPS[sh, i]

    dbehv, behv  = _crut.get_dbehv_combined([fNGS, fNGSRPS, fNGSDSURPS], gk, equalize=False)
    #dbehv, behv  = _crut.get_dbehv_combined([fNGS], gk, equalize=True)

    maxima = _N.where((behv[0:-3] < behv[1:-2]) & (behv[1:-2] > behv[2:-1]))[0]
    minima = _N.where((behv[0:-3] > behv[1:-2]) & (behv[1:-2] < behv[2:-1]))[0]
    nMaxs = len(maxima)
    nMins = len(minima)

    max_thr = _N.sort(behv[minima + 1])[int(0.9*nMins)]  #  we don't want maxes to be below any mins
    #max_thr = _N.sort(behv[maxima + 1])[int(0.1*nMaxs)]  #  we don't want maxes to be below any mins
    maxs = maxima[_N.where(behv[maxima+1] > max_thr)[0]] + win//2+1


    gt_probs = lm["Ts_timeseries"]

    gt_chg_pts = _N.sum(_N.sum(_N.abs(_N.diff(gt_probs, axis=0)), axis=1), axis=1)
    t_ones     = _N.where(gt_chg_pts != 0)[0]
    gt_chg_pts[t_ones] = 1
    #_plt.plot(gt_chg_pts)
    #_plt.plot(_N.arange(len(dbehv))+2, dbehv)

    #t_pcs = _N.zeros((len(t_ones)-12, 30))

    for i in _N.where((t_ones > 15+2) & (t_ones < 300-(15+2)))[0]:
        #print(dbehv[t_ones[i]-2-15:t_ones[i]-2+15].shape)
        trig_GT_rc.append(dbehv[t_ones[i]-2-15:t_ones[i]-2+15])


    #t_pcsB = _N.zeros((len(maxs)-12, 30))
    #for i in range(6, len(maxs)-6):
    for i in _N.where((maxs > 15) & (maxs < 300-15))[0]:
        #t_pcsB[i-5] = gt_chg_pts[maxs[i]-15:maxs[i]+15]
        trig_infrd.append(gt_chg_pts[maxs[i]-15:maxs[i]+15])

    #t_pcsBs.append(t_pcsB)


trig_GT_rc_arr = _N.array(trig_GT_rc)
trig_infrd_arr = _N.array(trig_infrd)
