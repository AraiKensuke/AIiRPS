import numpy as _N
import AIiRPS.utils.read_taisen as _rt
import matplotlib.pyplot as _plt
import scipy.signal as _ss
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

#    "20Apr18-2148-58" : ["Apr182020_22_01_44_artfctrmvd.dat", "12:00:00", "12:00:00"]}
#dats = {"20Jan09-1504-32" : ["Jan092020_15_05_39", "gcoh_384_384", "12:00:00", "12:00:00", 0, None, 6, [_CHG_WTL1, _CHG_WTL2, _LTP_WTL3], 5, 4, 2],
#dats = {"20Jan09-1504-32" : ["Jan092020_15_05_39", "gcoh_256_256", "12:00:00", "12:00:00", 0, None, 6, [_CHG_WTL2], 5, 4, 2],
#dats = {"20Jan09-1504-32" : ["Jan092020_15_05_39", "gcoh_256_256", "12:00:00", "12:00:00", 0, None, 6, [_CHG_WTL1, _CHG_WTL2, _CHG_WTL3, _LTP_WTL1, _LTP_WTL2, _LTP_WTL3, _BTP_WTL1, _BTP_WTL2, _BTP_WTL3], 5, 4, 3],
#dats = {"20Jan09-1504-32" : ["Jan092020_15_05_39", "gcoh_256_256", "12:00:00", "12:00:00", 0, None, 6, 6, 6, 6, 6, 6, 5, 4, 2],
#dats = {"20Jan09-1504-32" : ["Jan092020_15_05_39", "gcoh_256_64", "12:00:00", "12:00:00", 0, None, 6, 6, 6, 6, 6, 6, 5, 4, 3],
"""
time RPS  (same time)
time EEG  (same time)
label
armv_ver
GCoh_ver
savgol_win
"""

gv = 2
av  = 1
sw = None
lab = 71
#  participantID as rpsm_key
params4partID = {"20200109_1504-32": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20200108_1642-20": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20200812_1252-50": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20200812_1331-06": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20200818_1546-13": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20200818_1603-42": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20200818_1624-01": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20200818_1644-09": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210609_1747-07": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210609_1230-28": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210609_1248-16": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210609_1321-35": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210526_1318-12": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210526_1358-27": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210526_1416-25": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]],
                 "20210526_1503-39": ["12:00:00", "12:00:01", lab, av, gv, sw, [_cnst._WTL, _cnst._HUMRPS, _cnst._AIRPS]]}
#        "20210609_1710-28" : ["12:00:00", "12:00:01", 14, 1, 3, 5]

#dats = {"rpsm_20Jan09-1500-00.dat" : ["Jan092020_15_00_35_artfctrmvd.dat", "12:00:00", "12:00:00"]}
#        "rpsm_20Apr10-2307-53.dat" : "Apr102020_23_08_48_artfctrmvd.dat",
#        "rpsm_20Apr15-2034-12.dat" : "Apr102020_20_34_05_artfctrmvd.dat"}

#  raw eeg is used to calculate GCoh - 

def overlapping_window_center_times(N, binsz, shft, fs_eeg):
    """
    """
    dt    = 1./fs_eeg
    t0    = 0
    t1    = binsz

    bins    = 0

    mid_win_t = []
    while t1 <= N:
        #print("[%(0)d %(1)d]" % {"0" : t0, "1" : t1})
        mid_win_t.append(0.5*(t1+t0) * dt)
        t0 += shft
        t1 += shft
        bins += 1

    return _N.array(mid_win_t)

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

    same_time_RPSM  = params4partID[partID][0]   #  from calibration.html 
    same_time_EEG   = params4partID[partID][1]   #  from calibration.html 

    rpsm_fn         = "rpsm_%s.dat" % partID

    _hnd_dat, start_time, end_time            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True)

    reprtd_start_RPSM_str  = start_time#rpsm_key[8:15]     #  from key name
    if dsi_fn is not None:
         reprtd_start_EEG_str   = dsi_fn[10:18]

    day = partID[0:8]
    time= partID[9:]
    print(day)
    print(time)
    eeg_dir       = "~/Sites/taisen/DATA/EEG1/%(day)s/%(dt)s/" % {"day" : day, "dt" : partID}

    trials          = _hnd_dat.shape[0]
    print("!!!!!  trials %d" % trials)

    #  the weight each one should have is just proportional to the # of events 
    #  of the condition (WTL) observed.  

    tr0     = 0 
    tr1     = trials

    label   = params4partID[partID][2]
    #covWTLRPS     = dats[rpsm_key][6] if dats[rpsm_key][6] is not None else [_N.array([_W, _T, _L]), _N.array([_R, _P, _S])]
    #covWTLRPS     = dats[rpsm_key][6] if dats[rpsm_key][6] is not None else [_N.array([_W, _T, _L])]

    armv_ver = params4partID[partID][3]
    gcoh_ver = params4partID[partID][4]

    win_spec, slideby_spec      = _ppv.get_win_slideby(gcoh_ver)
    win_gcoh        = win_spec
    slideby_gcoh    = slideby_spec

    gcoh_fn  = "gcoh_%(ws)d_%(ss)d" % {"ws" : win_spec, "ss" : slideby_spec}

    hnd_dat         = _N.array(_hnd_dat[tr0:tr1])#0:trials-trim_trials])

    """
    ##  calculate weight each conditional 
    stay_win, strg_win, wekr_win, stay_tie, wekr_tie, strg_tie, stay_los, wekr_los, strg_los, win_cond, tie_cond, los_cond = _rt.get_ME_WTL(hnd_dat, tr0, tr1)
    #  p(stay | W)     

    nWins = len(win_cond)
    nTies = len(tie_cond)
    nLoss = len(los_cond)

    cond_events = [[stay_win, wekr_win, strg_win],
                   [stay_tie, wekr_tie, strg_tie],
                   [stay_los, wekr_los, strg_los]]
    marg_cond_events = [win_cond, tie_cond, los_cond]

    cond_wgts   = _N.array([[[win_cond.shape[0]]], [[tie_cond.shape[0]]], [[los_cond.shape[0]]]])

    cond_wgts   = cond_wgts / _N.sum(cond_wgts)
    """

    dat_pkg["lat"]               = []
    dat_pkg["behv"]              = []
    dat_pkg["behv_ts"]           = []

    savgol_win                   = params4partID[partID][5]
    #%(dat)s,%(rel)s,%(cov)s%(ran)s
    sum_chosen_behv_sig          = None#_N.zeros((len(behv_list), Tm1))
    sigcov_behv_sig              = None
    sigcov_behv_fsig              = None
    behv_list                    = _N.array([0, 1, 2], _N.int)
    fig = _plt.figure(figsize=(10, 8))
    for bi in range(3):
         #covWTLRPS = _N.array([_W, _T, _L])# if behv_list[bi] == _ME_WTL else _N.array([_R, _P, _S])
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
        Tm1     = cond_probs.shape[2]    # because AR1 filter is shorter by 1
        #  dNGS then should at most be Tm1 - 1
        print("trials %(tr)d   Tm1 %(Tm1)d" % {"tr" : trials, "Tm1" : Tm1})
        if sigcov_behv_sig is None:
            #  Tm1 - 1 (because derivative)
            sigcov_behv_sig  = _N.zeros((3, Tm1-1))
        sigcov_behv_fsig = _N.zeros((3, Tm1-1))
        
        prob_mvs  = cond_probs
        #prob_mvs[1] *= 0.001   #  make TIE condition contribution small
        prob_fmvs = _N.zeros((3, 3, Tm1))


        for iw in range(3):
            for ix in range(3):
                # if savgol_win is not None:
                #     prob_fmvs[iw, ix] = savgol_filter(prob_mvs[iw, ix], savgol_win, 3) # window size 51, polynomial ord
#                else:
                prob_fmvs[iw, ix] = prob_mvs[iw, ix]

        these_covs = _N.array([0, 1, 2])
         
        #  sum over WTL condition first
        sigcov_behv_sig[bi] = _N.sum(_N.sum(_N.abs(_N.diff(prob_mvs[these_covs], axis=2)), axis=1), axis=0)
        sigcov_behv_fsig[bi] = _N.sum(_N.sum(_N.abs(_N.diff(prob_fmvs[these_covs], axis=2)), axis=1), axis=0)
        n = 0
        
        fig.add_subplot(3, 2, bi*2+1)
        bhv = sigcov_behv_sig[bi]
        _plt.acorr(bhv - _N.mean(bhv), maxlags=30)
        _plt.grid()
        fig.add_subplot(3, 2, bi*2+2)
        fbhv = sigcov_behv_fsig[bi]
        _plt.acorr(fbhv - _N.mean(fbhv),  maxlags=30)
        _plt.grid()

        print("..................................  %d" % bi)
        print(sigcov_behv_sig[bi])
        print(sigcov_behv_fsig[bi])
    # bhv1 = sigcov_behv_sig[0]
    # bhv2 = sigcov_behv_sig[1]
    # fig.add_subplot(3, 2, 5)
    # _plt.xcorr(bhv1 - _N.mean(bhv1), bhv2 - _N.mean(bhv2), maxlags=30)
    # bhv1 = sigcov_behv_fsig[0]
    # bhv2 = sigcov_behv_fsig[1]
    # fig.add_subplot(3, 2, 6)
    # _plt.xcorr(bhv1 - _N.mean(bhv1), bhv2 - _N.mean(bhv2), maxlags=30)

    print(sigcov_behv_sig)    
    dat_pkg["behv"]  = sigcov_behv_sig
    print(sigcov_behv_fsig)
    dat_pkg["fbehv"]  = sigcov_behv_fsig
    dat_pkg["savgol_win"] = savgol_win
    #  It is 2: because derivative of filter signal.
    #  original hand data:   size N.  N-1 filtered time points, N-2 derivative points.  So our behavioral data is size N-2
    dat_pkg["behv_ts"] = hnd_dat[2:, 3]  
    print("behv_ts shape")
    print(dat_pkg["behv_ts"].shape)
    dat_pkg["hnd_dat"] = hnd_dat
          
    #chg_rps         = _N.loadtxt("Results/%s/mdl7b_chg_rps_mns" % rpsm_key)
    #btp_rps         = _N.loadtxt("Results/%s/mdl7b_btp_rps_mns" % rpsm_key)


    # combine this with time stamp

    ###########  Reported start times of RPS, EEG
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    reprtd_start_RPSM  = int(reprtd_start_RPSM_str[8:10])*3600 + 60*int(reprtd_start_RPSM_str[10:12]) + int(reprtd_start_RPSM_str[13:15])
    if gcoh_fn is not None:
         reprtd_start_EEG   = int(reprtd_start_EEG_str[0:2])*3600  + 60*int(reprtd_start_EEG_str[3:5])  + int(reprtd_start_EEG_str[6:8])
    else:
         reprtd_start_EEG = reprtd_start_RPSM
    d_reprtd_start = reprtd_start_RPSM - reprtd_start_EEG

    rpsm_hrs, rpsm_min, rpsm_secs = same_time_RPSM.split(":")
    eeg_hrs,  eeg_min,  eeg_secs  = same_time_EEG.split(":")

    t_rpsm = int(rpsm_hrs)*3600+int(rpsm_min)*60+int(rpsm_secs)
    t_eeg  = int(eeg_hrs)*3600 +int(eeg_min)*60 +int(eeg_secs)

    d_start = d_reprtd_start - (t_rpsm - t_eeg)

    print("recording start time difference between page load and EEG hit record (negative means EEG started earlier)  %d" % d_start)

    #######################################
    hnd_dat[:, 3] += d_start*1000
    #   RPS hands       N pts
    #   latent state is paired with previous hand and next hand obs (N-1) pts
    #   diff latent state is difference between points (N-2) pts
    #dchg_wtl_with_ts[:, 3]  = hnd_dat[2:, 3]
    #dbtp_wtl_with_ts[:, 3]  = hnd_dat[2:, 3]

    #dat_pkg["dchg_wtl_with_ts"] = dchg_wtl_with_ts
    #dat_pkg["dbtp_wtl_with_ts"] = dbtp_wtl_with_ts

    if gcoh_fn is not None:
         eeg_dat = _N.loadtxt("../DSi_dat/%(dsf)s_artfctrmvd/v%(av)d/%(dsf)s_artfctrmvd_v%(av)d.dat" % {"dsf" : dsi_fn, "av" : armv_ver, "gv" : gcoh_ver})
         gcoh_lm         = depickle("../DSi_dat/%(dsf)s_artfctrmvd/v%(av)d/%(dsf)s_%(gf)s_v%(av)d%(gv)d.dmp" % {"gf" : gcoh_fn, "dsf" : dsi_fn, "av" : armv_ver, "gv" : gcoh_ver})

    if not os.access(getResultFN("%(dsf)s" % {"dsf" : dsi_fn}), os.F_OK):
         os.mkdir(getResultFN("%(dsf)s" % {"dsf" : dsi_fn}))
    savedir = getResultFN("%(dsf)s/v%(av)d%(gv)d" % {"gf" : gcoh_fn, "dsf" : dsi_fn, "av" : armv_ver, "gv" : gcoh_ver})
    if not os.access(savedir, os.F_OK):
         os.mkdir(savedir)

    if gcoh_fn is not None:
        gcoh_fs   = gcoh_lm["fs"]
        imag_evs  = gcoh_lm["VEC"][:, :, 0:2]
        gcoh      = gcoh_lm["Cs"]

        print("imag_evs.shape")
        print(imag_evs.shape)
        num_f_lvls= len(gcoh_fs)
        L_gcoh    = imag_evs.shape[0]
        nChs      = imag_evs.shape[3]
        
        real_evs  = _N.empty((2, L_gcoh, num_f_lvls, nChs))
        print("real_evs.shape")
        print(real_evs.shape)
        for ti in range(L_gcoh):
            real_evs[0, ti] = _N.abs(imag_evs[ti, :, 0])
            real_evs[1, ti] = _N.abs(imag_evs[ti, :, 1])

        """
        mn = _N.mean(real_evs, axis=0)
        sd = _N.std(real_evs, axis=0)

        OUTLR = 10
        outlrs = []
        for ifr in range(num_f_lvls):
            for ich in range(nChs):
                abv = _N.where(real_evs[:, ifr, ich] > mn[ifr, ich] + OUTLR*sd[ifr, ich])[0]
                bel = _N.where(real_evs[:, ifr, ich] < mn[ifr, ich] - OUTLR*sd[ifr, ich])[0]
                outlrs.extend(abv)
                outlrs.extend(bel)
        unq_outlrs = _N.unique(outlrs)


        for io in unq_outlrs[0:-1]:
            real_evs[io+1]   = real_evs[io] + 0.1*sd*_N.random.randn()
        """
        dat_pkg["EIGVS"] = real_evs
        dat_pkg["fs"]    = gcoh_fs
        dat_pkg["Cs"]    = gcoh
        
        print(eeg_dat.shape[0])
        print(win_gcoh)
        print(slideby_gcoh)

        dat_pkg["ts_gcoh"]= overlapping_window_center_times(eeg_dat.shape[0], win_gcoh, slideby_gcoh, 300.)
        print(dat_pkg["ts_gcoh"])
        dat_pkg["gcoh_fn"]  = gcoh_fn
        ####  time 
        ts_eeg_dfind  = _N.linspace(0, eeg_dat.shape[0]/300., eeg_dat.shape[0])
        
        ch_spectrograms  = []
        maxHz         = 50

        for ch in range(21):
            spectrograms = []
            fs, ts, Sxx =  _ss.spectrogram(eeg_dat[:, ch], fs=300, nperseg=win_spec, noverlap=(win_spec - slideby_spec))
            use_fs = _N.where(fs < maxHz)[0]

            for ihz in use_fs:
                spectrograms.append(Sxx[ihz])
            ch_spectrograms.append(_N.array(spectrograms))

        dat_pkg["ch_spectrograms"] = ch_spectrograms
        dat_pkg["ts_spectrograms"] = ts
        dat_pkg["fs_spectrograms"] = fs
        dat_pkg["eeg_smp_dt"]          = 1./300
    dat_pkg["win_gcoh"]             = win_gcoh
    dat_pkg["slide_gcoh"]           = slideby_gcoh
    dat_pkg["win_spec"]             = win_spec
    dat_pkg["slide_spec"]           = slideby_spec


    return savedir, dat_pkg, win_gcoh, slideby_gcoh, armv_ver, gcoh_ver, label

#  THESE KEYS ARE RPS GAME DATA NAMES
#for rpsm_key in ["20Jan08-1703-13"]:
#for rpsm_key in ["20Jan09-1504-32"]:
for partID in ["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1331-06", "20200812_1252-50", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]:
#for partID in ["20210609_1248-16"]:    
#for partID in ["20210609_1747-07"]:
#for partID in [
#for rpsm_key in ["20Aug12-1331-06"]:
#for partID in ["20200109_1504-32"]:
#for rpsm_key in ["20Aug12-1252-50", "20Jan09-1504-32", "20Aug18-1644-09", "20Aug18-1624-01", "20Aug12-1331-06"]:
#for rpsm_key in ["20Jan08-1703-13"]:#, "20Jan09-1504-32", "20Aug12-1252-50", "2Aug12-1331-06", "20Aug18-1546-13"]:
#for rpsm_key in ["20Aug12-1252-50"]:
#for rpsm_key in ["20Aug18-1644-09"]:
#for rpsm_key in ["20Aug18-1546-13"]:
#for rpsm_key in ["20Aug18-1603-42"]:
#for rpsm_key in ["20Aug18-1624-01"]:
#for rpsm_key in ["20Aug18-1603-42"]:
#for partID in ["20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09"]:
#for partID in ["20200108_1642-20", "20200812_1331-06"]:
    savedir, dat_pkg, win_gcoh, slideby_gcoh, armv_ver, gcoh_ver, label = pkg_all_data(partID)  #  first nperseg/2 points are constant
    #pkg_all_data(rpsm_key)  #  first nperseg/2 points are constant

    #  combine
    pklfn = "%(sd)s/%(rk)s_%(w)d_%(s)d_pkld_dat_v%(av)d%(gv)d_%(lb)d.dmp" % {"rk" : partID, "w" : win_gcoh, "s" : slideby_gcoh, "av" : armv_ver, "gv" : gcoh_ver, "sd" : savedir, "lb" : label}
    dmp = open(pklfn, "wb")

    pickle.dump(dat_pkg, dmp, -1)
    dmp.close()

    bhv = dat_pkg["behv"]
    fbhv = dat_pkg["fbehv"]

