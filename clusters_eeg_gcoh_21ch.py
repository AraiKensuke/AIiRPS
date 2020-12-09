import numpy as _N
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
from scipy.signal import savgol_filter
from sklearn import mixture
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, mtfftc
import skull_plot as _sp
import os
import AIiRPS.rpsms as rpsms
import GCoh.preprocess_ver as _ppv
from AIiRPS.utils.dir_util import getResultFN


import sys
from sumojam.devscripts.cmdlineargs import process_keyval_args

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

def autocorrelate(signal, maxlag):
     AC = _N.empty(maxlag*2+1)
     sigN= signal.shape[0]
     lag0mag = _N.dot(signal, signal) / (sigN*sigN)
     for lg in range(1, maxlag+1):
          datlen = sigN-lg
          AC[maxlag-lg] = (_N.dot(signal[0:datlen], signal[lg:lg+datlen])/(datlen*datlen)) / lag0mag
          AC[maxlag+lg] = AC[maxlag-lg]
     AC[maxlag] = 1
     return AC

arr_ch_names=_N.array(["P3", "C3", "F3", "Fz", "F4",   
                       "C4", "P4", "Cz", "Pz", "A1",    #  "Pz" is "CM"
                       "Fp1","Fp2","T3", "T5", "O1",
                       "O2", "F7", "F8", "A2", "T6",
                       "T4"])

ch_names = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "T5'", "T6'", "O1'", "O2'", "P3\"", "P4\"", "PZ\"", "OZ", "I", "CB1\"", "CB2\"", "CB1", "CB2"]


Fs=300
ev_n   = 0
ignore_stored = False

_WIDE = 0
_MED  = 1
_FINE = 2
_FINE1 = 3   #

#dat     = "Apr112020_21_26_01"#"Apr312020_16_53_03"
#dat     = "Apr112020_13_00_00"#"Apr312020_16_53_03"
#dat     = "Apr242020_09_00_00"#"Apr312020_16_53_03"
#dat     = "Apr242020_13_00_00"#"Apr312020_16_53_03"
dat      = "Jan082020_17_03_48"
#dat      = "Jan082020_16_56_08"
#dat      = "Jan092020_14_55_38"
#dat     = "Jan092020_14_00_00"#"Apr312020_16_53_03"
#dat     = "Jan092020_15_05_39"#"Apr312020_16_53_03"
#dat     = "May042020_22_23_04"#"Apr312020_16_53_03"
#dat     = "May052020_13_00_00"#"Apr312020_16_53_03"
#dat     = "May052020_21_08_01"
#dat     = "May052020_14_00_00"
#dat     = "May082020_23_34_58"
#dat =      "Apr152020_20_34_20"
#dat =      "Apr152020_13_00_00"
#dat =      "May042020_14_00_00"
#dat =      "Apr242020_12_00_00"
#dat =      "Apr182020_22_02_03"
#dat =      "Apr182020_13_00_00"   possible
#dat =      "Apr242020_16_53_03"
#dat  = "May142020_23_16_34"   #  35 seconds    #  15:04:32
#dat  = "May142020_23_31_04"   #  35 seconds    #  15:04:32
#dat  = "May142020_13_00_00"   #  35 seconds    #  15:04:32
#dat  = "Jul012020_12_00_00"
#dat  = "Aug122020_13_17_39"
#dat  = "Aug122020_12_52_44"
#dat  = "Aug122020_13_30_23"
#dat  = "Aug182020_13_57_26"
#dat   = "Aug182020_15_45_27"
#dat  = "Aug182020_16_44_18"
#dat  = "Aug182020_16_25_28"
#dat  = "Jan012019_16_00_00"

#bin     = 512
#slide   = 64
_WIDE = 0
_FINE = 1
wid_fin=_FINE1
s_res  = "wide" if wid_fin == _WIDE else "fine"

manual_cluster=False
armv_ver = 1
gcoh_ver = 3   #  bandwidth 7 ver 1, bandwidth 5 ver 2, bandwidth 9 ver 3

frngs = [[35, 47]]
process_keyval_args(globals(), sys.argv[1:])
win, slideby      = _ppv.get_win_slideby(gcoh_ver)

hlfOverlap = int((win/slideby)*0.5)


#s = "../Neurable/DSi_dat/%(dsf)s_artfctrmvd_v%(av)d/%(dsf)s_gcoh_%(wn)d_%(sld)d_v%(av)d%(gv)d.dmp" % {"gf" : rpsm[dat], "dsf" : dat, "av" : armv_ver, "gv" : gcoh_ver, "wn" : bin, "sld" : slide}
#print("!!!!!!!!!!   %s" % s)
lm         = depickle("../DSi_dat/%(dsf)s_artfctrmvd/v%(av)d/%(dsf)s_gcoh_%(wn)d_%(sld)d_v%(av)d%(gv)d.dmp" % {"gf" : rpsms.rpsm_eeg_as_key[dat], "dsf" : dat, "av" : armv_ver, "gv" : gcoh_ver, "wn" : win, "sld" : slideby})
# #lm         = depickle("../Neurable/DSi_dat/%(dat)s_gcoh_%(w)s_%(s)s.dmp" % {"dat" : dat, "w" : bin, "s" : slide})
# #A_gcoh_mat = _scio.loadmat("DSi_dat/%(dat)s_gcoh_%(w)d_%(sl)d.mat" % {"dat" : dat, "w" : bin, "sl" : slide})
# #A_gcoh     = A_gcoh_mat["Cs"]

strt       = 0  #  if start at middle of experiment
A_gcoh     = lm["Cs"][strt:]
n_fs       = lm["fs"]


outdir     = getResultFN("%(dir)s/v%(av)d%(gv)d" % {"dir" : dat, "av" : armv_ver, "gv" : gcoh_ver})
if not os.access(getResultFN(dat), os.F_OK):
     os.mkdir(getResultFN(dat))
################  egenvectors
#imag_evs  = A_gcoh_mat["VEC"][0]


imag_evs  = lm["VEC"][strt:, :, ev_n]

L_gcoh  = A_gcoh.shape[0]
nChs    = imag_evs.shape[2]
real_evs  = _N.empty((L_gcoh, n_fs.shape[0], nChs))

chs = lm["chs_picks"]
ch_names = arr_ch_names[chs].tolist()

for ti in range(L_gcoh):
    real_evs[ti] = _N.abs(imag_evs[ti])

mn = _N.mean(real_evs, axis=0)
sd = _N.std(real_evs, axis=0)

fs = lm["fs"]


#frngs = [[12, 18], [20, 25], [28, 35], [38, 45]]
#frngs = [[12, 18], [20, 25], [28, 35], [35, 42], [38, 45]]
#frngs = [[10, 15]]
#frngs = [[10, 15]]
#frngs = [[12, 18]]
#frngs = [[10, 15], [20, 25], [30, 40]]
#frngs = [[18, 25]]
#frngs = [[35, 42]]

#frngs = [[25, 35]]
#frngs = [[43, 49]]
#frngs = [[40, 50]]
#frngs = [[35, 45]]
#frngs = [[38, 45]]
frngs = [[35, 47]]
#frngs = [[8, 12], [12, 18]]
#frngs = [[33, 40], [34, 41], [35, 42], [36, 43], [37, 44]]
#frngs = [[12, 18], [18, 25], [25, 35], [35, 45]]
#frngs = [[22, 28], [35, 42]]
#frngs = [[28, 35], [38, 45]]

pcs     = _N.empty(len(frngs))
minK    = 6
maxK    = 8
try_Ks  = _N.arange(minK, maxK+1)
#TRs      = _N.array([1, 1, 3, 5, 10, 15, 20, 25, 25])  # more tries for higher K
TRs      = _N.array([1, 15, 20, 25, 25, 30, 40, 50, 60, 60, 60, 60, 60, 60, 60])  # more tries for higher K
#TRs      = _N.array([60])  # more tries for higher K

bics = _N.ones(((maxK-minK), _N.max(TRs)))*1000000
labs = _N.empty((maxK-minK, _N.max(TRs), real_evs.shape[0]), dtype=_N.int)

nState_start = 0

for ich in range(len(frngs)):
    fL = frngs[ich][0]
    fH = frngs[ich][1]

    irngs = _N.where((fs > fL) & (fs < fH))[0]
    iL    = irngs[0]
    iH    = irngs[-1]    

    #Apr242020_16_53_03_gcoh_256_64
    nStates, rmpd_lab = find_or_retrieve_GMM_labels(dat, "%(gf)s_gcoh%(evn)d_%(wn)d_%(sld)d_v%(av)d%(gv)d" % {"gf" : dat, "av" : armv_ver, "gv" : gcoh_ver, "wn" : win, "sld" : slideby, "evn" : ev_n}, real_evs, iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=try_Ks, TRs=TRs, ignore_stored=ignore_stored, manual_cluster=manual_cluster, do_pca=True, min_var_expld=0.99)
    ps = _N.arange(nStates)
    ps += nState_start
    nState_start += nStates

    #nStates, rmpd_lab = find_or_retrieve_GMM_labels(rpsm[dat], "%(dat)s_gcoh_%(w)d_%(s)d" % {"dat" : dat, "w" : bin, "s" : slide}, real_evs, iL, iH, fL, fH, which=0, try_K=try_Ks, TRs=TRs, log_transform=False)
    """
    ###############
    for K in range(minK, maxK):
        for tr in range(TRs[K]):
            gmm = mixture.GaussianMixture(n_components=K, covariance_type="full")

            gmm.fit(_N.sum(real_evs[:, iL:iH], axis=1))
            bics[K-minK, tr] = gmm.bic(_N.sum(real_evs[:, iL:iH], axis=1))
            labs[K-minK, tr] = gmm.predict(_N.sum(real_evs[:, iL:iH], axis=1))

    coords = _N.where(bics == _N.min(bics))
    print("min bic %.4e" % _N.min(bics))
    bestLab = labs[coords[0][0], coords[1][0]]   #  indices in 2-D array
    rmpd_lab = increasing_labels_mapping(bestLab)

    nStates =  list(range(minK, maxK))[coords[0][0]]
    """
    out_u = _N.mean(real_evs[:, iL:iH], axis=1)
    out = _N.empty((L_gcoh, nChs))
    iS  = 0
    for ns in range(nStates):
        ls = _N.where(rmpd_lab == ns)[0]
        out[iS:iS+len(ls)] = _N.mean(real_evs[ls, iL:iH], axis=1)
        iS += len(ls)

    iS = 0
    clrs  = ["black", "orange", "blue", "green", "red", "lightblue", "grey", "pink", "yellow", "brown", "cyan", "purple", "black", "orange", "blue", "green", "red"]
    W   = L_gcoh
    H   = nChs
    disp_wh_ratio = 3
    aspect = (W/H)/disp_wh_ratio
    unit = 2.5
    fig = _plt.figure(figsize=(disp_wh_ratio*unit + 1, 3*unit+unit/2))
    _plt.subplot2grid((2, 1), (0, 0))        
    _plt.title("1st GCoh eigenvector - temporal order")
    #fig.add_subplot(nStates+2, 1, 1)  
    _plt.imshow(out_u.T, aspect=aspect)
    _plt.ylim(-(nStates+2), nChs+0.1)
    for ns in range(nStates):
        nsx = _N.where(rmpd_lab == ns)[0]
        _plt.scatter(nsx, _N.ones(len(nsx))*ns - nStates - 1, color=clrs[ns], lw=1.5, s=4)
    _plt.xlim(0, L_gcoh)
    _plt.xlabel("(sample #) - not in experimental temporal order", fontsize=17)
    _plt.ylabel("electrode #", fontsize=16)
    _plt.xlabel("time bin", fontsize=16)
    _plt.xticks(fontsize=14)
    _plt.yticks(fontsize=14)

    _plt.subplot2grid((2, 1), (1, 0))        
    _plt.title("1st GCoh eigenvector - reordered by cluster label")
    #fig.add_subplot(nStates+2, 1, 1)    
    _plt.imshow(out.T, aspect=aspect)
    _plt.ylim(-(nStates+2), nChs+0.1)
    for ns in range(nStates):
        ls = _N.where(rmpd_lab == ns)[0]
        liS = iS
        iS += len(ls)
        _plt.plot([liS, iS], [ns-nStates-1, ns-nStates-1], color=clrs[ns], lw=3.5)
        if ns < nStates-1:
            _plt.axvline(x=iS, color="white", lw=1)
    _plt.xlim(0, L_gcoh)
    _plt.suptitle("%(ky)s   %(1)d-%(2)dHz    GCoh val: %(gcoh).3f   %(sts)s" % {"1" : fL, "2" : fH, "gcoh" : _N.mean(lm["Cs"][:, irngs]), "ky" : dat, "sts" : str(ps)})
    _plt.xlabel("(sample #) - not in experimental temporal order", fontsize=17)
    _plt.ylabel("electrode #", fontsize=16)
    _plt.xlabel("time bin", fontsize=16)
    _plt.xticks(fontsize=14)
    _plt.yticks(fontsize=14)
    

    iS = 0
    for ns in range(nStates):
        ls = _N.where(rmpd_lab == ns)[0]
        iS += len(ls)
        if ns < nStates-1:
            _plt.axvline(x=iS, color="white", lw=1)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, hspace=0.3)

    _plt.savefig("%(od)s/%(dat)s_%(w)d_%(sl)d_clusters_coh_pattern_%(evn)d_%(1)d_%(2)d_v%(av)d%(gv)d" % {"1" : fL, "2" : fH, "dat" : dat, "w" : win, "sl" : slideby, "od" : outdir, "av" : armv_ver, "gv" : gcoh_ver, "evn" : ev_n}, transparent=True)
    #_plt.close()

    max_over_fs_each_state = _N.empty((nChs, nStates))
    for ns in range(nStates):
        ls = _N.where(rmpd_lab == ns)[0]
        mn_over_fs = _N.mean(real_evs[ls, iL:iH], axis=1)
        #min_all    = _N.min(mn_over_fs, axis=0)
        max_over_fs_each_state[:, ns]    = _N.max(mn_over_fs, axis=0)
    maxComp = _N.max(max_over_fs_each_state)

    all_vecs = _N.empty((nChs, nStates))
    
    for ns in range(nStates):
        ls = _N.where(rmpd_lab == ns)[0]
        mn_over_fs = _N.mean(real_evs[ls, iL:iH], axis=1)
        min_all    = _N.min(mn_over_fs, axis=0)
        all_vecs[:, ns] = (min_all / maxComp)*1e-5
    _sp.do_skull_plot_all_EVs(all_vecs, ps, ch_names, "%(od)s/%(dat)s_%(w)d_%(sl)d_skull_coh_pattern_%(evn)d_%(1)d_%(2)d_v%(av)d%(gv)d" % {"1" : fL, "2" : fH, "dat" : dat, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n}, dat, fL, fH)

    sts = _N.zeros(real_evs.shape[0])
    fig = _plt.figure(figsize=(12, 3))

    SHUFFLES = 100
    maxlags=150
    acfs     = _N.empty((SHUFFLES+1, maxlags*2+1))
    shf_rmpd_lab = _N.empty((SHUFFLES+1, rmpd_lab.shape[0]), dtype=_N.int)
    shf_rmpd_lab[0] = rmpd_lab
    
    for shf in range(1, SHUFFLES+1):
         rl = shift_correlated_shuffle(rmpd_lab, low=hlfOverlap, high=(hlfOverlap*3), local_shuffle=True, local_shuffle_pcs=6)
         #rl = shift_correlated_shuffle(rmpd_lab, low=1, high=2, local_shuffle=True, local_shuffle_pcs=6)
         shf_rmpd_lab[shf] = rl

    for ns in range(nStates):
        _plt.subplot2grid((1, nStates), (0, ns))

        for hlvs in range(2):
            t0 = hlvs*(L_gcoh//2)
            t1 = (hlvs+1)*(L_gcoh//2)
            for shf in range(SHUFFLES+1):
                sts[:]=0
                sts[_N.where(shf_rmpd_lab[shf] == ns)[0]] = 1
                #_plt.acorr(sts - _N.mean(sts), maxlags=150, usevlines=False, ms=2, color=clr, lw=lw)
                acfs[shf] = autocorrelate(sts[t0:t1] - _N.mean(sts[t0:t1]), maxlags)
                acfs[shf, maxlags] = 0
            sACFS = _N.sort(acfs[1:], axis=0)
            _plt.plot(_N.arange(-maxlags, maxlags+1), acfs[0] + (1-hlvs)*0.7, color="black", lw=2)
            _plt.fill_between(_N.arange(-maxlags, maxlags+1), sACFS[int(SHUFFLES*0.025)] + 0.7*(1-hlvs), sACFS[int(SHUFFLES*0.975)] + 0.7*(1-hlvs), alpha=0.3, color="blue")

            #_plt.xticks([-(Fs/slideby)*15, -(Fs/slideby)*10, -(Fs/slideby)*5, 0, (Fs/slideby)*5, (Fs/slideby)*10, (Fs/slideby)*15], [-15, -10, -5, 0, 5, 10, 15], fontsize=15)   #stroop
            #_plt.xticks([-(Fs/slideby)*45, -(Fs/slideby)*30, -(Fs/slideby)*15, 0, (Fs/slideby)*15, (Fs/slideby)*30, (Fs/slideby)*45], [-45, -30, -15, 0, 15, 30, 45], fontsize=15)   #RPS
            _plt.xticks([-(Fs/slideby)*30, -(Fs/slideby)*20, -(Fs/slideby)*10, 0, (Fs/slideby)*10, (Fs/slideby)*20, (Fs/slideby)*30], [-30, -20, -10, 0, 10, 20, 30], fontsize=15)   #RPS
            _plt.yticks(fontsize=14)
            #_plt.ylim(-0.08, 0.2)
            _plt.ylim(-0.08, 1.4)
            #_plt.xlim(-(Fs/slideby)*15, (Fs/slideby)*15)    #  Stroop
            #_plt.xlim(-(Fs/slideby)*50, (Fs/slideby)*50)    #  RPS
        _plt.xlim(-(Fs/slideby)*30, (Fs/slideby)*30)    #  RPS
        _plt.grid(ls=":")
        _plt.xlabel("lag (seconds)", fontsize=16)
        _plt.ylabel("autocorrelation", fontsize=16)
        _plt.title("pattern %d" % ns)
    _plt.suptitle("%(1)d-%(2)dHz" % {"1" : fL, "2" : fH})
    fig.subplots_adjust(left=0.15, bottom=0.2, wspace=0.4, right=0.98, top=0.9)
    _plt.savefig("%(od)s/%(dat)s_%(w)d_%(sl)d_acorr_%(evn)d_%(1)d_%(2)d_v%(av)d%(gv)d" % {"1" : fL, "2" : fH, "dat" : dat, "w" : win, "sl" : slideby, "av" : armv_ver, "gv" : gcoh_ver, "od" : outdir, "evn" : ev_n}, transparent=True)

"""
    ##############  temporary
    delays            = _N.array([1, -30])
    dat_tms_fn        = "%s_tms.dat" % dat
    
    print("#############################################")
    print("dat:  %s" % dat)
    print("ev:   %d" % ev_n)
    print("fr:   %s" % str(frngs[ich]))
    dat_tms = _N.loadtxt("../Neurable/DSi_dat/%s" % dat_tms_fn, dtype=_N.int)
    
    con     = _N.where(_N.where(dat_tms[:, 3] == 0)[0] > 62)[0]
    inc     = _N.where(_N.where(dat_tms[:, 3] == 1)[0] > 62)[0]
    
    for dly in delays:
         print("dly: %d" % dly)
         start_con   = _N.array(dat_tms[con, 0]/64+dly, dtype=_N.int)   #  before con
         start_inc   = _N.array(dat_tms[inc, 0]/64+dly, dtype=_N.int)   #  before inc

         print("-----")
         for ns in range(nStates):
              ntms = len(_N.where(rmpd_lab[start_inc] == ns)[0])
              print("before inc, # of pattern %(n)3d  %(r).3f" % {"n" : ntms, "r" : (ntms / len(start_inc))})

         print("-----")
         for ns in range(nStates):
              ntms = len(_N.where(rmpd_lab[start_con] == ns)[0])   #  all trials
              print("before con, # of pattern %(n)3d  %(r).3f" % {"n" : ntms, "r" : (ntms / len(start_con))})


"""
