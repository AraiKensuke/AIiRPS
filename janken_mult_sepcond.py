#!/usr/bin/python

import scipy.stats as _ss
import numpy as _N  #  must import this before pyPG when running from shell
import pyPG as lw
import LOSTtmp.kfARlib1c as _kfar
import pickle

import read_taisen as _rd
import matplotlib.pyplot as _plt
import os
import sys
import labels
from cmdlineargs import process_keyval_args

from dir_util import getResultFN

def union_arrs(a1, a2):
     return _N.sort(_N.array(a1.tolist() + a2.tolist()))

def fill_unobserved(arr):
    #  we assume old value is kept until new observation
    nz = _N.where(arr != -100)[0]
    iLastObs = arr[0] if (arr[0] != -100) else _N.mean(arr[nz])
    for i in range(1, arr.shape[0]):
        arr[i] = arr[i] if (arr[i] != -100) else iLastObs
        if arr[i] != -100:
            iLastObs = arr[i]
        
            
    
def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

###
#   Here the latent state is B1, B2.
#   Probability of going to stay, weaker, stronger for just the WIN condition
#   

#  input data
#  m_hands    # machine hands  size N         1 -1  1
#  h_hands    # human hands    size N        -1 -1  1
#  -1 -1  1  1 -1  1 
#      0  1  0  1  1

fns     = ["ME,WTL", "ME,RPS", "AI,WTL", "AI,RPS", "OU,WTL", "OU,RPS"]

fnt_tck = 10
fnt_lbl = 12

fnt_tck = 15
fnt_lab = 16
_WTL = 10
_RPS = 11
_ME_WTL = 0
_ME_RPS = 1
_AI_WTL = 2
_AI_RPS = 3
_OU_WTL = 4
_OU_RPS = 5


_RELATIVE_LAST_ME = 1    #  what to use as a comparison for when player switches
_RELATIVE_LAST_AI = 2
_RELATIVE_LAST_OU = 3    #  next outcome outcome

smp_every = 50
ITER = 30000
it0  = 28000
it1  = 30000

label=8
#  priors for inv gamma   #  B/(a+1)

###  simulation
a_q2 = 5
B_q2= 5.
###  1504
a_q2 = 6
B_q2= 12.

#a_q2 = 10.
#a_q2 = 1.01
#a_q2 = 50.
#B_q2 = 200.
#B_q2= 80.
#B_q2= 30

#B_q2 = 30

rndmz = False
capped = 0
l_capped= []

"""
B_n     AR to sample
"""
def sampleAR_and_offset(it, Tm1, vrnc, vrncL, \
                        B_n, offset, \
                        kappa, ws, q2_B_n, a_F0, b_F0, a_q2, B_q2, px, pV, fx, fV, K, random_walk):
    global capped, l_capped
    offset_mu = kappa / ws - B_n
    mu_w  = _N.sum(offset_mu*ws) / _N.sum(ws)   #  from likelihood
    mu  = (mu_w*off_sig2 + off_mu*vrncL) / (off_sig2 + vrncL) # lklhd & prior
    offset[:] = mu + _N.sqrt(vrnc)*_N.random.randn()

    F0AA = _N.dot(B_n[0:-1], B_n[0:-1])
    F0BB = _N.dot(B_n[0:-1], B_n[1:])

    F0_B_n = 1
    if not random_walk:
        F0std= _N.sqrt(q2_B_n/F0AA)
        F0a, F0b  = (a_F0 - F0BB/F0AA) / F0std, (b_F0 - F0BB/F0AA) / F0std
        F0_B_n=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)

    #   sample q2
    a = a_q2 + 0.5*Tm1  #  N + 1 - 1
    #a = 0.5*Nm1  #  N + 1 - 1
    rsd_stp = B_n[1:] - F0_B_n*B_n[0:-1]
    BB = B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
    #BB = 0.5 * _N.dot(rsd_stp, rsd_stp)
    q2_B_n = _ss.invgamma.rvs(a, scale=BB)

    y             = kappa/ws - offset
    Rv = 1. / ws

    _kfar.armdl_FFBS_1itr_singletrial(Tm1, y, Rv, F0_B_n, q2_B_n,
                                      fx, fV, px, pV, B_n, K)    
    near1 = _N.where(B_n > 300)[0]
    if len(near1) > 0:
        B_n[near1] = 300   #  cap it   #  prevents overflow in exp
        capped += 1
        l_capped.append(it)
        
    return offset, F0_B_n, q2_B_n

a_F0      = 0.5;    b_F0      =  1

######  int main 
#int i,pred,m,v[3],x[3*N+1],w[9*N+3],fw[3];

know_gt  = False
signal   = _RELATIVE_LAST_ME
covariates = _WTL
sig_cov   = _ME_WTL

tr0      = 0
tr1      = -1

#dat_fn      = "20Jan09-1455-06"

dat_fn      = "20Nov10-1235-20"
dat_fn      = "20Nov10-1321-15"
dat_fn      = "20Nov10-1400-09"
dat_fn      = "20Nov10-1403-04"
dat_fn      = "20Nov10-1517-18"
dat_fn      = "20Nov10-1522-53"
dat_fn      = "20Nov10-1524-41"
dat_fn      = "20Nov10-1526-30"
dat_fn      = "20Nov10-1532-15"
dat_fn      = "20Nov10-1535-16"
dat_fn      = "20Nov10-1537-59"
#dat_fn      = "20Jan09-1504-32"
#dat_fns="20Nov08-0904-26"
# dat_fn      = "20Aug12-1252-50"
#dat_fn      = "20Nov09-1623-42"
# dat_fn      = "20Nov10-1733-51"
# dat_fn      = "20Nov10-1756-55"
# dat_fn      = "20Nov10-1759-28"
# dat_fn      = "20Nov10-1808-16"
# dat_fn      = "20Nov10-1819-10"
# dat_fn      = "20Nov10-1823-22"
dat_fn      = "20Aug18-1644-09"
#dat_fn       = "20Nov10-2157-31"
#dat_fn       = "20Nov11-0502-55"

#### USE simulation
#dat_fn       = "20Nov11-0515-04"
# dat_fn        = "20Nov11-0728-51"
# dat_fn        = "20Nov11-0734-43"
# dat_fn        = "20Nov11-0740-13"
dat_fn = "20Aug18-1546-13"
dat_fn = "20Aug18-1603-42"

#####USE real
#dat_fn        = "20Aug18-1624-01"

#dat_fn="20Jun01-0748-03"
#dat_fn="20May29-1923-44"
#dat_fn="20May29-1419-14"
#
#dat_fn      = "20Jan09-1504-32"
#dat_fn="20Aug12-1331-06"
#dat_fn="20Jan08-1703-13"
#dat_fn="20Aug12-1252-50"
#dat_fn="20Nov22-1108-25"
#dat_fn="20Nov21-2131-38"
#dat_fn="20Nov21-1959-30"

random_walk = True
flip_human_AI=False
#########################################
process_keyval_args(globals(), sys.argv[1:])
a_q2, B_q2 = labels.get_a_B(label)

scov = "WTL"
ssig = "ME"

if tr1 < 0:
    tr1 = None
else:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!  tr1 %d" % tr1)
sran   = ""

s_flip = "_flip" if flip_human_AI else ""
if not know_gt: 
    _hnd_dat = _rd.return_hnd_dat(dat_fn, flip_human_AI=flip_human_AI)    
else:
    _hnd_dat     = _N.loadtxt("/Users/arai/nctc/Workspace/AIiRPS_SimDAT/rpsm_%s.dat" % dat_fn, dtype=_N.int)
    lmGT = depickle("/Users/arai/nctc/Workspace/AIiRPS_SimDAT/rpsm_%s.dmp" % dat_fn)
    Ts = lmGT["Ts_timeseries"]
    
if tr1 is None:
    tr1          = _hnd_dat.shape[0]
    rr_hnd_dat     = _N.array(_hnd_dat)
else:
    rr_hnd_dat      = _N.array(_hnd_dat[tr0:tr1])

inds = _N.arange(tr1)        
if rndmz:
    _N.random.shuffle(inds)
    sran = "rndmz"
hnd_dat = _N.array(rr_hnd_dat[inds])
N_all  = hnd_dat.shape[0]-1
smp_offsets_all = _N.empty((6, ITER, N_all))
smp_Bns_all     = _N.ones((6, ITER, N_all))*-100

conds = []
if covariates == _RPS:
    conds.append(_N.where(hnd_dat[0:-1, 0] == 1)[0])
    conds.append(_N.where(hnd_dat[0:-1, 0] == 2)[0])
    conds.append(_N.where(hnd_dat[0:-1, 0] == 3)[0])
elif covariates == _WTL:
    conds.append(_N.where(hnd_dat[0:-1, 2] == 1)[0])
    conds.append(_N.where(hnd_dat[0:-1, 2] == 0)[0])
    conds.append(_N.where(hnd_dat[0:-1, 2] == -1)[0])



out_dir = getResultFN("%(dfn)s" % {"dfn" : dat_fn})
if not os.access(out_dir, os.F_OK):
    os.mkdir(out_dir)
#out_dir = "Results/%(dfn)s/%(lbl)d" % {"dfn" : dat_fn, "lbl" : label}
out_dir = getResultFN("%(dfn)s/%(lbl)d" % {"dfn" : dat_fn, "lbl" : label})
if not os.access(out_dir, os.F_OK):
    os.mkdir(out_dir)


if signal == _RELATIVE_LAST_ME:
    col_n0 = 0    #  current
    col_n1 = 0    #  previous
elif signal == _RELATIVE_LAST_AI:
    col_n0 = 0    #  did player copy AI's last move
    col_n1 = 1    #  or did player go to move that beat (loses) to the last AI
elif signal == _RELATIVE_LAST_OU:
    col_n0 = 0    #  did player copy AI's last move
    col_n1 = 2    #  or did player go to move that beat (loses) to the last AI

#  can't directly use hnd_dat, as it condX are discontinuous, 

if sig_cov == _ME_WTL:
    stay_win, wekr_win, strg_win, \
    stay_tie, wekr_tie, strg_tie, \
    stay_los, wekr_los, strg_los, \
    win_cond, tie_cond, los_cond = _rd.get_ME_WTL(hnd_dat, tr0, tr1)
    #  p(stay | W)     

    nWins = len(win_cond)
    nTies = len(tie_cond)
    nLoss = len(los_cond)

    cond_events = [[stay_win, wekr_win, strg_win],
                   [stay_tie, wekr_tie, strg_tie],
                   [stay_los, wekr_los, strg_los]]
    off_cond_events = [[union_arrs(wekr_win, strg_win), union_arrs(stay_win, strg_win), union_arrs(stay_win, wekr_win)],
                       [union_arrs(wekr_tie, strg_tie), union_arrs(stay_tie, strg_tie), union_arrs(stay_tie, wekr_tie)],
                       [union_arrs(wekr_los, strg_los), union_arrs(stay_los, strg_los), union_arrs(stay_los, wekr_los)]]
    marg_cond_events = [win_cond, tie_cond, los_cond]

    static_cnd_probs = _N.array([[len(stay_win) / nWins, len(wekr_win) / nWins, len(strg_win) / nWins],
                                 [len(stay_tie) / nTies, len(wekr_tie) / nTies, len(strg_tie) / nTies],
                                 [len(stay_los) / nLoss, len(wekr_los) / nLoss, len(strg_los) / nLoss]])
else:
    stay_R, strg_R, wekr_R, stay_P, strg_P, wekr_P, stay_S, strg_S, wekr_S, R_cond, P_cond, S_cond = _rt.get_ME_RPS(hnd_dat, tr0, tr1)
    
    nRs = len(R_cond)
    nPs = len(P_cond)
    nSs = len(S_cond)
    
    # cond_events = [[stay_R, strg_R, wekr_R],
    #                [stay_P, strg_P, wekr_P],
    #                [stay_S, strg_S, wekr_S]]
    cond_events = [[stay_R, wekr_R, strg_R],
                   [stay_P, wekr_P, strg_P],
                   [stay_S, wekr_S, strg_S]]
    
    marg_cond_events = [R_cond, P_cond, S_cond]
    
    static_cnd_probs = _N.array([[len(stay_R) / nRs, len(wekr_R) / nRs, len(strg_R) / nRs],
                                 [len(stay_P) / nPs, len(wekr_P) / nPs, len(strg_P) / nPs],
                                 [len(stay_S) / nSs, len(wekr_S) / nSs, len(strg_S) / nSs]])

s_WTL_conds = ["W", "T", "L"] if sig_cov == _ME_WTL else ["R", "P", "S"]
if sig_cov == _OU_WTL:
    s_trans = ["W", "T", "L"]
elif (sig_cov == _ME_WTL) or (sig_cov == _ME_RPS):
    s_trans = ["stay", "wekr", "strgr"]

    

cond_probs = _N.empty((3, 3, hnd_dat.shape[0]-1))
upper_conf = _N.empty(hnd_dat.shape[0]-1)
lower_conf = _N.empty(hnd_dat.shape[0]-1)


fig = _plt.figure(figsize=(12, 7))
for cond in range(3):
    thecond = conds[cond]
    Tm1     = len(thecond)

    y_vec = _N.zeros((Tm1, 3), dtype=_N.int)
    y     = 100*_N.ones(Tm1, dtype=_N.int)   #  indices of the random var

    ####  OBSERVATION
    ni = -1

    for n in range(Tm1):   #  
        nc = thecond[n]
        ni += 1
        if (hnd_dat[nc+1, col_n0] == hnd_dat[nc, col_n1]):
            y[ni] = 0  #  Goo, choki, paa   goo->choki
            #   choki->paa
            y_vec[ni, 0] = 1    #  [1, 0, 0]    stay
        elif ((hnd_dat[nc+1, col_n0] == 1) and (hnd_dat[nc, col_n1] == 3)) or \
             ((hnd_dat[nc+1, col_n0] == 2) and (hnd_dat[nc, col_n1] == 1)) or \
             ((hnd_dat[nc+1, col_n0] == 3) and (hnd_dat[nc, col_n1] == 2)):
             #  nc is paa, nc+1 is goo
            y[ni] = -1
            y_vec[ni, 1] = 1    #  [0, 1, 0]    choose weaker
        elif ((hnd_dat[nc+1, col_n0] == 1) and (hnd_dat[nc, col_n1] == 2)) or \
             ((hnd_dat[nc+1, col_n0] == 2) and (hnd_dat[nc, col_n1] == 3)) or \
             ((hnd_dat[nc+1, col_n0] == 3) and (hnd_dat[nc, col_n1] == 1)):
            y[ni] = 1
            y_vec[ni, 2] = 1    #  [0, 0, 1]    choose stronger

    """
    fig.add_subplot(3, 1, cond+1)
    i_sty= _N.where(y == 0)[0]
    i_dn= _N.where(y == -1)[0]
    i_up= _N.where(y == 1)[0]
    upto = len(thecond)
    _plt.scatter(i_sty, _N.ones(len(i_sty))*1, s=5)
    _plt.xlim(-1, upto+1)
    _plt.ylim(-2, 2)
    _plt.scatter(i_dn,  _N.ones(len(i_dn))*0, marker="|")
    _plt.xlim(-1, upto+1)
    _plt.ylim(-2, 2)
    _plt.scatter(i_up,  _N.ones(len(i_up))*-1, marker="|")
    _plt.xlim(-1, upto+1)
    _plt.ylim(-2, 2)
    """
    w1_px        = _N.random.randn(Tm1)
    w1_pV        = _N.ones(Tm1)*0.2
    w1_fx        = _N.zeros(Tm1)
    w1_fV        = _N.ones(Tm1)*0.1
    w2_px        = _N.random.randn(Tm1)
    w2_pV        = _N.random.rand(Tm1)
    w2_fx        = _N.zeros(Tm1)
    w2_fV        = _N.ones(Tm1)*0.1


    w1_K         = _N.empty(Tm1)
    w2_K         = _N.empty(Tm1)

    ws1  = _N.random.rand(Tm1)
    ws2  = _N.random.rand(Tm1)

    K     = 3
    N_vec = _N.zeros((Tm1, K), dtype=_N.int)     #  The N vector
    N     = 1
    kappa   = _N.empty((Tm1, K))

    for n in range(Tm1):
      N_vec[n, 0] = 1
      for k in range(1, K):
        N_vec[n, k] = N - _N.sum(y_vec[n, 0:k])
      for k in range(K):
        kappa[n, k] = y_vec[n, k] - 0.5 * N_vec[n, k]

    zr2  = _N.where(N_vec[:, 1] == 0)[0]     #  dat where N_2 == 0  (only 1 PG var)
    nzr2 = _N.where(N_vec[:, 1] == 1)[0]   

    smp_offsets = _N.empty((2, ITER, Tm1))
    smp_Bns     = _N.empty((2, ITER, Tm1))
    smp_q2s     = _N.empty((ITER, 6))
    smp_F0s     = _N.empty((ITER, 6))

    o_w1        = _N.random.randn(Tm1)   #  start at 0 + u
    o_w2        = _N.random.randn(Tm1)   #  start at 0 + u

    B1wn        = _N.random.randn(Tm1)   #  start at 0 + u
    B2wn        = _N.random.randn(Tm1)   #  start at 0 + u

    q2_Bw1 = 1.
    q2_Bw2 = 1.
    F0_Bw1 = 0
    F0_Bw2 = 0

    off_sig2 = 0.4
    off_mu   = 0

    o_w1[:] = 0
    o_w2[:] = 0

    do_order = _N.arange(2)
    for it in range(ITER):
        if it % 1000 == 0:
            print("%(it)d   capped %(cp)d" % {"it" : it, "cp" : capped})

        vrncL1 = 1/_N.sum(ws1)   
        vrnc1  = (off_sig2*vrncL1) / (off_sig2 + vrncL1)
        vrncL2 = 1/_N.sum(ws2)
        vrnc2  = (off_sig2*vrncL2) / (off_sig2 + vrncL2)

        _N.random.shuffle(do_order)

        for di in do_order:
            #################
            if di == 0:
                o_w1, F0_Bw1, q2_Bw1 = sampleAR_and_offset(it, Tm1, vrnc1, vrncL1,
                                                           B1wn, o_w1, 
                                                           kappa[:, 0], ws1, q2_Bw1, a_F0, b_F0, a_q2,
                                                           B_q2, w1_px, w1_pV, w1_fx, w1_fV, w1_K, random_walk)
                smp_offsets[0, it] = o_w1[0]
            elif di == 1:
                #################
                o_w2, F0_Bw2, q2_Bw2 = sampleAR_and_offset(it, Tm1, vrnc2, vrncL2,
                                                           B2wn, o_w2, 
                                                           kappa[:, 1], ws2, q2_Bw2, a_F0, b_F0, a_q2,
                                                           B_q2, w2_px, w2_pV, w2_fx, w2_fV, w2_K, random_walk)
                smp_offsets[1, it] = o_w2[0]

        smp_Bns[0, it] = B1wn
        smp_Bns[1, it] = B2wn

        smp_q2s[it, 2*cond:2*cond+2]  = q2_Bw1, q2_Bw2
        #if random_walk:
        #    F0_Bw1 = F0_Bt1 = F0_Bl1 = F0_Bw2 = F0_Bt2 = F0_Bl2 = 1
        smp_F0s[it, 2*cond:2*cond+2]  = F0_Bw1, F0_Bw2

        lw.rpg_devroye(N_vec[:, 0], B1wn+o_w1, out=ws1)
        lw.rpg_devroye(N_vec[:, 1], B2wn+o_w2, out=ws2)

        ws2[zr2] = 1e-20#1e-20
        smp_Bns_all[cond, it, conds[cond]] = smp_Bns[0, it]
        smp_Bns_all[cond+3, it, conds[cond]] = smp_Bns[1, it]
        smp_offsets_all[cond, it] = smp_offsets[0, it, 0]
        smp_offsets_all[cond+3, it] = smp_offsets[1, it, 1]

    BnW1s    = smp_Bns[0, it0:it1:smp_every]
    BnW2s    = smp_Bns[1, it0:it1:smp_every]
    oW1s    = smp_offsets[0, it0:it1:smp_every, 0].reshape((it1-it0)//smp_every, 1)
    oW2s    = smp_offsets[1, it0:it1:smp_every, 0].reshape((it1-it0)//smp_every, 1)

    prob_mvs = _N.zeros((3, Tm1))
    prob_mvs_smps = _N.zeros((3, (it1-it0)//smp_every, Tm1))

    exp1 = _N.exp(BnW1s+oW1s)
    exp2 = _N.exp(BnW2s+oW2s)

    ix = -1
    for x in [_N.array([1, 0, 0]), _N.array([0, 1, 0]), _N.array([0, 0, 1])]:
        ix += 1
        if x[0] == 1:         #  x_vec=[1, 0, 0], N_vec=[1, 0, 0]  STAY
            trm1 = exp1 / (1 + exp1)
            trm2 = 1   #  1 / 1
            prob_mvs_smps[0] = trm1*trm2
        elif x[1] == 1:  #  x_vec=[0, 1, 0], N_vec=[1, 1, 0]       LOSER
            trm1 = 1 / (1 + exp1)
            trm2 = exp2 / (1 + exp2)
            prob_mvs_smps[1] = trm1 * trm2
        elif x[2] == 1:  #  x_vec=[0, 0, 1], N_vec=[1, 1, 1]       WINNER
            trm1 = 1 / (1 + exp1)
            trm2 = 1 / (1 + exp2)
            prob_mvs_smps[2] = trm1 * trm2


    prob_mvs      = _N.mean(prob_mvs_smps, axis=1)
    prob_mvs_srtd = _N.sort(prob_mvs_smps, axis=1)
    itL           = int(0.1*((it1-it0)//smp_every))
    itH           = int(0.9*((it1-it0)//smp_every))

    maxX = _N.max(_N.array([nWins, nTies, nLoss]))
    for i in range(3):
        if i == 0:  #  stay
            tTrans    = _N.where(y == 0)[0]
            tTransOth = _N.where((y == 1) | (y == -1))[0]
        elif i == 1:  #  dn
            tTrans    = _N.where(y == -1)[0]
            tTransOth = _N.where((y == 1) | (y == 0))[0]
        elif i == 2:  #  up
            tTrans    = _N.where(y == 1)[0]
            tTransOth = _N.where((y == -1) | (y == 0))[0]

        cond_probs[cond, i] = -100
        lower_conf[:] = -100
        upper_conf[:] = -100
        lower_conf[conds[cond]] = prob_mvs_srtd[i, itL]
        upper_conf[conds[cond]] = prob_mvs_srtd[i, itH]
        cond_probs[cond, i, conds[cond]] = prob_mvs[i]
        fill_unobserved(cond_probs[cond, i])
        fill_unobserved(lower_conf)
        fill_unobserved(upper_conf)
        ax = fig.add_subplot(9, 1, 3*cond+1+i)
        # _plt.fill_between(1+_N.arange(N_all), lower_conf, upper_conf, color="#FFAAAA")
        _plt.plot(1+_N.arange(Tm1), prob_mvs[i], color="black")
        _plt.scatter(tTrans+1, _N.ones(len(tTrans))*-0.01, marker=".", s=9, color="black")
        _plt.scatter(tTransOth+1, _N.ones(len(tTransOth))*-0.08, marker="|", s=5, color="grey")
        #_plt.plot(_N.arange(N_all), cond_probs[cond, i], color="black")
        _plt.ylim(-0.15, 1)
        _plt.xlim(0, maxX)
        if know_gt:
            _plt.plot(Ts[:, cond, i])
        #_plt.plot(_N.arange(Tm1+2), Ts[:, 0, i], lw=2, color="blue")

        # ax = _plt.subplot2grid((10, 9), (cond*3+i, 0), colspan=6)
        # #_plt.plot(prob_mvs[iw, itrans, tr0:tr1], color=colors[iw], lw=2)
        # clr = "black"#"#5555FF" if ((i == 0)) and ((cond == 0) or (cond == 2)) else "black"
        # _plt.plot(_N.arange(N_all), cond_probs[cond, i], color=clr, lw=2)
        # #_plt.plot(mednLatSt, color=clr, lw=2)

        # _plt.plot([0, N_all], [static_cnd_probs[cond, i], static_cnd_probs[cond, i]], color=clr, lw=1, ls=":")
        # _plt.fill_between(_N.arange(N_all), lower_conf, y2=upper_conf, color="#000000", alpha=0.1)
        # #_plt.plot(prob_fmvs[iw, i], color="red", lw=1)
        # if know_gt:
        #     _plt.plot(Ts[:, cond, i], color="blue",lw=2)
        # ax.set_ylabel("p(%(trs)s | %(wtl)s)" % {"wtl" : s_WTL_conds[cond], "trs" : s_trans[i]}, rotation=30, labelpad=25, fontsize=fnt_lbl)
        # _plt.scatter(off_cond_events[cond][i], _N.ones(len(off_cond_events[cond][i]))*-0.1, s=8, color="grey", marker="|")
        # _plt.scatter(cond_events[cond][i], _N.ones(len(cond_events[cond][i]))*-0.04, s=5, color="black", marker="o")
        # _plt.ylim(-0.15, 1)
        # _plt.xlim(0, 250)   #  for proposal
             
        #_plt.axhline(y=0.5, ls="--", color="grey")
        # for ix in _N.arange(50, tr1, 50):
        #     _plt.axvline(x=ix, ls=":", color="grey")
        if cond == 0:
            ax.set_facecolor("#BBFFBB")
        elif cond == 1:
            ax.set_facecolor("#FFFFBB")
        if cond == 2:
            ax.set_facecolor("#FFBBBB")
                  
        _plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=fnt_tck)
        #_plt.xticks([], fontsize=fnt_tck)
        fig.subplots_adjust(top=0.96, left=0.2, right=0.98, bottom=0.04, wspace=0.02)
        _plt.suptitle("%(df)s  %(sr)s" % {"df" : dat_fn, "sr" : sran})

        _plt.savefig("%(od)s/cond_probs_%(fns)s%(sr)s%(fl)s.png" % {"rpsm" : dat_fn, "fl" : fns[sig_cov], "lb" : label, "fns" : fns[sig_cov], "sr" : sran, "od" : out_dir, "fl" : s_flip})

        #_plt.xlim(tr0, tr1)


             # #######   autocorrelation of NGS(t) component
             # ax = _plt.subplot2grid((10, 9), (iw*3+itrans, 7), colspan=2)
             # #_plt.acorr(prob_mvs[iw, itrans, tr0:tr1] - _N.mean(prob_mvs[iw, itrans, tr0:tr1]), maxlags=30)

             # xs, ac = _eu.autocorrelate(mednLatSt, 30)
             # _plt.plot(xs, ac, color="black")
             # _plt.xticks([-30, -20, -10, 0, 10, 20, 30])
             # _plt.grid(ls=":")
             # #_plt.ylim(-0.15, 0.5)
             # _plt.ylim(-0.3, 0.5)

             # _plt.xticks([-30, -20, -10, 0, 10, 20, 30], ["", "", "", "", "", "", ""])


corrs1 = _N.zeros((9, 9))
corrs2 = _N.zeros((9, 9))
corrs12 = _N.zeros((9, 9))
for cond1 in range(3):
    for cond2 in range(cond1, 3):
        for itr1 in range(3):
            for itr2 in range(3):
                if ((itr1 == itr2) and (cond1 != cond2)) or (itr1 != itr2):
                     pc1, pv1 = _ss.pearsonr(cond_probs[cond1, itr1, 2:N_all//2], cond_probs[cond2, itr2, 2:N_all//2])
                     pc2, pv2 = _ss.pearsonr(cond_probs[cond1, itr1, N_all//2:], cond_probs[cond2, itr2, N_all//2:])
                     pc12, pv12 = _ss.pearsonr(cond_probs[cond1, itr1, 2:], cond_probs[cond2, itr2, 2:])
                     str1 = "p(%(trs)s | %(wtl)s)" % {"wtl" : s_WTL_conds[cond1], "trs" : s_trans[itr1]},
                     str2 = "p(%(trs)s | %(wtl)s)" % {"wtl" : s_WTL_conds[cond2], "trs" : s_trans[itr2]},
                     print("%(s1)s %(s2)s" % {"s1" : str1, "s2" : str2})
                     print("    %(pc).3fpv=%(pv).3f" % {"pc" : pc1, "pv" : pv1, "s1" : str1, "s2" : str2})
                     print("    %(pc).3fpv=%(pv).3f" % {"pc" : pc2, "pv" : pv2, "s1" : str1, "s2" : str2})
                     corrs1[cond1*3+itr1, cond2*3+itr2]  = pc1
                     corrs1[cond2*3+itr2, cond1*3+itr1]  = pc1
                     corrs2[cond1*3+itr1, cond2*3+itr2]  = pc2
                     corrs2[cond2*3+itr2, cond1*3+itr1]  = pc2
                     corrs12[cond1*3+itr1, cond2*3+itr2] = pc12
                     corrs12[cond2*3+itr2, cond1*3+itr1] = pc12

fig = _plt.figure(figsize=(5, 12))
_plt.suptitle(dat_fn)
fig.add_subplot(3, 1, 1)
_plt.imshow(corrs1, vmin=-1, vmax=1, cmap="seismic")
_plt.plot([-0.5, 8.5], [5.5, 5.5], color="black", lw=2)
_plt.plot([-0.5, 8.5], [2.5, 2.5], color="black", lw=2)
_plt.plot([5.5, 5.5], [-0.5, 8.5], color="black", lw=2)
_plt.plot([2.5, 2.5], [-0.5, 8.5], color="black", lw=2)
_plt.plot([-0.5, 8.5], [-0.5, 8.5], color="black", lw=2)
fig.add_subplot(3, 1, 2)
_plt.imshow(corrs2, vmin=-1, vmax=1, cmap="seismic")
_plt.plot([-0.5, 8.5], [5.5, 5.5], color="black", lw=2)
_plt.plot([-0.5, 8.5], [2.5, 2.5], color="black", lw=2)
_plt.plot([5.5, 5.5], [-0.5, 8.5], color="black", lw=2)
_plt.plot([2.5, 2.5], [-0.5, 8.5], color="black", lw=2)
_plt.plot([-0.5, 8.5], [-0.5, 8.5], color="black", lw=2)
fig.add_subplot(3, 1, 3)
_plt.imshow(corrs12, vmin=-1, vmax=1, cmap="seismic")
_plt.plot([-0.5, 8.5], [5.5, 5.5], color="black", lw=2)
_plt.plot([-0.5, 8.5], [2.5, 2.5], color="black", lw=2)
_plt.plot([5.5, 5.5], [-0.5, 8.5], color="black", lw=2)
_plt.plot([2.5, 2.5], [-0.5, 8.5], color="black", lw=2)
_plt.plot([-0.5, 8.5], [-0.5, 8.5], color="black", lw=2)

_plt.savefig("%(od)s/cond_probs_corrs_%(fns)s%(sr)s%(flp)s.png" % {"rpsm" : dat_fn, "fl" : fns[sig_cov], "lb" : label, "fns" : fns[sig_cov], "sr" : sran, "od" : out_dir, "flp" : s_flip})

pklme = {}
#smp_every =  50
for i in range(0, ITER//smp_every):
    for comp in range(6):
        fill_unobserved(smp_Bns_all[comp, i*smp_every])
pklme["smp_Bns"] = smp_Bns_all[:, ::smp_every]
pklme["smp_q2s"] = smp_q2s[::smp_every]
pklme["smp_F0s"] = smp_F0s[::smp_every]
pklme["smp_offsets"] = smp_offsets_all[:, ::smp_every]
pklme["smp_every"] = smp_every
pklme["hnd_dat"]   = hnd_dat
pklme["y_vec"]     = y_vec
pklme["N_vec"]     = N_vec
pklme["a_q2"]      = a_q2
pklme["B_q2"]      = B_q2
pklme["cond_probs"] = cond_probs
pklme["l_capped"]      = l_capped
pklme["separate"]  = True
pklme["corrs1"] = corrs1
pklme["corrs2"] = corrs2
pklme["corrs12"] = corrs12
pklme["flip"]   = s_flip
dmp = open("%(dir)s/%(rel)s,%(cov)s%(ran)s%(flp)s.dmp" % {"rel" : ssig, "cov" : scov, "ran" : sran, "dir" : out_dir, "flp" : s_flip}, "wb")
pickle.dump(pklme, dmp, -1)
dmp.close()
print("capped:  %d" % capped)

#dsig = _N.sum(_N.sum(_N.abs(_N.diff(cond_probs, axis=2)), axis=1), axis=0)
#_plt.acorr(dsig[10:] - _N.mean(dsig[10:]), maxlags=30)
