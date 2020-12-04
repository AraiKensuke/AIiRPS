#!/usr/bin/python

import numpy as _N  #  must import this before pyPG when running from shell
import pickle

import AIiRPS.utils.read_taisen as _rd
import matplotlib.pyplot as _plt
import os
import sys
import AIiRPS.models.labels as labels
from cmdlineargs import process_keyval_args
import multinomial_gibbs as _mg

from AIiRPS.utils.dir_util import getResultFN

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
_ME_WTL = 0
_RELATIVE_LAST_ME = 1    #  what to use as a comparison for when player switches

smp_every = 50
ITER = 15000
it0  = 14000
it1  = 15000

label=5
#  priors for inv gamma   #  B/(a+1)

rndmz = False

######  AR(1) coefficient range 
a_F0      = -1;    b_F0      =  1

######  int main 
#int i,pred,m,v[3],x[3*N+1],w[9*N+3],fw[3];

know_gt  = False
signal   = _RELATIVE_LAST_ME
covariates = _WTL
sig_cov   = _ME_WTL

tr0      = 0
tr1      = -1

#####USE real
#dat_fn        = "20Aug18-1624-01"

#dat_fn="20Jun01-0748-03"
#dat_fn="20May29-1923-44"
dat_fn="20May29-1419-14"
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
smp_offsets_all = _N.empty((6, ITER))
smp_Bns_all     = _N.ones((6, ITER, N_all))*-100

conds = []
conds.append(_N.where(hnd_dat[0:-1, 2] == 1)[0])   #  WIN
conds.append(_N.where(hnd_dat[0:-1, 2] == 0)[0])   #  TIE
conds.append(_N.where(hnd_dat[0:-1, 2] == -1)[0])  #  LOSE

out_dir = getResultFN("%(dfn)s" % {"dfn" : dat_fn})
if not os.access(out_dir, os.F_OK):
    os.mkdir(out_dir)
out_dir = getResultFN("%(dfn)s/%(lbl)d" % {"dfn" : dat_fn, "lbl" : label})
if not os.access(out_dir, os.F_OK):
    os.mkdir(out_dir)

col_n0 = 0    #  current
col_n1 = 0    #  previous

#  can't directly use hnd_dat, as it condX are discontinuous, 

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

s_WTL_conds = ["W", "T", "L"]
s_trans = ["stay", "wekr", "strgr"]

cond_probs = _N.empty((3, 3, hnd_dat.shape[0]-1))
upper_conf = _N.empty(hnd_dat.shape[0]-1)
lower_conf = _N.empty(hnd_dat.shape[0]-1)

fig = _plt.figure(figsize=(12, 7))

K     = 3
N     = 1   #  # of events observed, and distributed across K choices

y_all = _N.empty(len(conds[0]) + len(conds[1]) + len(conds[2]), dtype=_N.int)

for cond in range(3):
    thecond = conds[cond]
    Tm1     = len(thecond)

    y_vec = _N.zeros((Tm1, 3), dtype=_N.int)
    y     = 100*_N.ones(Tm1, dtype=_N.int)   #  indices of the random var

    ####   multinomial observation -> kappa (for Polya-Gamma)
    N_vec = _N.zeros((Tm1, K), dtype=_N.int)     #  The N vector
    kappa   = _N.empty((Tm1, K))

    for n in range(Tm1):   #  
        nc = thecond[n]
        if (hnd_dat[nc+1, col_n0] == hnd_dat[nc, col_n1]):
            y[n] = 0  #  Goo, choki, paa   goo->choki
            #   choki->paa
            y_vec[n, 0] = 1    #  [1, 0, 0]    stay
        elif ((hnd_dat[nc+1, col_n0] == 1) and (hnd_dat[nc, col_n1] == 3)) or \
             ((hnd_dat[nc+1, col_n0] == 2) and (hnd_dat[nc, col_n1] == 1)) or \
             ((hnd_dat[nc+1, col_n0] == 3) and (hnd_dat[nc, col_n1] == 2)):
             #  nc is paa, nc+1 is goo
            y[n] = -1
            y_vec[n, 1] = 1    #  [0, 1, 0]    choose weaker
        elif ((hnd_dat[nc+1, col_n0] == 1) and (hnd_dat[nc, col_n1] == 2)) or \
             ((hnd_dat[nc+1, col_n0] == 2) and (hnd_dat[nc, col_n1] == 3)) or \
             ((hnd_dat[nc+1, col_n0] == 3) and (hnd_dat[nc, col_n1] == 1)):
            y[n] = 1
            y_vec[n, 2] = 1    #  [0, 0, 1]    choose stronger

        N_vec[n, 0] = 1
        for k in range(1, K):
            N_vec[n, k] = N - _N.sum(y_vec[n, 0:k])
        for k in range(K):
            kappa[n, k] = y_vec[n, k] - 0.5 * N_vec[n, k]
    y_all[conds[cond]] = y

    smp_offsets = _N.empty((2, ITER))
    smp_Bns     = _N.empty((2, ITER, Tm1))
    smp_q2s     = _N.empty((ITER, 6))
    smp_F0s     = _N.empty((ITER, 6))

    #####################  THE GIBBS SAMPLING DONE HERE
    gb_sampler = _mg.multinomial_gibbs(N, K, N_vec, kappa, Tm1)
    gb_sampler.sample_posterior(ITER, a_F0, b_F0, a_q2, B_q2, smp_Bns, smp_offsets, smp_F0s[:, cond*2:cond*2+2], smp_q2s[:, cond*2:cond*2+2], random_walk=random_walk)

    #  weird behavior here.
    #a = _N.zeros((6, 500, 150))
    #ths = _N.arange(0, 150, 10)
    #a.shape   #  6, 500, 150
    #a[:, :, ths].shape   #  6, 500, 15
    #a[0, :, ths].shape   #  15, 500       #  WHY?

    smp_Bns_all[cond, :, conds[cond]] = smp_Bns[0].T
    smp_Bns_all[cond+3, :, conds[cond]] = smp_Bns[1].T
    smp_offsets_all[cond] = smp_offsets[0]
    smp_offsets_all[cond+3] = smp_offsets[1]

    ###  thin the samples out when saving.
    BnW1s    = smp_Bns[0, it0:it1:smp_every]
    BnW2s    = smp_Bns[1, it0:it1:smp_every]
    oW1s    = smp_offsets[0, it0:it1:smp_every].reshape((it1-it0)//smp_every, 1)
    oW2s    = smp_offsets[1, it0:it1:smp_every].reshape((it1-it0)//smp_every, 1)

    
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
        ####  stitch together
        fill_unobserved(cond_probs[cond, i])
        fill_unobserved(lower_conf)
        fill_unobserved(upper_conf)
        ax = fig.add_subplot(9, 1, 3*cond+1+i)
        #_plt.fill_between(1+_N.arange(N_all), lower_conf, upper_conf, color="#FFAAAA")
        _plt.plot(1+_N.arange(Tm1), prob_mvs[i], color="black")
        ####  plot raw data
        _plt.scatter(tTrans+1, _N.ones(len(tTrans))*-0.01, marker=".", s=9, color="black")
        _plt.scatter(tTransOth+1, _N.ones(len(tTransOth))*-0.08, marker="|", s=5, color="grey")
        #_plt.plot(_N.arange(N_all), cond_probs[cond, i], color="black")
        _plt.ylim(-0.15, 1)
        _plt.xlim(0, maxX)
        if know_gt:
            _plt.plot(Ts[:, cond, i])

        if cond == 0:
            ax.set_facecolor("#BBFFBB")
        elif cond == 1:
            ax.set_facecolor("#FFFFBB")
        if cond == 2:
            ax.set_facecolor("#FFBBBB")
                  
        _plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=fnt_tck)

        fig.subplots_adjust(top=0.96, left=0.2, right=0.98, bottom=0.04, wspace=0.02)
        _plt.suptitle("%(df)s  %(sr)s" % {"df" : dat_fn, "sr" : sran})

        _plt.savefig("%(od)s/cond_probs_%(fns)s%(sr)s%(fl)s.png" % {"rpsm" : dat_fn, "lb" : label, "fns" : fns[sig_cov], "sr" : sran, "od" : out_dir, "fl" : s_flip})

# fig = _plt.figure(figsize=(12, 7))
# for iWTL in range(3):
#    for iTrn in range(3):
#       fig.add_subplot(9, 1, 3*iWTL+iTrn+1)
#       if iTrn == 0:  #  stay
#         tTrans    = _N.where(y_all == 0)[0]
#         tTransOth = _N.where((y_all == 1) | (y_all == -1))[0]
#       elif iTrn == 1:  #  dn
#         tTrans    = _N.where(y_all == -1)[0]
#         tTransOth = _N.where((y_all == 1) | (y_all == 0))[0]
#       elif iTrn == 2:  #  up
#         tTrans    = _N.where(y_all == 1)[0]
#         tTransOth = _N.where((y_all == -1) | (y_all == 0))[0]

#       _plt.scatter(tTrans+1, _N.ones(len(tTrans))*-0.01, marker=".", s=9, color="black")
#       _plt.scatter(tTransOth+1, _N.ones(len(tTransOth))*-0.08, marker="|", s=5, color="grey")
#       _plt.ylim(-0.3, 0.1)


fig = _plt.figure(figsize=(12, 12))
for iWTL1 in range(3):
     for iWTL2 in range(3):
          for iTr1 in range(3):
               for iTr2 in range(3):
                    st1 = iWTL1*3+iTr1
                    st2 = iWTL2*3+iTr2
                    if (st2 > st1):
                         ax = fig.add_subplot(9, 9, st1*9 + st2 + 1)
                         _plt.scatter(cond_probs[iWTL1, iTr1, 2:], cond_probs[iWTL2, iTr2, 2:], s=1, color="black")
                         _plt.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
                    ax.set_aspect("equal")

"""
pklme = {}
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
#pklme["l_capped"]      = l_capped
pklme["separate"]  = True
pklme["corrs1"] = corrs1
pklme["corrs2"] = corrs2
pklme["corrs12"] = corrs12
pklme["flip"]   = s_flip
dmp = open("%(dir)s/%(rel)s,%(cov)s%(ran)s%(flp)s.dmp" % {"rel" : ssig, "cov" : scov, "ran" : sran, "dir" : out_dir, "flp" : s_flip}, "wb")
pickle.dump(pklme, dmp, -1)
dmp.close()
print("capped:  %d" % capped)

"""
