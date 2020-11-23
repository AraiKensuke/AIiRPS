#!/usr/bin/python

"""
Look at 3signals.py to play with how 3 signals that must add to 1 can 
be correlated.
"""

import eeg_util as _eu
import scipy.stats as _ss
from scipy.signal import savgol_filter
import numpy as _N  #  must import this before pyPG when running from shell
import pyPG as lw
import LOSTtmp.kfARlib1c as _kfar
import pickle
import rpsms


import read_taisen as _rt
import matplotlib.pyplot as _plt
import os
import sys
from cmdlineargs import process_keyval_args

def union_arrs(a1, a2):
     return _N.sort(_N.array(a1.tolist() + a2.tolist()))

#  input data
#  m_hands    # machine hands  size N         1 -1  1
#  h_hands    # human hands    size N        -1 -1  1
#  -1 -1  1  1 -1  1 
#      0  1  0  1  1

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

fns     = ["ME,WTL", "ME,RPS", "AI,WTL", "AI,RPS", "OU,WTL", "OU,RPS"]

fnt_tck = 10
fnt_lbl = 12

_ME_WTL = 0
_ME_RPS = 1
_AI_WTL = 2
_AI_RPS = 3
_OU_WTL = 4
_OU_RPS = 5

_W      = 0
_T      = 1
_L      = 2
_R      = 0
_P      = 1
_S      = 2

ITER = 30000

label=2
#  priors for inv gamma   #  B/(a+1)
#a_q2 = 100
a_q2 = 2.
#a_q2 = 10.
#a_q2 = 1.01
#a_q2 = 50.
#B_q2 = 200.
#B_q2= 80.
#B_q2= 30
B_q2= 0.1
#B_q2 = 30

rndmz = False

know_gt  = False
#signal   = _RELATIVE_LAST_ME
#covariates = _WTL
sig_cov   = _ME_WTL

#dat_fns       = ["20Apr24-1650-24", "20Aug18-1624-01", "20Aug18-1644-09", "20Jan08-1703-13", "20Jan09-1504-32", "20Aug12-1252-50"]
#dat_fns       = ["20May29-1923-44"]#, 
#dat_fns       = ["20May29-1419-14"]#, 
#dat_fns       = ["20Jun01-0748-03"]
#
#
#dat_fns        = ["20Aug12-1331-06"]
dat_fns        = ["20Nov09-1113-44"]

#dat_fns        = ["20Aug18-1624-01"]
#
#dat_fns       = ["22Jan01-0000-01"]
#dat_fns       = ["20Aug18-1624-01", "20Aug12-1252-50", "20Aug18-1644-09", "20Jan08-1703-13", "20Jan09-1504-32", "20Apr24-1650-24"]
#dat_fns       = ["20Apr24-1650-24"]
#
#dat_fns       = ["20Jan08-1703-13"]
#dat_fns       = ["20Oct26-0120-42"]
#dat_fns       = ["20Oct26-0136-29"]
#dat_fns       = ["20Oct26-0208-25"]

dat_fns=["20Nov09-1623-42"]
dat_fns=["20Nov09-1627-30"]
# dat_fns=["20Nov09-1634-17"]
# dat_fns=["20Nov09-1647-51"]
# dat_fns=["20Nov09-1652-59"]
# dat_fns=["20Nov09-1701-56"]
# dat_fns=["20Nov09-1708-32"]
# dat_fns=["20Nov09-1714-13"]
dat_fns=["20Nov09-1728-37"]
# dat_fns=["20Nov09-1750-02"]
# dat_fns=["20Nov09-1756-23"]
# dat_fns=["20Nov09-1801-22"]
# dat_fns=["20Nov09-1810-15"]
# dat_fns=["20Nov09-1826-22"]
# dat_fns=["20Nov09-1830-54"]
# dat_fns=["20Nov09-1835-20"]
dat_fns=["20Nov09-2000-42"]
dat_fns=["20Nov09-2011-15"]
dat_fns=["20Nov09-2017-40"]
dat_fns=["20Nov09-2020-37"]
dat_fns=["20Nov10-0519-06"]
dat_fns=["20Nov10-0525-06"]
dat_fns=["20Nov10-0533-07"]
dat_fns=["20Nov10-0537-37"]
dat_fns=["20Nov10-0543-02"]
dat_fns=["20Nov10-0547-10"]
dat_fns=["20Nov10-0647-19"]
dat_fns=["20Nov10-0709-33"]
dat_fns=["20Nov10-0802-30"]
dat_fns=["20Nov10-0916-21"]
#dat_fns=["20Nov10-0741-43"]
dat_fns=["20Nov10-0958-25"]
dat_fns=["20Nov10-1011-30"]
dat_fns=["20Nov10-2157-31"]




dat_fns=["20Nov11-0740-13"]
dat_fns=["20Nov11-0900-59"]
dat_fns       = ["20Aug18-1644-09"]
dat_fns=["20Aug18-1546-13"]
dat_fns=["20Jan08-1703-13"]
dat_fns=["20Aug18-1624-01"]
#dat_fns        = ["20Aug12-1252-50"]
#dat_fns       = ["20Apr24-1650-24"]
#dat_fns       = ["20Jan09-1504-32"]
#dat_fns = ["20Nov08-0904-26"]
dat_fns       = ["20May29-1419-14"]#, 
dat_fns       = ["20Jun01-0748-03"]


dat_fns=["20Aug12-1331-06"]
dat_fns=["20Apr18-2148-58"]
dat_fns=["20May04-2219-50"]
dat_fns = ["20Nov08-0904-26"]

dat_fns       = ["20Jun01-0748-03"]
dat_fns       = ["20Nov21-1921-05"]


dat_fns       = ["20Nov22-0025-50"]

dat_fns       = ["20Nov21-2131-38"]
dat_fns       = ["20Nov21-1959-30"]
dat_fns       = ["20Jan09-1504-32"]#, 
dat_fns       = ["20Nov21-1921-05"]
#dat_fns       = ["20May29-1419-14"]#, 
#dat_fns       = ["20May29-1923-44"]
#dat_fns       = ["20Jan09-1504-32"]
dat_fns=["20Nov21-1959-30", "20Nov21-2131-38", "20Nov22-1108-25"]
#dat_fns=["20May29-1419-14", "20Jun01-0748-03", "20May29-1923-44"]
#########################################

for dat_fn in dat_fns:
     print(dat_fn)
     sran = ""
     if rndmz:
         sran = "rndmz"
     if know_gt:
         lmGT = depickle("simDAT/rpsm_%s.dmp" % dat_fn)
         Ts = lmGT["Ts_timeseries"]



     #scov = "WTL" if covariates == _WTL else "RPS"
     #ssig = "ME"  if signal == _RELATIVE_LAST_ME else "AI"

     #print("%(dat)s,%(rel)s,%(cov)s%(ran)s.dmp" % {"dat" : dat_fn, "rel" : ssig, "cov" : scov, "ran" : sran})
     lm       = depickle("Results/%(rpsm)s/%(lb)d/%(fl)s%(rnd)s.dmp" % {"rpsm" : dat_fn, "fl" : fns[sig_cov], "lb" : label, "rnd" : sran})
     #lm = depickle("%(dat)s,%(rel)s,%(cov)s%(ran)s.dmp" % {"dat" : dat_fn, "rel" : ssig, "cov" : scov, "ran" : sran})

     y_vec   = lm["y_vec"]
     N_vec   = lm["N_vec"]
     smp_Bns = lm["smp_Bns"]
     smp_evry= lm["smp_every"]
     smp_offsets= lm["smp_offsets"]
     Tm1     = y_vec.shape[0] - 1

     itr0    = 29000//smp_evry
     itr1    = 30000//smp_evry

     BnW1    = smp_Bns[0, itr0:itr1].T
     BnT1    = smp_Bns[1, itr0:itr1].T
     BnL1    = smp_Bns[2, itr0:itr1].T
     BnW2    = smp_Bns[3, itr0:itr1].T
     BnT2    = smp_Bns[4, itr0:itr1].T
     BnL2    = smp_Bns[5, itr0:itr1].T

     tr0     = 0
     tr1     = BnW1.shape[0]-1
     print(tr1)

     oW1     = smp_offsets[0, itr0:itr1, 0].T  #  itr1-itr0 x ndatlength
     oT1     = smp_offsets[1, itr0:itr1, 0].T
     oL1     = smp_offsets[2, itr0:itr1, 0].T
     oW2     = smp_offsets[3, itr0:itr1, 0].T
     oT2     = smp_offsets[4, itr0:itr1, 0].T
     oL2     = smp_offsets[5, itr0:itr1, 0].T

     hnd_dat = lm["hnd_dat"]

     if sig_cov == _ME_WTL:
          stay_win, strg_win, wekr_win, stay_tie, wekr_tie, strg_tie, stay_los, wekr_los, strg_los, win_cond, tie_cond, los_cond = _rt.get_ME_WTL(hnd_dat, tr0, tr1)
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


     # p(x | history)
     prob_mvs  = _N.zeros((3, 3, Tm1, itr1-itr0))
     prob_fmvs = _N.zeros((3, 3, Tm1))


     savgol_win = 7
     WTL_conds = _N.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]])

     ic = 0
     #for WTL_cond in [[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]:
     iw = -1

     colors = ["black", "orange", "blue"]
     s_WTL_conds = ["W", "T", "L"] if sig_cov == _ME_WTL else ["R", "P", "S"]
     if sig_cov == _OU_WTL:
          s_trans = ["W", "T", "L"]
     elif (sig_cov == _ME_WTL) or (sig_cov == _ME_RPS):
          s_trans = ["stay", "wekr", "strgr"]
     #fig = _plt.figure(figsize=(9.5, 12))
     fig = _plt.figure(figsize=(6, 12))
     for WTL_cond in WTL_conds:
         iw += 1   #  
         #fig = _plt.figure(figsize=(11, 11))
         ic += 1   #  category

         #_plt.title("conditioned on last WTL=%d" % (ic-1))
         W = WTL_cond[0]
         T = WTL_cond[1]
         L = WTL_cond[2]
         for n in range(Tm1):
             #  for each of the 6
              exp1 = _N.exp((BnW1[n]+oW1)*W + (BnT1[n] + oT1)*T + (BnL1[n] + oL1)*L)
              exp2 = _N.exp((BnW2[n]+oW2)*W + (BnT2[n] + oT2)*T + (BnL2[n] + oL2)*L)
              ix = -1
              for x in [_N.array([1, 0, 0]), _N.array([0, 1, 0]), _N.array([0, 0, 1])]:
                   ix += 1
                   if x[0] == 1:         #  x_vec=[1, 0, 0], N_vec=[1, 0, 0]  STAY
                        trm1 = exp1 / (1 + exp1)
                        trm2 = 1   #  1 / 1
                        prob_mvs[iw, 0, n] = trm1*trm2
                   elif x[1] == 1:  #  x_vec=[0, 1, 0], N_vec=[1, 1, 0]       LOSER
                        trm1 = 1 / (1 + exp1)
                        trm2 = exp2 / (1 + exp2)
                        prob_mvs[iw, 1, n] = trm1 * trm2
                   elif x[2] == 1:  #  x_vec=[0, 0, 1], N_vec=[1, 1, 1]       WINNER
                        trm1 = 1 / (1 + exp1)
                        trm2 = 1 / (1 + exp2)
                        prob_mvs[iw, 2, n] = trm1 * trm2

         #_N.sum(Ts[n, 0]) = 1   #  stay
         for itrans in range(3):   
             #ax = fig.add_subplot(10, 1, iw*3+itrans+1)
             ax = _plt.subplot2grid((10, 9), (iw*3+itrans, 0), colspan=6)
             #_plt.plot(prob_mvs[iw, itrans, tr0:tr1], color=colors[iw], lw=2)
             clr = "#5555FF" if ((itrans == 0)) and ((iw == 0) or (iw == 2)) else "black"
             mednLatSt = _N.median(prob_mvs[iw, itrans, tr0:tr1], axis=1)
             srtdLatSts = _N.sort(prob_mvs[iw, itrans, tr0:tr1], axis=1)
             lowLatSt= srtdLatSts[:, int(0.1*(itr1-itr0))]
             hiLatSt = srtdLatSts[:, int(0.9*(itr1-itr0))]
             _plt.plot(mednLatSt, color=clr, lw=2)

             _plt.plot([tr0, tr1], [static_cnd_probs[iw, itrans], static_cnd_probs[iw, itrans]], color=clr, lw=1, ls=":")
             _plt.fill_between(_N.arange(tr0, tr1), lowLatSt, y2=hiLatSt, color="#000000", alpha=0.1)
             #_plt.plot(prob_fmvs[iw, i], color="red", lw=1)
             if know_gt:
                    _plt.plot(Ts[tr0:tr1, iw, itrans], color="blue",lw=2)
             ax.set_ylabel("p(%(trs)s | %(wtl)s)" % {"wtl" : s_WTL_conds[iw], "trs" : s_trans[itrans]}, rotation=30, labelpad=25, fontsize=fnt_lbl)
             #ylb.set_rotation(0)
             _plt.scatter(off_cond_events[iw][itrans], _N.ones(len(off_cond_events[iw][itrans]))*-0.1, s=8, color="grey", marker="|")
             _plt.scatter(cond_events[iw][itrans], _N.ones(len(cond_events[iw][itrans]))*-0.04, s=5, color="black", marker="o")
             _plt.ylim(-0.15, 1)
             
             #_plt.axhline(y=0.5, ls="--", color="grey")
             for ix in _N.arange(50, tr1, 50):
                  _plt.axvline(x=ix, ls=":", color="grey")
             if iw == 0:
                  ax.set_facecolor("#BBFFBB")
             elif iw == 1:
                  ax.set_facecolor("#FFFFBB")
             if iw == 2:
                  ax.set_facecolor("#FFBBBB")
                  
             _plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=fnt_tck)
             _plt.xticks([], fontsize=fnt_tck)
             #_plt.xlim(tr0, tr1)
             _plt.xlim(tr0, tr0+200)   #  for proposal

             #######   autocorrelation of NGS(t) component
             ax = _plt.subplot2grid((10, 9), (iw*3+itrans, 7), colspan=2)
             #_plt.acorr(prob_mvs[iw, itrans, tr0:tr1] - _N.mean(prob_mvs[iw, itrans, tr0:tr1]), maxlags=30)

             xs, ac = _eu.autocorrelate(mednLatSt, 30)
             _plt.plot(xs, ac, color="black")
             _plt.xticks([-30, -20, -10, 0, 10, 20, 30])
             _plt.grid(ls=":")
             #_plt.ylim(-0.15, 0.5)
             _plt.ylim(-0.3, 0.5)

             _plt.xticks([-30, -20, -10, 0, 10, 20, 30], ["", "", "", "", "", "", ""])


     ##################################   dNGS
     all_meds = _N.median(prob_mvs, axis=3)


     for iw in range(3):
          for ix in range(3):
               #prob_fmvs[iw, ix] = savgol_filter(_N.arctanh(2*(all_meds[iw, ix]-0.5)), savgol_win, 3)
               #prob_fmvs[iw, ix] = savgol_filter(all_meds[iw, ix], savgol_win, 3)
               #prob_fmvs[iw, ix] = _N.arctanh(2*(all_meds[iw, ix]-0.5))
               prob_fmvs[iw, ix] = all_meds[iw, ix]

     chg_sgnl = _N.sum(_N.sum(_N.abs(_N.diff(prob_fmvs, axis=2)), axis=1), axis=0)/9
     ax = _plt.subplot2grid((10, 9), (9, 0), colspan=6)
     _plt.plot(chg_sgnl, color="black", lw=2)
     _plt.ylabel("dNGS($t$)", rotation=30, labelpad=25)
     #_plt.ylim(0, 0.1)
     #_plt.xlim(tr0, tr1)
     _plt.xlim(tr0, tr0+200)   #  for proposal
     _plt.xticks([])
     for ix in _N.arange(50, tr1, 50):
          _plt.axvline(x=ix, ls=":", color="grey")

     xs, ac = _eu.autocorrelate(chg_sgnl, 30)

     ax = _plt.subplot2grid((10, 9), (9, 7), colspan=2)
     _plt.plot(xs, ac, color="black")
     _plt.xticks([-30, -20, -10, 0, 10, 20, 30], ["", "", "", "", "", "", ""])

     _plt.ylim(-0.3, 0.5)
     row123 = all_meds.reshape((9, prob_fmvs.shape[2]))
     _N.savetxt("Results/%(rpsm)s/%(lb)d/cond_probs_%(fns)s%(sr)s.dat" % {"rpsm" : dat_fn, "fl" : fns[sig_cov], "lb" : label, "fns" : fns[sig_cov], "sr" : sran}, row123.T, fmt=("%.3f " * 9))


     ##################################   clustered states
     #row123 = all_meds[:, :, tr0:tr1].reshape((9, tr1-tr0))
     #row123 = prob_fmvs[:, :, tr0:tr1].reshape((9, tr1-tr0))
     #row123 = prob_fmvs.reshape((9, prob_fmvs.shape[2]))
     #print(row123)


     """
     #nStates123, labs123, pca_prj = _eu.find_GMM_labels(row123.T, try_K=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], TRs=[1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80], do_pca=True, min_var_expld=0.99)
     nStates123, labs123 = _eu.find_GMM_labels(row123.T, try_K=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], TRs=[1, 2, 4, 8, 16, 32, 40, 50, 60, 70, 80], do_pca=False, min_var_expld=0.99)

     ax = _plt.subplot2grid((10, 9), (9, 0), colspan=6)
     _plt.scatter(_N.arange((tr1-tr0)), labs123[tr0:tr1], s=3)
     _plt.ylim(-0.5, nStates123 - 1 + 0.5)
     #_plt.xlim(tr0, tr1)
     _plt.xlim(tr0, tr0+200)   #  for proposal
     for ix in _N.arange(50, tr0+200, 50):
          _plt.axvline(x=ix, ls=":", color="grey")

     _plt.xticks(_N.arange(0, tr0+200+1, 50), fontsize=fnt_tck)
     _plt.yticks(_N.arange(0, nStates123), _N.arange(0, nStates123)+1, fontsize=fnt_tck)
     _plt.xlabel("game #", fontsize=fnt_lbl)
     _plt.ylabel("strategy #", rotation=30, labelpad=25, fontsize=fnt_lbl)
     """
     _plt.suptitle("%(df)s  %(sr)s" % {"df" : dat_fn, "sr" : sran})
     #fig.subplots_adjust(top=0.96, left=0.13, right=0.98, bottom=0.04, wspace=0.33)
     fig.subplots_adjust(top=0.96, left=0.2, right=0.98, bottom=0.04, wspace=0.02)

     _plt.savefig("Results/%(rpsm)s/%(lb)d/cond_probs_%(fns)s%(sr)s.png" % {"rpsm" : dat_fn, "fl" : fns[sig_cov], "lb" : label, "fns" : fns[sig_cov], "sr" : sran})
     """
     _N.savetxt("Results/%(rpsm)s/%(lb)d/row123_%(fns)s%(sr)s.dat" % {"rpsm" : dat_fn, "fl" : fns[sig_cov], "lb" : label, "fns" : fns[sig_cov], "sr" : sran}, row123.T, fmt=("%.4f " * 9))

     row123_01 = 0.5*(_N.tanh(row123)+1)
     fig = _plt.figure(figsize=(9, 14))
     
     for ns in range(nStates123):
          _plt.subplot2grid((nStates123, 2), (ns, 0))
          sts = _N.where(labs123 == ns)[0]

          for st in sts:
               _plt.plot(row123_01[:, st], color="grey")
          _plt.plot(_N.mean(row123_01[:, sts], axis=1), color="black", marker=".", ms=10)
          _plt.axvline(x=2.5, ls=":")
          _plt.axvline(x=5.5, ls=":")
          _plt.ylim(0, 1)
          _plt.xlim(-0.5, 8.5)
          _plt.fill_between([-0.5, 2.5], [0, 0], [1, 1], color="#BBFFBB", alpha=0.5)
          _plt.fill_between([2.5,  5.5], [0, 0], [1, 1], color="#FFFFBB", alpha=0.5)
          _plt.fill_between([5.5,  8.5], [0, 0], [1, 1], color="#FFBBBB", alpha=0.5)


     _plt.subplot2grid((nStates123, 2), (0, 1), rowspan=(nStates123//2))
     for ns in range(nStates123):
          sts = _N.where(labs123 == ns)[0]

          for st in sts:
               _plt.plot(row123_01[:, st], color="grey", lw=0.5)
     _plt.ylim(0, 1)
     _plt.xlim(-0.5, 8.5)
     _plt.fill_between([-0.5, 2.5], [0, 0], [1, 1], color="#BBFFBB", alpha=0.5)
     _plt.fill_between([2.5,  5.5], [0, 0], [1, 1], color="#FFFFBB", alpha=0.5)
     _plt.fill_between([5.5,  8.5], [0, 0], [1, 1], color="#FFBBBB", alpha=0.5)

     _plt.savefig("Results/%(rpsm)s/%(lb)d/cond_probs_%(fns)s%(sr)s_collapsed.png" % {"rpsm" : dat_fn, "fl" : fns[sig_cov], "lb" : label, "fns" : fns[sig_cov], "sr" : sran})
     """
