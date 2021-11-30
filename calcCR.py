import AIiRPS.models.empirical_ken as empirical
from filter import gauKer
import numpy as _N
import matplotlib.pyplot as _plt
import GCoh.eeg_util as _eu
import AIiRPS.utils.read_taisen as _rt
from AIiRPS.utils.dir_util import getResultFN
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

wins= 4
gk_w = 3

visit = 1

#  These are ParticipantIDs.

expt="TMB2"

datetms = []
#fp = open("%(f)sfns_%(v)d.txt" % {"f" : expt, "v" : visit}, "r")
fp = open("%sfns.txt" % expt)
contents = fp.readlines()
for fn in contents:
    datetms.append(fn.rstrip())
fp.close()

pcs_123 = _N.zeros((len(datetms), 3))
id = -1

for datetm in datetms:
    id += 1

    print("...............   %s" % datetm)
    flip_human_AI = False

    #fig = _plt.figure(figsize=(11, 11))

    for cov in [_AIconst._WTL]:#, _AIconst._HUMRPS, _AIconst._AIRPS]:
        scov = _AIconst.sCOV[cov]
        sran = ""

        SHUFFLES = 1
        a_s = _N.zeros((len(datetms), SHUFFLES+1))
        acs = _N.zeros((len(datetms), SHUFFLES+1, 61))

        if gk_w > 0:
            gk = gauKer(gk_w)
            gk /= _N.sum(gk)
        sFlip = "_flip" if flip_human_AI else ""

        label="%(wins)d%(gkw)d" % {"wins" : wins, "gkw" : gk_w}
        
        out_dir = getResultFN("%(dfn)s" % {"dfn" : datetm})

        if not os.access(out_dir, os.F_OK):
            os.mkdir(out_dir)
        out_dir = getResultFN("%(dfn)s/%(lbl)s" % {"dfn" : datetm, "lbl" : label})
        if not os.access(out_dir, os.F_OK):
            os.mkdir(out_dir)

        td, start_time, end_time, UA, cnstr, inp_meth, ini_percep, fin_percep = _rt.return_hnd_dat(datetm, has_useragent=True, has_start_and_end_times=True, has_constructor=True, expt=expt, visit=visit)

        weights, preds, iw = sim_prc.recreate_percep_istate(td, ini_percep, fin_percep)
        
        ngs, ngsRPS, ngsDSURPS, ngsSTSW, all_tds, TGames  = empirical.empirical_NGS(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, covariates=cov, expt=expt, visit=visit)

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
            for sh in range(SHUFFLES+1):
                for i in range(6):
                    if gk_w > 0:
                        fNGSSTSW[sh, i] = _N.convolve(ngsSTSW[sh, i], gk, mode="same")
                    else:
                        fNGSSTSW[sh, i] = ngsSTSW[sh, i]

            pklme = {}
            pklme["cond_probs"] = fNGS
            pklme["cond_probsRPS"] = fNGSRPS            
            pklme["cond_probsSTSW"] = fNGSSTSW
            pklme["cond_probsDSURPS"] = fNGSDSURPS
            pklme["all_tds"] = all_tds
                
            pklme["start_time"] = start_time
            pklme["end_time"] = end_time
            pklme["AI_weights"] = weights[iw]
            pklme["AI_preds"] = preds[iw]

            dmp = open("%(dir)s/%(cov)s_%(visit)d.dmp" % {"cov" : scov, "dir" : out_dir, "visit" : visit}, "wb")
            pickle.dump(pklme, dmp, -1)
            dmp.close()









        
