from AIiRPS.models import empirical
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
datetms = ["20211102_0025-00"]

pcs_123 = _N.zeros((len(datetms), 3))
id = -1

for datetm in datetms[0:1]:
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

        weights, iw = sim_prc.recreate_percep_istate(td, ini_percep, fin_percep)

    fig = _plt.figure()
    dsy = _N.zeros((weights.shape[1], 3, 3))
    for i in range(3):
        for j in range(3):
            dy = _N.diff(weights[iw, :, i, j], axis=1)
            #_plt.plot(dy)
            dsy[:, i, j] = dy[:, 0]
