import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
import AIiRPS.models.empirical as _emp
import AIiRPS.utils.read_taisen as _rt
from AIiRPS.utils.dir_util import getResultFN
import os
import pickle

#dats=["20Aug18-1624-01"]#, "20Aug18-1644-09", 
#dats=["20Apr24-1650-24"]
dats=["20Aug18-1644-09"]
#dats=["20Apr24-1650-24"]
#dats=["20Jan09-1504-32"]#, "20Aug12-1252-50", "20Jan08-1703-13"]
#dats = ["20May29-1419-14", "20Jun01-0748-03", "20May29-1923-44"]
#dats = ["20Nov22-1108-25"]#, "20Aug18-1624-01", "20Jan09-1504-32"]
#dats = ["20Nov21-2131-38"]
#dats = ["20Nov21-1959-30", "20Nov21-2131-38"]
#dats = ["20Nov21-2131-38"]

tops = []

expt_scores = []

di = 0

label=100
win = 40
SHUF = 0
scov = "WTL"
ssig = "ME"
sran = ""
all_cprobs = []

for dat in dats:
    hnd_dat    = _rt.return_hnd_dat(dat)
    out_dir = getResultFN("%(dfn)s" % {"dfn" : dat})
    if not os.access(out_dir, os.F_OK):
        os.mkdir(out_dir)
    out_dir = getResultFN("%(dfn)s/%(lbl)d" % {"dfn" : dat, "lbl" : label})
    if not os.access(out_dir, os.F_OK):
        os.mkdir(out_dir)


    #fig = _plt.figure(figsize=(13, 9))
    di += 1
    
    cond_probs  = _emp.kernel_NGS(dat, kerwin=2)

    pklme = {}
    pklme["hnd_dat"]   = hnd_dat
    pklme["cond_probs"] = cond_probs
    pklme["separate"]  = False
    pklme["flip"]   = False
    #dmp = open("%(dir)s/%(rel)s,%(cov)s%(ran)s%(flp)s.dmp" % {"rel" : ssig, "cov" : scov, "ran" : sran, "dir" : out_dir, "flp" : s_flip}, "wb")
    dmp = open("%(dir)s/%(rel)s,%(cov)s%(ran)s.dmp" % {"rel" : ssig, "cov" : scov, "ran" : sran, "dir" : out_dir}, "wb")
    pickle.dump(pklme, dmp, -1)
    dmp.close()
