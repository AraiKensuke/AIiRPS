import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
import AIiRPS.models.empirical as _emp

#dats=["20Aug18-1624-01"]#, "20Aug18-1644-09", "20Apr24-1650-24"]
#dats=["20Apr24-1650-24"]
dats=["20Jan09-1504-32"]#, "20Aug12-1252-50", "20Jan08-1703-13"]
#dats = ["20May29-1419-14", "20Jun01-0748-03", "20May29-1923-44"]
#dats = ["20Nov22-1108-25"]#, "20Aug18-1624-01", "20Jan09-1504-32"]
#dats = ["20Nov21-2131-38"]
#dats = ["20Nov21-1959-30", "20Nov21-2131-38"]
#dats = ["20Nov21-2131-38"]

tops = []

expt_scores = []

di = 0

label="E1"
win = 40
SHUF = 100
all_cprobs = []

for dat in dats:
    fig = _plt.figure(figsize=(13, 9))
    di += 1
    
    cprobs, Tgame = _emp.empirical_NGS(dat, win=win, SHUF=SHUF)
    for ch in range(9):
        fig.add_subplot(9, 1, ch+1)

        srtd = _N.sort(cprobs[1:, ch], axis=0)
        lo   = srtd[int(SHUF*0.05)]
        hi   = srtd[int(SHUF*0.95)]
        _plt.fill_between(_N.arange(Tgame-win), lo, hi, color="#CCCCFF")
        _plt.plot(_N.arange(Tgame-win), cprobs[0, ch], color="black")
        _plt.ylim(-0.1, 1.1)
    
    all_cprobs.append(cprobs)

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
pklme["separate"]  = True
pklme["flip"]   = s_flip
dmp = open("%(dir)s/%(rel)s,%(cov)s%(ran)s%(flp)s.dmp" % {"rel" : ssig, "cov" : scov, "ran" : sran, "dir" : out_dir, "flp" : s_flip}, "wb")
pickle.dump(pklme, dmp, -1)
dmp.close()
