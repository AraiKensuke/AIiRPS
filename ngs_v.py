import numpy as _N
import matplotlib.pyplot as _plt

rndmz = False
sran  = "rndm_" if rndmz else ""
#datfn       = "20Jun01-0748-03"
#dat_fns       = ["20May29-1419-14"]#, 
#dat_fns       = "20May29-1923-44"

#datfn          = "20Nov21-1959-30"

#datfn          = "20Nov21-1921-05"
#
#
datfn="20Jan08-1703-13"
datfn="20Jan09-1504-32"
datfn="20Nov22-0025-50"
datfn       = "20Nov21-1921-05"
datfn="20Nov21-1959-30"
datfn="20Nov22-1108-25"
datfn="20Nov22-1243-40"
datfn="20Nov22-1407-17"
datfn="20Nov22-1655-02"
#dat_fns=("20Nov21-1959-30" "20Nov21-2131-38" "20Nov22-1108-25")
#datfn          = "20Nov21-2131-38"
#datfn          = "20Nov22-1407-18
#datfn="20Aug12-1252-50"
lab            = 5
_dA  = _N.loadtxt("Results/%(df)s/%(lb)d/cond_probs_ME,WTL.dat" % {"df" : datfn, "lb" : lab})

#_dA  = _N.loadtxt("Results/20Nov21-2131-38/2/cond_probs_ME,WTL.dat")
#_dA  = _N.loadtxt("Results/20Nov22-0025-50/2/cond_probs_ME,WTL.dat")

dA  = _dA.T.reshape(3, 3, _dA.shape[0])
#

fnt_lbl = 10
tck_lbl = 8
#  sb = 3, 6, 24
trns = ["stay", "wkr", "stgr"]
cond = ["WIN", "TIE", "LOS"]
prs = [[[0, 0], [0, 1]],
       [[0, 0], [0, 2]],
       [[0, 0], [1, 0]],
       [[0, 0], [1, 1]],
       [[0, 0], [1, 2]],
       [[0, 0], [2, 0]],
       [[0, 0], [2, 1]],
       [[0, 0], [2, 2]],
       [[0, 1], [0, 2]],  ##
       [[0, 1], [1, 0]], 
       [[0, 1], [1, 1]],
       [[0, 1], [1, 2]],
       [[0, 1], [2, 0]],
       [[0, 1], [2, 1]],
       [[0, 1], [2, 2]],
       [[0, 2], [1, 0]],  ##
       [[0, 2], [1, 1]],
       [[0, 2], [1, 2]],
       [[0, 2], [2, 0]],
       [[0, 2], [2, 1]],
       [[0, 2], [2, 2]],
       [[1, 0], [1, 1]],  ##
       [[1, 0], [1, 2]],
       [[1, 0], [2, 0]],
       [[1, 0], [2, 1]],
       [[1, 0], [2, 2]],
       [[1, 1], [1, 2]],  ## 
       [[1, 0], [2, 0]],
       [[1, 0], [2, 1]],
       [[1, 0], [2, 2]],
       [[1, 2], [2, 0]],  ##
       [[1, 0], [2, 1]],
       [[1, 0], [2, 2]],
       [[2, 0], [2, 1]],  ##
       [[2, 0], [2, 2]],
       [[2, 1], [2, 2]]]  ##

ip = 0

#  36 prs

fig = _plt.figure(figsize=(11, 11))
ip  = -1

for pr in prs:
    ip += 1

    cnd1, tr1 = pr[0][0], pr[0][1]
    cnd2, tr2 = pr[1][0], pr[1][1]

    x_axis = "p(%(tr)s | %(cd)s)" % {"tr" : trns[tr1], "cd" : cond[cnd1]}
    y_axis = "p(%(tr)s | %(cd)s)" % {"tr" : trns[tr2], "cd" : cond[cnd2]}

    ax = fig.add_subplot(6, 6, ip+1)
    if (ip+1 == 3) or (ip+1 == 6) or (ip+1 == 24):  #  WSLS, WSTS, TSLS
        ax.set_facecolor("#DDDDFF")
    if (ip+1 == 15) or (ip+1 == 20):
        ax.set_facecolor("#FFDDDD")
        if ip + 1 == 15:
            y_axis = "cp AI aft LOS"
            x_axis = "cp AI aft WIN"
    if cnd1 == cnd2:   ##  same condition compare
        ax.set_facecolor("yellow")

    ax.set_aspect("equal")
    dat_y = _N.array(dA[cnd2, tr2])
    if rndmz:
        _N.random.shuffle(dat_y)
    _plt.scatter(dA[cnd1, tr1], dat_y, s=1, color="black")
    _plt.text(0.1, 0.8, "round 1", color="black", fontsize=tck_lbl)
    _plt.xlim(0, 1)
    _plt.ylim(0, 1)
    _plt.xlabel(x_axis, fontsize=fnt_lbl)
    _plt.ylabel(y_axis, fontsize=fnt_lbl)
    _plt.yticks([])
    _plt.xticks([])
    _plt.plot([0, 1], [0, 1], ls=":", color="grey")

fig.subplots_adjust(wspace=0.0, hspace=0.4)

#_plt.savefig("NGS_pat_%(xy)s%(rn)s" % {"xy" : sxsy, "rn" : sran})
#_plt.close()
_plt.suptitle("yellow:  same condition, red:  copy, vs AI,  blue:  WSLS, WSTS, TSLS  %s" % datfn)
_plt.savefig("Results/%(df)s/%(lb)d/NGS_%(rn)sME,WTL.png" % {"df" : datfn, "lb" : lab, "rn" : sran})
