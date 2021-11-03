import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

lm = depickle("AQ28_vs_RPS.dmp")

margCRs      = lm["marginalCRs"]
netwins      = lm["AQ28_netwins"][:, 1]

win_aft_win  = lm["AQ28_win_aft_win"][:, 1]
tie_aft_win  = lm["AQ28_tie_aft_win"][:, 1]
los_aft_win  = lm["AQ28_los_aft_win"][:, 1]

win_aft_tie  = lm["AQ28_win_aft_tie"][:, 1]
tie_aft_tie  = lm["AQ28_tie_aft_tie"][:, 1]
los_aft_tie  = lm["AQ28_los_aft_tie"][:, 1]

win_aft_los  = lm["AQ28_win_aft_los"][:, 1]
tie_aft_los  = lm["AQ28_tie_aft_los"][:, 1]
los_aft_los  = lm["AQ28_los_aft_los"][:, 1]

AQ28scrs     = lm["AQ28scrs"]
soc_skils    = lm["soc_skils"]
imag         = lm["imag"]
rout         = lm["rout"]
switch       = lm["switch"]
fact_pat     = lm["fact_pat"]

performOutcomes = [netwins,
                   win_aft_win, tie_aft_win, los_aft_win,
                   win_aft_tie, tie_aft_tie, los_aft_tie,
                   win_aft_los, tie_aft_los, los_aft_los]
behvfeat     = [soc_skils, imag, rout, switch, fact_pat, AQ28scrs]
######################################################

mCs = [#margCRs[:, 1, 0] - margCRs[:, 1, 2],
       #margCRs[:, 1, 1] - margCRs[:, 2, 1],
       #margCRs[:, 1, 1] - margCRs[:, 0, 1],
       #margCRs[:, 2, 1] - margCRs[:, 0, 1],
       #margCRs[:, 2, 0] - margCRs[:, 2, 2],
       #margCRs[:, 0, 0] - margCRs[:, 0, 2],
       "margCRs[:, 0, 0]", "margCRs[:, 0, 1]", "margCRs[:, 0, 2]",
       "margCRs[:, 1, 0]", "margCRs[:, 1, 1]", "margCRs[:, 1, 2]",
       "margCRs[:, 2, 0]", "margCRs[:, 2, 1]", "margCRs[:, 2, 2]"]

pcpvs = _N.empty((10, 2))

fig = _plt.figure(figsize=(11, 11))
for wtl in range(3):
    for act in range(3):
        mC = margCRs[:, wtl, act]
        colors = ["#CCCCCC"] * 10

        fig.add_subplot(3, 3, wtl*3+act+1)
        for i in range(10):
            pO = performOutcomes[i]
            pc, pv = _ss.pearsonr(mC, pO)
            pcpvs[i, 0] = pc
            pcpvs[i, 1] = pv
            if (pv > 0.01) and (pv < 0.05):
                _plt.text(i-0.15, 0.28, "*", fontsize=10)
            elif (pv > 0.005) and (pv <= 0.01):
                _plt.text(i-0.23, 0.28, "**", fontsize=10)
            elif (pv > 0.001) and (pv <= 0.005):              
                _plt.text(i-0.31, 0.28, "***", fontsize=10)
            if pv < 0.05:
                colors[i] = "black"

        _plt.ylim(-0.35, 0.35)            
        _plt.bar(_N.arange(10), pcpvs[:, 0], color=colors)
        _plt.axvline(x=0.5, ls="--")
        _plt.title("%(w)d  %(a)d" % {"w" : wtl, "a" : act})

    # pcpvs = _N.empty((6, 2))
    
    # fig = _plt.figure(figsize=(9, 3))
    # for i in range(6):
    #     behvf = behvfeat[i]
    #     pc, pv = _ss.pearsonr(behvf, mC)
    #     pcpvs[i, 0] = pc
    #     pcpvs[i, 1] = pv
    #     if (pv > 0.01) and (pv < 0.05):
    #         _plt.text(i-0.1, 0.28, "*", fontsize=18)
    #     elif (pv > 0.005) and (pv <= 0.01):
    #         _plt.text(i-0.15, 0.28, "**", fontsize=18)
    #     elif (pv > 0.001) and (pv <= 0.005):              
    #         _plt.text(i-0.2, 0.28, "***", fontsize=18)
    #     if pv < 0.05:
    #         colors[i] = "black"

    # _plt.ylim(-0.35, 0.35)
    # _plt.axvline(x=4.5, ls="--")

    # _plt.bar(_N.arange(6), pcpvs[:, 0], color=colors)
    


fig = _plt.figure(figsize=(10, 10))
wa = -1
for wtl in range(3):
    for act in range(3):
        colors = ["#CCCCCC"] * 6

        wa += 1
        fig.add_subplot(3, 3, wa + 1)
        pcpvs = _N.empty((6, 2))
        mC = margCRs[:, wtl, act]
        for i in range(6):
            behvf = behvfeat[i]
            pc, pv = _ss.pearsonr(behvf, mC)
            pcpvs[i, 0] = pc
            pcpvs[i, 1] = pv
            if (pv > 0.01) and (pv < 0.05):
                _plt.text(i-0.1, 0.28, "*", fontsize=18)
            elif (pv > 0.005) and (pv <= 0.01):
                _plt.text(i-0.15, 0.28, "**", fontsize=18)
            elif (pv > 0.001) and (pv <= 0.005):              
                _plt.text(i-0.2, 0.28, "***", fontsize=18)
            if pv < 0.05:
                colors[i] = "black"
                
        _plt.ylim(-0.35, 0.35)
        _plt.axvline(x=4.5, ls="--")
        
        _plt.bar(_N.arange(6), pcpvs[:, 0], color=colors)
