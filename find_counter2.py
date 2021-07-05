from AIiRPS.models import empirical as _em
from filter import gauKer
import numpy as _N
import matplotlib.pyplot as _plt
import GCoh.eeg_util as _eu
import AIiRPS.utils.read_taisen as _rt
from AIiRPS.utils.dir_util import getResultFN
import os
import pickle

def getTargetCR_cntrmvs(condition):
    if condition ==0:
        rep_cntr_noncntr = [[_em.st_win, _em.dn_win, _em.up_win],
                            [_em.dn_win, _em.up_win, _em.st_win],
                            [_em.up_win, _em.st_win, _em.dn_win]]
        s_rep_cntr_noncntr = [["ST | WIN", "DN | WIN", "UP | WIN"],
                              ["DN | WIN", "UP | WIN", "ST | WIN"],
                              ["UP | WIN", "ST | WIN", "DN | WIN"]]
            
    elif condition == 1:
        rep_cntr_noncntr = [[_em.st_tie, _em.dn_tie, _em.up_tie],
                            [_em.dn_tie, _em.up_tie, _em.st_tie],
                            [_em.up_tie, _em.st_tie, _em.dn_tie]]
        s_rep_cntr_noncntr = [["ST | TIE", "DN | TIE", "UP | TIE"],
                              ["DN | TIE", "UP | TIE", "ST | TIE"],
                              ["UP | TIE", "ST | TIE", "DN | TIE"]]
            
    elif condition == 2:
        rep_cntr_noncntr = [[_em.st_los, _em.dn_los, _em.up_los],
                            [_em.dn_los, _em.up_los, _em.st_los],
                            [_em.up_los, _em.st_los, _em.dn_los]]
        s_rep_cntr_noncntr = [["ST | LOS", "DN | LOS", "UP | LOS"],
                              ["DN | LOS", "UP | LOS", "ST | LOS"],
                              ["UP | LOS", "ST | LOS", "DN | LOS"]]
    return rep_cntr_noncntr, s_rep_cntr_noncntr

ratio = True
def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)

_W = 1
_T = 0
_L = -1

rep_CR = 2

flip_human_AI = False
#  These are ParticipantIDs.
#datetms=["20210609_1230-28"]#, "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09", "20200601_0748-03", "20210529_1923-44", "20210529_1419-14", "20210606_1237-17", "20210609_1517-23", "20201122_1108-25", "20201121_1959-30", "20201121_2131-38"]
datetms=["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39", "20200108_1642-20", "20200109_1504-32", "20200812_1252-50", "20200812_1331-06", "20200818_1546-13", "20200818_1603-42", "20200818_1624-01", "20200818_1644-09", "20200601_0748-03", "20210529_1923-44", "20210529_1419-14", "20210606_1237-17", "20210609_1517-23", "20201122_1108-25", "20201121_1959-30", "20201121_2131-38", "20200410_2203-19", "20200410_2248-43", "20200415_2034-12", "20200418_2148-58"]
datetms=["20210609_1230-28"]

#datetms=["20210607_1434-03", "20210607_1434-20", "20210607_1434-52", "20210607_1435-13", "20210607_1435-36", "20210607_1435-56", "20210607_1436-20", "20210607_1436-36", "20210607_1502-42", "20210607_1503-21", "20210607_1503-36", "20210607_1503-52", "20210607_1504-07", "20210607_1504-49", "20210607_1505-09", "20210607_1505-39", "20210607_1951-49"]


#fig = _plt.figure(figsize=(11, 11))
id = -1

#  len(dates), WTL, 
wtl_num_CNTR_nonCNTR = _N.zeros((len(datetms), 3, 3, 2))


di = -1
cts = _N.zeros((len(datetms), 3, 3, 3), dtype=_N.int)  # date, Cond, rule, cntr,keep, ncntr

conditions = [_W, _T, _L]

for datetm in datetms:
    print("%s------------------------"  % datetm)
    di += 1

    CRs  = _em.CRs(datetm)
    td, start_tm, end_tm = _rt.return_hnd_dat(datetm, has_useragent=True, has_start_and_end_times=True, has_constructor=True, flip_human_AI=False)
    #  First find all the win conditions

    #  _W followed by a win

    #ic = -1
    for ic in range(3):#condition in range(3):
        #ic += 1

        rep_cntr_noncntr, s_rep_cntr_noncntr = getTargetCR_cntrmvs(ic)
        
        #iWTLs = _N.where(td[0:-1, 2] == conditions[ic])[0]
        iWTLs = _N.where(td[0:-2, 2] == conditions[ic])[0]

        # #Find next win after each win-win.

        for ir in range(3):   ##  for a given repeating CR, 
            CR_trgt = rep_cntr_noncntr[ir][0]
            cntr    = rep_cntr_noncntr[ir][1]
            ncntr    = rep_cntr_noncntr[ir][2]                        


            for i in range(len(iWTLs)-2):
                iww0 = iWTLs[i]
                iww1 = iWTLs[i+1]
                iww2 = iWTLs[i+2]        

                #repeatedCR = _N.ones([True]*len(iWTLs))
                repeatedCR = (CRs[iww0] == CRs[iww1])
                #repeatedCR_and_won2 = (td[iww0+1, 2] == _W) and (td[iww1+1, 2] == _W)

                if repeatedCR:
                    if (CRs[iww0] == CR_trgt) and (CRs[iww2] == cntr):
                        cts[di, ic, ir, 0] += 1
                    elif (CRs[iww0] == CR_trgt) and (CRs[iww2] == CR_trgt):
                        cts[di, ic, ir, 1] += 1                
                    elif (CRs[iww0] == CR_trgt) and (CRs[iww2] == ncntr):
                        cts[di, ic, ir, 2] += 1
            """ 
            for i in range(len(iWTLs)-3):
                iww0 = iWTLs[i]
                iww1 = iWTLs[i+1]
                iww2 = iWTLs[i+2]
                iww3 = iWTLs[i+3]                        

                repeatedCR = (CRs[iww0] == CRs[iww1]) and (CRs[iww0] == CRs[iww2])

                if repeatedCR:
                    if (CRs[iww2] == CR_trgt) and (CRs[iww3] == cntr):
                        cts[di, ic, ir, 0] += 1
                    elif (CRs[iww2] == CR_trgt) and (CRs[iww3] == CR_trgt):
                        cts[di, ic, ir, 1] += 1                
                    elif (CRs[iww2] == CR_trgt) and (CRs[iww3] == ncntr):
                        cts[di, ic, ir, 2] += 1
            """ 
            

"""
sMv  = ["ST", "DN", "UP"]
sCnd = ["WIN", "TIE", "LOS"]        
fig = _plt.figure(figsize=(12, 12))

iwt = 0
for wtl in range(1):
    rep_cntr_noncntr, s_rep_cntr_noncntr = getTargetCR_cntrmvs(wtl)   
    for typ in range(1):
        iwt += 1
        ax = fig.add_subplot(3, 3, iwt)
        ax.set_facecolor("#CCCCCC")

        ratCs = _N.zeros(len(datetms))
        ratNs = _N.zeros(len(datetms))        
        for i in range(len(datetms)):
            if cts[i, wtl, typ, 0] > cts[i, wtl, typ, 2]:
                clr = "black"
            elif cts[i, wtl, typ, 0] == cts[i, wtl, typ, 2]:
                clr = "yellow"
            if cts[i, wtl, typ, 0] < cts[i, wtl, typ, 2]:
                clr = "red"

            #  plot number of times participant uses cntr vs. doesn't use cntr
            #_plt.plot([0, 1], [cts[i, wtl, typ, 0], cts[i, wtl, typ, 2]], color=clr, lw=2, marker=".")

            thesum = cts[i, wtl, typ, 0] + cts[i, wtl, typ, 2]
            ratCs[i]   = cts[i, wtl, typ, 0] / thesum
            ratNs[i]   = cts[i, wtl, typ, 2] / thesum
            if thesum == 0:
                ratCs[i]   = 0.5
                ratNs[i]   = 0.5
            
            nz0 = 0.05*_N.random.randn()
            
            _plt.plot([0, 1], [ratCs[i] + nz0, ratNs[i] + nz0], color=clr, lw=2, marker=".")
        _plt.scatter([-0.1], [_N.mean(ratCs)], marker="*", s=20, color="blue")
        _plt.scatter([1.1], [_N.mean(ratNs)], marker="*", s=20, color="blue")
        #  row has same condition
        _plt.text(-1.7, 0.2, "player repeats\n%(rptedCR)s\n\nCountering move\n%(cntr)s\n\nTying move\n%(tying)s" % {"rptedCR" : s_rep_cntr_noncntr[typ][0], "cntr" : s_rep_cntr_noncntr[typ][1], "tying" : s_rep_cntr_noncntr[typ][2]})
        #_plt.text(-1.2, _N.max(cts[:, wtl, typ, _N.array([0, 2])].flatten())*0.9, "%(wtl)d  %(typ)d" % {"wtl" : wtl, "typ" : typ})
        _plt.xlim(-1.8, 1.2)
        _plt.ylim(-0.1, 1.1)
        _plt.axhline(y=1, ls=":", color="grey")
        _plt.axhline(y=0, ls=":", color="grey")

        _plt.xticks([0, 1], ["#counters", "#tying"])

_plt.suptitle("If player keeps playing a CR, say STAY | WIN, chances are opponent will expect player to STAY the next time player WINS.\nTo win against this STAY by the player, opponent needs to downgrade their losing hand.  The player, instead of STAYing,\ncan also downgrade their winning hand in anticipation that the opponent will try to beat a STAY. This is a counter move.\nThe player can also keep on playing STAY|WIN, or they can play a move that will tie the move the\nopponent should make if they believed the player would STAY.")
fig.subplots_adjust(bottom=0.1, left=0.08, right=0.95, top=0.85)
_plt.savefig("find_counter2.png")

# net_cntr = _N.zeros((3, 3))
# fig = _plt.figure(figsize=(13, 13))
# for i in range(len(datetms)):
#     fig.add_subplot(5, 5, i+1)
#     for wtl in range(3):
#         for typ in range(3):
#             thesum = cts[i, wtl, typ, 0] + cts[i, wtl, typ, 2]
#             net_cntr[wtl, typ] = cts[i, wtl, typ, 0]/thesum - cts[i, wtl, typ, 2] / thesum
#     _plt.title("%(cn)d  %(ncn)d" % {"cn" : _N.sum(cts[i, : , :, 0]), "ncn" : _N.sum(cts[i, : , :, 2])})
#     _plt.ylim(-1, 1)
#     _plt.axhline(y=0, ls=":")
#     _plt.plot(net_cntr.flatten(), color="black", marker=".")
# fig.subplots_adjust(wspace=0.35, hspace=0.45)



cntr_bias = _N.zeros(len(datetms))
#for wtl in range(3):
#    for typ in range(3):
for i in range(len(datetms)):
    thesum = cts[i, 0, 0, 0] + cts[i, 0, 0, 2]
    cntr_bias[i] = (cts[i, 0, 0, 0] - cts[i, 0, 0, 2]) / thesum

srtdInds = cntr_bias.argsort()
i0 = 0
i1 = 14
i2 = 28
srtdInds[i0:i1]
srtdInds[i1:i2]

wtl=0
typ=2

totals = _N.zeros(len(datetms))
for wtl in range(3):
    for typ in range(3):
        if (wtl != 0) or (typ != 0):
            totals[:] = 0
            for i in range(len(datetms)):
                thesum = cts[i, wtl, typ, 0] + cts[i, wtl, typ, 2]
                totals[i] = (cts[i, wtl, typ, 0] - cts[i, wtl, typ, 2]) / thesum

            print("..............%(wtl)d  %(typ)d" % {"wtl" : wtl, "typ" : typ})
            mask = _N.isnan(totals[srtdInds[i0:i1]] )
            imask = _N.logical_not(mask)
            ths1  = _N.where(imask)[0]
            mask = _N.isnan(totals[srtdInds[i1:i2]] )
            imask = _N.logical_not(mask)
            ths2  = _N.where(imask)[0]
            print(totals[srtdInds[i0:i1]])
            print(totals[srtdInds[i1:i2]])
            print("%(1).3f   %(2).3f" % {"1" : _N.sum(totals[srtdInds[i0:i1][ths1]]), "2" : _N.sum(totals[srtdInds[i1:i2][ths2]])})
"""
