#!/usr/bin/python

import numpy as _N
import AIiRPS.utils.read_taisen as _rt
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import AIiRPS.utils.read_taisen as _rd
from filter import gauKer
from scipy.signal import savgol_filter
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, shuffle_discrete_contiguous_regions, mtfftc
import AIiRPS.skull_plot as _sp
import os
import sys
from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
from filter import gauKer
import GCoh.eeg_util as _eu
import AIiRPS.rpsms as rpsms
import GCoh.preprocess_ver as _ppv

import AIiRPS.constants as _cnst
from AIiRPS.utils.dir_util import getResultFN
import GCoh.datconfig as datconf

nicknames = {"FixedSequence(__moRSP__, " +
             "[1, 1, 2, 1, 3, 1, 1, 1, 1, 3, " +
             "1, 1, 2, 1, 1, 1, 1, 1, 1, 3, " + 
             "1, 1, 2, 1, 1, 3, 1, 2, 1, 1, " + 
             "2, 1, 1, 1, 3, 1, 1, 1, 1, 1]);" : "Biased_Random",
             #
             "FixedSequence(__moRSP__, " +
              "[3, 1, 2, 3, 2, 1, 2, 3, 3, 1, " +
             "1, 1, 2, 1, 3, 3, 2, 1, 2, 3, " + 
             "3, 1, 2, 1, 2, 1, 3, 2, 2, 3, " + 
             "2, 1, 3, 3, 2, 2, 3, 1, 3, 1]);" : "Unbiased_Random",
             #
             "WTL(__moRSP__, " +
             "[0.05, 0.85, 0.1], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3], false);"
             : "Exploitable_Win",
             #
             "Mimic(__moRSP__, 0, 0.2);" : "Mimic"}

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

def rm_outliersCC(x, y):
    ix = x.argsort()
    iy = y.argsort()    
    L = len(x)
    x_std = _N.std(x)
    y_std = _N.std(y)
    rmv   = []
    i = 0
    while x[ix[i+1]] - x[ix[i]] > x_std:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > x_std:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > y_std:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > y_std:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    #print("%(ths)d" % {"ths" : len(ths)})
    return _ss.pearsonr(x[interiorPts], y[interiorPts])



#  (12)
#  (13)

#  possible 2-patterns
#
oldnew   = "new"

if oldnew == "old":
    dates = _rt.date_range(start='7/13/2021', end='08/17/2021')
elif oldnew == "new":
    dates = _rt.date_range(start='08/18/2021', end='10/30/2021')

partIDs, dats, cnstrs = _rt.filterRPSdats("TMB1", dates, visits=[1], domainQ=_rt._TRUE_ONLY_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=300, maxIGI=30000, MinWinLossRat=0., has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=4)

pid = 0

netwins  = _N.empty((4, len(partIDs)))
win_aft_win  = _N.empty((4, len(partIDs)))
win_aft_los  = _N.empty((4, len(partIDs)))
win_aft_tie  = _N.empty((4, len(partIDs)))
netwin_aft_win  = _N.empty((4, len(partIDs)))
netwin_aft_los  = _N.empty((4, len(partIDs)))
netwin_aft_tie  = _N.empty((4, len(partIDs)))
tie_aft_win  = _N.empty((4, len(partIDs)))
tie_aft_los  = _N.empty((4, len(partIDs)))
tie_aft_tie  = _N.empty((4, len(partIDs)))
los_aft_win  = _N.empty((4, len(partIDs)))
los_aft_los  = _N.empty((4, len(partIDs)))
los_aft_tie  = _N.empty((4, len(partIDs)))

AQ28scrs  = _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))
#

the_cnstrs = ["FixedSequence(__moRSP__, [1, 1, 2, 1, 3, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1]);",
              "FixedSequence(__moRSP__, [3, 1, 2, 3, 2, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 3, 2, 1, 2, 3, 3, 1, 2, 1, 2, 1, 3, 2, 2, 3, 2, 1, 3, 3, 2, 2, 3, 1, 3, 1]);",
              "Mimic(__moRSP__, 0, 0.2);",
              "WTL(__moRSP__, [0.05, 0.85, 0.1], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3], false);"]
              
pid = 0
for partID in partIDs:
    pid += 1
    AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("/Users/arai/Sites/taisen/DATA/TMB1/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1]})

frmgm  = 20
for blk in range(4):   #  the blocks here are relative to 'the_cnstrs'
    pid = 0
    for partID in partIDs:
        pid += 1
        print("partID    %s" % partID)
        _hnd_dat, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB1", block=(blk+1))

        ind = the_cnstrs.index(cnstr)
        netwins[ind, pid-1] = _N.sum(_hnd_dat[frmgm:, 2])

        #############  HP wins case
        wins = _N.where(_hnd_dat[frmgm:-1, 2] == 1)[0]+frmgm      #  HP wins
        ww   = _N.where(_hnd_dat[wins+1, 2] == 1)[0]
        lw   = _N.where(_hnd_dat[wins+1, 2] == -1)[0]        
        win_aft_win[ind, pid-1] = len(ww) / len(wins)
        netwin_aft_win[ind, pid-1] = (len(ww)-len(lw)) / len(wins)
        
        loses = _N.where(_hnd_dat[frmgm:-1, 2] == -1)[0]+frmgm
        if len(loses) > 0:
            wl   = _N.where(_hnd_dat[loses+1, 2] == 1)[0]
            ll   = _N.where(_hnd_dat[loses+1, 2] == -1)[0]            
            win_aft_los[ind, pid-1] = len(wl) / len(loses)
            netwin_aft_los[ind, pid-1] = (len(wl)-len(ll)) / len(loses)
        else:
            #win_aft_los[blk, pid-1] = 0
            ll   = _N.where(_hnd_dat[loses+1, 2] == -1)[0]            
            win_aft_los[ind, pid-1] = len(wl) / len(loses)
            netwin_aft_los[ind, pid-1] = (0-len(ll)) / len(loses)
            
        ties = _N.where(_hnd_dat[frmgm:-1, 2] == 0)[0]+frmgm
        if len(ties) > 0:
            wt   = _N.where(_hnd_dat[ties+1, 2] == 1)[0]
            lt   = _N.where(_hnd_dat[ties+1, 2] == -1)[0]            
            win_aft_tie[ind, pid-1] = len(wt) / len(ties)
            netwin_aft_tie[ind, pid-1] = (len(wt)-len(lt)) / len(ties)
        else:
            #win_aft_los[blk, pid-1] = 0
            lt   = _N.where(_hnd_dat[ties+1, 2] == -1)[0]
            win_aft_tie[ind, pid-1] = 0#len(wt) / len(ties)
            netwin_aft_tie[ind, pid-1] = 0#(0-len(lt)) / len(ties)

        # wt   = _N.where(_hnd_dat[ties+1, 2] == 1)[0]
        # lt   = _N.where(_hnd_dat[ties+1, 2] == -1)[0]        
        # win_aft_tie[ind, pid-1] = len(wt) / len(ties)
        # netwin_aft_tie[ind, pid-1] = (len(wt)-len(lt)) / len(ties)        
        

lblsz=15
tksz=14
titsz=17

nz = 0.0#11
AQ28scrs  += _N.random.randn(len(partIDs))*nz
soc_skils += _N.random.randn(len(partIDs))*nz
rout      += _N.random.randn(len(partIDs))*nz
switch    += _N.random.randn(len(partIDs))*nz
imag      += _N.random.randn(len(partIDs))*nz
fact_pat  += _N.random.randn(len(partIDs))*nz
netwins   += _N.random.randn(len(partIDs))*nz

#cmp_againsts = ["netwins", "win_aft_win", "win_aft_los", "win_aft_tie"]
cmp_againsts = ["netwins", "netwin_aft_win", "netwin_aft_los", "netwin_aft_tie"]
cmp_againsts_abv = ["NW", "W | W", "W | L", "W | T"]


AQ28cats = ["Soc:SS", "Soc:IM", "Soc:RT", "Soc:SW", "FNmPat", "AQ28"]

#ths = _N.arange(len(partIDs))##_N.where(netwins[blk] < 25)[0]
ths = _N.where(AQ28scrs > 50)[0]

pcspvs = _N.empty((4, 4, 6, 2))  #  4 rules x 4 conditions
for blk in range(4):
    fig = _plt.figure(figsize=(13, 10))
    #nknm = nicknames[cnstrs[partIDs[0]][blk]]
    nknm = nicknames[the_cnstrs[blk]]#nicknames[cnstrs[partIDs[0]][iblk]]    

    _plt.suptitle(nknm, fontsize=(titsz+4))
    ica = -1

    l_comps = len(cmp_againsts)
    for ca in cmp_againsts:
        ica += 1
        biosig = eval("%s[blk]" % ca)  #  like netwins[blk]
        #ths = _N.where(netwins[blk] < 25)[0]

        ax = fig.add_subplot(l_comps, 6, 6*ica+1)        
        ###################
        if ica == l_comps-1:
            _plt.xlabel("soc_skills", fontsize=lblsz)
        _plt.ylabel(ca, fontsize=lblsz)
        _plt.xticks(fontsize=tksz)
        _plt.yticks(fontsize=tksz)    
        _plt.scatter(soc_skils, biosig, color="black", s=6)
        _plt.axhline(y=_N.mean(biosig), ls="--", color="red")
        #pc, pv = _ss.pearsonr(soc_skils[ths], biosig[ths])
        pc, pv = rm_outliersCC(soc_skils[ths], biosig[ths])
        pcspvs[blk, ica, 0] = pc, pv

        if pv < 0.05:
            ax.set_facecolor("#DDDDFF")
            
        _plt.title("%(pc).2f pv<%(pv).3f" % {"pc" : pc, "pv" : pv}, fontsize=titsz)
        ###################
        ax = fig.add_subplot(l_comps, 6, 6*ica+2)
        if ica == l_comps-1:        
            _plt.xlabel("imag", fontsize=lblsz)
        _plt.xticks(fontsize=tksz)
        _plt.yticks(fontsize=tksz)        
        _plt.scatter(imag, biosig, color="black", s=6)
        _plt.axhline(y=_N.mean(biosig), ls="--", color="red")        
        #pc, pv = _ss.pearsonr(imag[ths], biosig[ths])
        pc, pv = rm_outliersCC(imag[ths], biosig[ths])
        pcspvs[blk, ica, 1] = pc, pv        
        if pv < 0.05:
            ax.set_facecolor("#DDDDFF")        
        _plt.title("%(pc).2f pv<%(pv).3f" % {"pc" : pc, "pv" : pv}, fontsize=titsz)        
        ###################
        ax = fig.add_subplot(l_comps, 6, 6*ica+3)
        if ica == l_comps-1:                
            _plt.xlabel("routine", fontsize=lblsz)
        _plt.xticks(fontsize=tksz)
        _plt.yticks(fontsize=tksz)        
        _plt.scatter(rout[ths], biosig[ths], color="black", s=6)
        _plt.axhline(y=_N.mean(biosig), ls="--", color="red")        
        #pc, pv = _ss.pearsonr(rout, biosig[ths])
        pc, pv = rm_outliersCC(rout[ths], biosig[ths])
        pcspvs[blk, ica, 2] = pc, pv                
        if pv < 0.05:
            ax.set_facecolor("#DDDDFF")        
        _plt.title("%(pc).2f pv<%(pv).3f" % {"pc" : pc, "pv" : pv}, fontsize=titsz)                
        ###################
        ax = fig.add_subplot(l_comps, 6, 6*ica+4)
        if ica == l_comps-1:                        
            _plt.xlabel("switch", fontsize=lblsz)
        _plt.xticks(fontsize=tksz)
        _plt.yticks(fontsize=tksz)        
        _plt.scatter(switch[ths], biosig[ths], color="black", s=6)
        _plt.axhline(y=_N.mean(biosig), ls="--", color="red")        
        #pc, pv = _ss.pearsonr(switch[ths], biosig[ths])
        pc, pv = rm_outliersCC(switch[ths], biosig[ths])
        pcspvs[blk, ica, 3] = pc, pv                        
        if pv < 0.05:
            ax.set_facecolor("#DDDDFF")        
        _plt.title("%(pc).2f pv<%(pv).3f" % {"pc" : pc, "pv" : pv}, fontsize=titsz)                        
        ###################
        ax = fig.add_subplot(l_comps, 6, 6*ica+5)
        if ica == l_comps-1:                                
            _plt.xlabel("fact numb pats", fontsize=lblsz)
        _plt.xticks(fontsize=tksz)
        _plt.yticks(fontsize=tksz)        
        _plt.scatter(fact_pat[ths], biosig[ths], color="black", s=6)
        _plt.axhline(y=_N.mean(biosig), ls="--", color="red")        
        #pc, pv = _ss.pearsonr(fact_pat[ths], biosig[ths])
        pc, pv = rm_outliersCC(fact_pat[ths], biosig[ths])
        pcspvs[blk, ica, 4] = pc, pv        
        if pv < 0.05:
            ax.set_facecolor("#DDDDFF")        
        _plt.title("%(pc).2f pv<%(pv).3f" % {"pc" : pc, "pv" : pv}, fontsize=titsz)                                
        ###################
        ax = fig.add_subplot(l_comps, 6, 6*ica+6)
        ax.spines["left"].set_linewidth(4)
        ax.spines["right"].set_linewidth(4)
        ax.spines["top"].set_linewidth(4)
        ax.spines["bottom"].set_linewidth(4)
        
        if ica == l_comps-1:                                        
            _plt.xlabel("AQ28", fontsize=(lblsz+4), fontweight="bold")
        _plt.xticks(fontsize=tksz)
        _plt.yticks(fontsize=tksz)    
        _plt.scatter(AQ28scrs[ths], biosig[ths], color="black", s=6)
        _plt.axhline(y=_N.mean(biosig), ls="--", color="red")        
        #pc, pv = _ss.pearsonr(AQ28scrs[ths], biosig[ths])
        pc, pv = rm_outliersCC(AQ28scrs[ths], biosig[ths])
        pcspvs[blk, ica, 5] = pc, pv        
        if pv < 0.05:
            ax.set_facecolor("#DDDDFF")        
        _plt.title("%(pc).2f pv<%(pv).3f" % {"pc" : pc, "pv" : pv}, fontsize=titsz)                                        

    fig.subplots_adjust(hspace=0.6, wspace=0.45, top=0.9, left=0.09, right=0.97)
    _plt.savefig("%(old)s_AQ_for_%(nknm)s" % {"nknm" : nknm, "old" : oldnew})


fig = _plt.figure(figsize=(13, 2))
ic  = 0
for cat in [soc_skils, imag, rout, switch, fact_pat, AQ28scrs]:
    ic += 1
    fig.add_subplot(1, 6, ic)
    bns = _N.linspace(_N.min(cat)-0.5, _N.max(cat)+0.5, int(_N.max(cat) - _N.min(cat)+2))
    _plt.hist(cat, bins=bns, color="black")
_plt.hist(cat, bins=bns, color="black")
fig.add_subplot(1, 6, 6)
cat = AQ28scrs
bns = _N.arange(_N.min(cat)-0.5, _N.max(cat)+0.5, 2)
_plt.hist(cat, bins=bns, color="black")
_plt.savefig("TMB1_%(old)s_AQ_hist" % {"old" : oldnew})

#
fig = _plt.figure(figsize=(9.2, 3.1))
for iblk in range(4):  ##   ITER OVER RPS RULES, Mimic, Biased Ran etc.
    #  4
    ica = -1
    nknm = nicknames[the_cnstrs[iblk]]#nicknames[cnstrs[partIDs[0]][iblk]]    
    
    for ca in cmp_againsts:
        ica += 1
        # rowstrt = len(cmp_againsts)+3

        # if (iblk == 0) or (iblk == 2):
        #     rowstrt = 0
        # if (iblk == 0) or (iblk == 1):
        #     col     = 0
        # if (iblk == 2) or (iblk == 3):
        #     col     = 1
        
        #_plt.subplot2grid((4, 2), (rowstrt+ica, col))
        _plt.subplot2grid((4, 4), (ica, iblk))

        if ica == 0:
            _plt.title(nknm, fontsize=(titsz-3))
        clrs = []
        for i in range(6):
            if pcspvs[iblk, ica, i, 1] < 0.05:
                clrs.append("black")
            else:
                clrs.append("#CDCDCD")                
        _plt.bar(range(6), pcspvs[iblk, ica, :, 0], color=clrs)
        _plt.ylim(-0.3, 0.3)
        _plt.axhline(y=0, ls=":")
        if iblk == 0:
            _plt.ylabel(cmp_againsts_abv[ica], rotation=0, horizontalalignment="right", fontsize=12)
        if ica == 3:
            _plt.xticks(range(6), AQ28cats, rotation=60, fontsize=12)
        else:
            _plt.xticks([])
        _plt.axvline(x=4.5, ls=":", color="black")
fig.subplots_adjust(wspace=0.37, left=0.12, top=0.91, right=0.99, bottom=0.24, hspace=0.2)
    #  fig = 
_plt.savefig("TMB1_AQ28_summary", transparent=True)
