import numpy as _N
import AIiRPS.utils.read_taisen as _rt
import matplotlib.pyplot as _plt
import scipy.signal as _ss
import scipy.stats as _sstats
from scipy.signal import savgol_filter
import scipy.io as _scio
import pickle
import os
import GCoh.preprocess_ver as _ppv
import rpsms
import glob
import AIiRPS.constants as _cnst

from AIiRPS.utils.dir_util import getResultFN

partIDs = ["20210609_1230-28", "20210609_1248-16",
           "20210609_1321-35", "20210609_1747-07",
           "20210526_1318-12", "20210526_1358-27",
           "20210526_1416-25", "20210526_1503-39",
           "20200108_1642-20", "20200109_1504-32",
           "20200812_1331-06", "20200812_1252-50",
           "20200818_1546-13", "20200818_1603-42",
           "20200818_1624-01", "20200818_1644-09"]#, "20200601_0748-03", "20210529_1923-44", "20210529_1419-14", "20210606_1237-17", "20201122_1108-25", "20201121_1959-30", "20201121_2131-38"]

running_total = _N.zeros((3, 3), dtype=_N.int)
individual_totals = []
for partID in partIDs:
    totals = _N.zeros((3, 3), dtype=_N.int)
    _hnd_dat, start_time, end_time            = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True)

    for i in range(_hnd_dat.shape[0]-1):
        if _hnd_dat[i+1, 2] == 1:  # WIN
            row = 0
        elif _hnd_dat[i+1, 2] == 0:  # TIE
            row = 1
        else:
            row = 2
            
        if (((_hnd_dat[i, 0] == 1) and (_hnd_dat[i+1, 0] == 2)) or # R to S
            ((_hnd_dat[i, 0] == 2) and (_hnd_dat[i+1, 0] == 3)) or # S to P
            ((_hnd_dat[i, 0] == 3) and (_hnd_dat[i+1, 0] == 1))):  # P to R
            #  DOWNGRADE
            totals[row, 0] += 1
            running_total[row, 0] += 1                
        elif (((_hnd_dat[i, 0] == 1) and (_hnd_dat[i+1, 0] == 3)) or # R to P
              ((_hnd_dat[i, 0] == 2) and (_hnd_dat[i+1, 0] == 1)) or # S to R
              ((_hnd_dat[i, 0] == 3) and (_hnd_dat[i+1, 0] == 2))):  # P to S
            #  UPGRADE
            totals[row, 2] += 1
            running_total[row, 2] += 1                            
        else:
            totals[row, 1] += 1   #  STAY
            running_total[row, 1] += 1                                        

    individual_totals.append(totals)

fig = _plt.figure(figsize=(12, 12))
ip = 0
disp_rows = 4
disp_cols = 6
xs = _N.array([0, 1, 2])
w = 0.9

row = 0
clrs = ["black", "grey"]
for partID in partIDs:
    ip += 1
    row = (ip-1)//disp_cols
    row2 = row%2

    #  1,3,5    2,4,6      
    #  7,9,11   8,10,12   
    totals = individual_totals[ip-1]
    #  First index is 3*disp_cols+ip
    fig.add_subplot(disp_rows, disp_cols, ip)
    _plt.title(partID, fontsize=8)
    for wtl in range(3):
        y0 = 0.7*wtl
        ys = totals[wtl] / _N.sum(totals[wtl])+y0        
        ye = 0.333333+y0
        _plt.fill_between([0, 0+w], [y0, y0], [ys[0], ys[0]], color=clrs[row2])
        _plt.fill_between([1, 1+w], [y0, y0], [ys[1], ys[1]], color=clrs[row2])
        _plt.fill_between([2, 2+w], [y0, y0], [ys[2], ys[2]], color=clrs[row2])
        _plt.plot([0, 2+w], [ye, ye], color="red")    

fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.95, hspace=0.2)

_plt.figure(figsize=(4, 4))
totals = running_total

for wtl in range(3):
    y0 = 0.7*wtl
    ys = totals[wtl] / _N.sum(totals[wtl])+y0
    ye = 0.333333+y0
    _plt.fill_between([0, 0+w], [y0, y0], [ys[0], ys[0]], color="black")
    _plt.fill_between([1, 1+w], [y0, y0], [ys[1], ys[1]], color="black")
    _plt.fill_between([2, 2+w], [y0, y0], [ys[2], ys[2]], color="black")
    _plt.plot([0, 2+w], [ye, ye], color="red")    

