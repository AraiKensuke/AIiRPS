import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss

#frng = [35, 45]
#frng = [15, 25]
#frng  = [32, 48]
#frng  = [34, 46]
#frng  = [30, 40]
#frng = [35, 45]
#frng = [5, 15]
#frng = [20, 30]
frng = [7, 15]
#frng = [48, 55]
#frng = [4, 8]
#frng = [7, 12]

lags_sec=50
MULT=1.
fxdGK  = None

if fxdGK is not None:
    sMULT = "G%d" % fxdGK
else:
    sMULT = "%.1f" % MULT

fig = _plt.figure(figsize=(8, 8))
fi = 0

lblsz = 24
tksz  = 20
for flip in [False, True]:
    fi += 1
    sflip = "_flip" if flip else ""
    xy = _N.loadtxt("times_O_%(1)d_%(2)d_%(lags)d_%(mult)s%(sflip)s.txt" % {"1" : frng[0], "2" : frng[1], "mult" : sMULT, "sflip" : sflip, "lags" : lags_sec})

    good_ax0 = _N.where(xy[:, 0] > 5)[0]
    good_ax1 = _N.where(xy[:, 1] > 5)[0]
    good12 = _N.intersect1d(good_ax1, good_ax0)
    print(good12)

    A = _N.vstack([xy[good12, 0], _N.ones(xy[good12].shape[0])]).T
    
    m, c = _N.linalg.lstsq(A, xy[good12, 1])[0]

    xdisp = _N.array([0, 60])

    fig.add_subplot(2, 2, fi)

    _plt.plot(xdisp, m*xdisp + c, lw=2, color="black")
    _plt.plot([0, 60], [0, 60], color="grey", ls=":", lw=4)
    _plt.scatter(xy[good12, 0], xy[good12, 1], color="black") # x-axis from EEG, y-axis behavior
    _plt.xlim(0, 60)
    _plt.ylim(0, 60)
    _plt.xticks(_N.arange(0, 61, 20), fontsize=tksz)
    _plt.yticks(_N.arange(0, 61, 20), fontsize=tksz)
    _plt.xlabel("PSI (sec.)", fontsize=lblsz)
    _plt.ylabel("RSI (sec.)", fontsize=lblsz)
    pc, pv = _ss.pearsonr(xy[good12, 0], xy[good12, 1])
    _plt.text(5, 50, "CC=%(pc).2f\npv < %(pv).1e" % {"pc" : pc, "pv" : pv}, fontsize=(tksz-5))
    if flip:
        _plt.title("flip")
#fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9, wspace=0.4)
#_plt.suptitle("%(1)d_%(2)d_%(lags)d_%(mult).1f" % {"1" : frng[0], "2" : frng[1], "mult" : MULT, "lags" : lags_sec})
#_plt.savefig("PSI_RSI_%(1)d_%(2)d_%(lags)d_%(mult).1f.png" % {"1" : frng[0], "2" : frng[1], "mult" : MULT, "sflip" : sflip, "lags" : lags_sec})

#fig = _plt.figure(figsize=(9, 9))
#fig = _plt.figure(figsize=(8, 4))
fi = 0
for flip in [False, ]:
    fi += 1
    sflip = "_flip" if flip else ""
    xyz = _N.loadtxt("times_gmint_v_ACts_%(1)d_%(2)d_%(lags)d_%(mult)s%(sflip)s.txt" % {"1" : frng[0], "2" : frng[1], "mult" : sMULT, "sflip" : sflip, "lags" : lags_sec})
    A = _N.vstack([xyz[:, 0], _N.ones(xyz.shape[0])]).T
    mR, cR = _N.linalg.lstsq(A, xyz[:, 1])[0]
    A = _N.vstack([xyz[:, 0], _N.ones(xyz.shape[0])]).T
    mP, cP = _N.linalg.lstsq(A, xyz[:, 2])[0]
    xdisp = _N.array([0, 5])


    fig.add_subplot(2, 2, 2+fi)
    _plt.scatter(xyz[:, 0], xyz[:, 1], color="black")
    _plt.plot(xdisp, mR*xdisp + cR, color="black")
    _plt.ylim(0, 50)
    _plt.xlim(0, 5)
    _plt.xlabel("IGI (sec.)", fontsize=lblsz)
    _plt.ylabel("RSI (sec.)", fontsize=lblsz)
    _plt.xticks(fontsize=tksz)
    _plt.yticks(fontsize=tksz)
    if flip:
        _plt.text(0.5, 37, "flip\n$T_{sw}$=%.1f" % mR, fontsize=lblsz)
    else:
        _plt.text(0.5, 37, "$T_{sw}$=%.1f" % mR, fontsize=lblsz)        

    fig.add_subplot(2, 2, fi+3)
    _plt.scatter(xyz[:, 0], xyz[:, 2], color="black")
    _plt.ylim(0, 50)
    _plt.xlim(0, 5)
    _plt.plot(xdisp, mP*xdisp + cP, color="black")
    _plt.xlabel("IGI (sec.)", fontsize=lblsz)
    _plt.ylabel("PSI (sec.)", fontsize=lblsz)
    _plt.xticks(fontsize=tksz)
    _plt.yticks(fontsize=tksz)
    if flip:
        _plt.text(0.5, 37, "flip\n$T_{sw}$=%.1f" % mP, fontsize=lblsz)
    else:
        _plt.text(0.5, 37, "$T_{sw}$=%.1f" % mP, fontsize=lblsz)        

fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.92, hspace=0.4, wspace=0.5)
_plt.suptitle("%(1)d_%(2)d_%(lags)d_%(mult)s %(sflip)s" % {"1" : frng[0], "2" : frng[1], "mult" : sMULT, "sflip" : sflip, "lags" : lags_sec})    
_plt.savefig("PSIRSIk_%(1)d_%(2)d_%(lags)d_%(mult)s.png" % {"1" : frng[0], "2" : frng[1], "mult" : sMULT, "lags" : lags_sec})



ms_f_nf = []
iflp = -1
for flip in [False, True]:
    iflp += 1
    fig = _plt.figure(figsize=(12, 14))

    sflip = "_flip" if flip else ""
    print(flip)
    with open("times_O_%(1)d_%(2)d_%(lags)d_%(mult)s%(sflip)s.txt" % {"1" : frng[0], "2" : frng[1], "mult" : sMULT, "sflip" : sflip, "lags" : lags_sec}) as fp:
        lines = fp.read().splitlines()

    lblsz=11
    tksz=9
    datForPart = []
    ip = 0

    sflip = "flpd" if flip else ""
    ms = []
    offset = 0#16 if flip else 0
    for line in lines:
        if line[0] == "#":
            ip += 1
            fig.add_subplot(5, 4, ip+offset)
            
            if len(datForPart) > 0:
                xy = _N.array(datForPart)
                A = _N.vstack([xy[:, 0], _N.ones(xy.shape[0])]).T
                m, c = _N.linalg.lstsq(A, xy[:, 1])[0]
                xs = _N.array([0, 120])
                _plt.scatter(xy[:, 0], xy[:, 1], color="black", s=3)
                clr = "blue" if m > 0 else "red"
                _plt.plot(xs, m*xs + c, color=clr)
                if xy.shape[0] > 2:
                    ms.append(m)
                    
                    print("%(1)s  %(2).2f" % {"1" : line[-16:], "2" : m})
                    _plt.xlim(0, 60)
                    _plt.ylim(0, 60)
                    _plt.xticks([0, 15, 30, 45], fontsize=tksz)
                    _plt.yticks([0, 15, 30, 45], fontsize=tksz)
                    _plt.xlabel("RSI", fontsize=lblsz)
                    _plt.ylabel("PSI", fontsize=lblsz)
                    _plt.plot([0, 60], [0, 60], ls=":", color="grey", lw=2)
                    _plt.text(50, 5, "%(f)s sbj %(s)d" % {"s" : ip, "f" : sflip}, fontsize=tksz)
                    _plt.title(line[-16:])

                    datForPart = []
        else:
            vals = line.split("  ")
            datForPart.append([float(vals[0]), float(vals[1])])
    ms_f_nf.append(ms)

    fig.subplots_adjust(wspace=0.4, hspace=0.4, left=0.09, right=0.98, bottom=0.09, top=0.9)
    _plt.suptitle("%(1)d_%(2)d_%(lags)d_%(mult).1f %(sflip)s" % {"1" : frng[0], "2" : frng[1], "mult" : MULT, "sflip" : sflip, "lags" : lags_sec})    
    _plt.savefig("PSI_RSI_per_participant_%(1)d_%(2)d_%(lags)d_%(mult).1f%(flip)s.png" % {"1" : frng[0], "2" : frng[1], "mult" : MULT, "lags" : lags_sec, "flip" : sflip})
    _plt.close()
fig = _plt.figure()
_plt.hist(ms_f_nf[iflp], bins=_N.linspace(-2, 2, 41))
_plt.axvline(x=1, color="red", lw=2)
_plt.savefig("PSI_RSI_hists_%(1)d_%(2)d_%(lags)d_%(mult).1f%(flip)s.png" % {"1" : frng[0], "2" : frng[1], "mult" : MULT, "lags" : lags_sec, "flip" : sflip})
