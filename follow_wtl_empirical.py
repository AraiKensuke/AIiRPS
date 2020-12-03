import numpy as _N
import read_taisen as _rt
import matplotlib.pyplot as _plt

lbl=13
#dat_fn="20Aug18-1644-09"
#dat_fn="20Jun01-0748-03"
#dat_fn="20May29-1923-44"


dat_fn       = "20Jan09-1504-32"
#
# dat_fn = "20Aug12-1331-06"
# dat_fn="20Nov27-0137-31"
# dat_fn="20Nov27-0250-02"
# dat_fn="20Nov27-0251-01"
dat_fn="20Nov27-0252-00"
# dat_fn="20Nov27-0253-01"
# dat_fn="20Nov27-0253-58"
# dat_fn="20Nov27-0614-01"
# dat_fn="20Nov27-0614-42"
# dat_fn="20Nov27-0619-11"
#dat_fn="20Nov27-0620-09"
# dat_fn="20Nov27-0704-20"
# dat_fn="20Nov27-0706-34"
# dat_fn="20Nov27-0712-21"
dat_fn="20Nov27-0715-26"
dat_fn="20Nov21-2131-38"
dat_fn="20Nov22-1108-25"
dat_fn="20Nov21-1959-30"
know_gt=False
#sran  = "ran" if rndmz else ""

def calc_lvs(Ts):
    n = Ts.shape[0]
    return 3*_N.sum((Ts[0:-1] - Ts[1:])**2 / (Ts[0:-1] + Ts[1:])**2) / (n-1)

def one_condition(cond, stays, dns, ups):
    """
    """
    all_trs = _N.arange(len(cond))
    l_cond = cond.tolist()

    sty_consecut    = _N.zeros(len(stays))
    dns_consecut    = _N.zeros(len(dns))
    ups_consecut    = _N.zeros(len(ups))

    for i in range(len(stays)):
        sty_consecut[i] = l_cond.index(stays[i])
    for i in range(len(dns)):
        dns_consecut[i] = l_cond.index(dns[i])
    for i in range(len(ups)):
        ups_consecut[i] = l_cond.index(ups[i])

    return all_trs, sty_consecut, dns_consecut, ups_consecut

def get_rank(stds, similars, SHUFFLE, col):
    print("len similars  %d" % len(similars))
    srtd = _N.sort(stds[similars, col])
    rank = _N.where((stds[SHUFFLE, col] >= srtd[0:-1]) & (stds[SHUFFLE, col] < srtd[1:]))[0]
    if len(rank) == 0:
        if stds[SHUFFLE, col] > srtd[-1]:
            return len(similars) + 1, srtd
        elif stds[SHUFFLE, col] < srtd[0]:
            return -1, srtd
    else:
        return rank[0], srtd
    print("woops %(0).3f   %(-1).3f   %(val).3f" % {"0" : srtd[0], "-1" : srtd[-1], "val" : stds[SHUFFLE, col]})
    return None, srtd

if not know_gt: 
    _hnd_dat = _rt.return_hnd_dat(dat_fn)    
else:
    _hnd_dat     = _N.loadtxt("/Users/arai/nctc/Workspace/AIiRPS_SimDat/rpsm_%s.dat" % dat_fn, dtype=_N.int)


SHUFFLE  = 2000
stds     = _N.empty((SHUFFLE+1, 9))
means    = _N.empty((SHUFFLE+1, 9))
lvs      = _N.empty((SHUFFLE+1, 9))
lens     = _N.zeros((SHUFFLE+1, 9), dtype=_N.int)

for shf in range(SHUFFLE + 1):
    if shf == SHUFFLE:
        hnd_dat = _hnd_dat
    else:
        inds = _N.arange(_hnd_dat.shape[0])
        _N.random.shuffle(inds)
        hnd_dat = _N.array(_hnd_dat[inds])

    win_st, win_dn, win_up, tie_st, tie_dn, tie_up, los_st, los_dn, los_up, wins, ties, loss = _rt.get_ME_WTL(hnd_dat, 0, hnd_dat.shape[0])

    win_cons, win_st_cons, win_dn_cons, win_up_cons = one_condition(_N.sort(wins), win_st, win_dn, win_up)
    tie_cons, tie_st_cons, tie_dn_cons, tie_up_cons = one_condition(_N.sort(ties), tie_st, tie_dn, tie_up)
    los_cons, los_st_cons, los_dn_cons, los_up_cons = one_condition(_N.sort(loss), los_st, los_dn, los_up)

    stds[shf, 0] = _N.std(_N.diff(win_st_cons))    
    stds[shf, 1] = _N.std(_N.diff(win_dn_cons))    
    stds[shf, 2] = _N.std(_N.diff(win_up_cons))    
    stds[shf, 3] = _N.std(_N.diff(tie_st_cons))    
    stds[shf, 4] = _N.std(_N.diff(tie_dn_cons))    
    stds[shf, 5] = _N.std(_N.diff(tie_up_cons))    
    stds[shf, 6] = _N.std(_N.diff(los_st_cons))    
    stds[shf, 7] = _N.std(_N.diff(los_dn_cons))    
    stds[shf, 8] = _N.std(_N.diff(los_up_cons))    
    lvs[shf, 0]  = calc_lvs(_N.diff(win_st_cons))
    lvs[shf, 1]  = calc_lvs(_N.diff(win_dn_cons))
    lvs[shf, 2]  = calc_lvs(_N.diff(win_up_cons))
    lvs[shf, 3]  = calc_lvs(_N.diff(tie_st_cons))
    lvs[shf, 4]  = calc_lvs(_N.diff(tie_dn_cons))
    lvs[shf, 5]  = calc_lvs(_N.diff(tie_up_cons))
    lvs[shf, 6]  = calc_lvs(_N.diff(los_st_cons))
    lvs[shf, 7]  = calc_lvs(_N.diff(los_dn_cons))
    lvs[shf, 8]  = calc_lvs(_N.diff(los_up_cons))
    means[shf, 0] = _N.mean(_N.diff(win_st_cons))    
    means[shf, 1] = _N.mean(_N.diff(win_dn_cons))    
    means[shf, 2] = _N.mean(_N.diff(win_up_cons))    
    means[shf, 3] = _N.mean(_N.diff(tie_st_cons))    
    means[shf, 4] = _N.mean(_N.diff(tie_dn_cons))    
    means[shf, 5] = _N.mean(_N.diff(tie_up_cons))    
    means[shf, 6] = _N.mean(_N.diff(los_st_cons))    
    means[shf, 7] = _N.mean(_N.diff(los_dn_cons))    
    means[shf, 8] = _N.mean(_N.diff(los_up_cons))    
    lens[shf, 0] = len(win_st_cons)
    lens[shf, 1] = len(win_dn_cons)
    lens[shf, 2] = len(win_up_cons)
    lens[shf, 3] = len(tie_st_cons)
    lens[shf, 4] = len(tie_dn_cons)
    lens[shf, 5] = len(tie_up_cons)
    lens[shf, 6] = len(los_st_cons)
    lens[shf, 7] = len(los_dn_cons)
    lens[shf, 8] = len(los_up_cons)


#for col in range(9):

lngst = _N.max(_N.array([len(win_cons), len(tie_cons), len(los_cons)]))

fig = _plt.figure(figsize=(11, 7))
_plt.suptitle("%(df)s" % {"df" : dat_fn})
ax = _plt.subplot2grid((6, 3), (0, 0), colspan=3)
ax.set_facecolor("#BBFFBB")
_plt.scatter(win_st_cons, _N.ones(len(win_st_cons)), marker=".", color="black")
_plt.scatter(win_dn_cons, _N.ones(len(win_dn_cons))-1, marker="|", color="black", s=100)
_plt.scatter(win_up_cons, _N.ones(len(win_up_cons))-2, marker="|", color="grey", s=100)
_plt.xlim(0, lngst)
_plt.yticks([-1, 0, 1], ["upgrd", "dngrd", "stay"])
_plt.ylim(-2, 2)
_plt.ylabel("after win", fontsize=lbl)
###################################
ax = _plt.subplot2grid((6, 3), (1, 0), colspan=3)
ax.set_facecolor("#FFFFBB")
_plt.scatter(tie_st_cons, _N.ones(len(tie_st_cons)), marker=".", color="black")
_plt.scatter(tie_dn_cons, _N.ones(len(tie_dn_cons))-1, marker="|", color="black", s=100)
_plt.scatter(tie_up_cons, _N.ones(len(tie_up_cons))-2, marker="|", color="grey", s=100)
_plt.xlim(0, lngst)
_plt.yticks([-1, 0, 1], ["upgrd", "dngrd", "stay"])
_plt.ylim(-2, 2)
_plt.ylabel("after tie", fontsize=lbl)
###################################
ax = _plt.subplot2grid((6, 3), (2, 0), colspan=3)
ax.set_facecolor("#FFBBBB")
_plt.scatter(los_st_cons, _N.ones(len(los_st_cons)), marker=".", color="black")
_plt.scatter(los_dn_cons, _N.ones(len(los_dn_cons))-1, marker="|", color="black", s=100)
_plt.scatter(los_up_cons, _N.ones(len(los_up_cons))-2, marker="|", color="grey", s=100)
_plt.ylim(-2, 2)
_plt.yticks([-1, 0, 1], ["upgrd", "dngrd", "stay"])
_plt.xlim(0, lngst)
_plt.ylabel("after loss", fontsize=lbl)

face_col = ["#BBFFBB", "#FFFFBB", "#FFBBBB"]
ig = -1
ranks = _N.empty(9)

for row in range(3, 6):
    for col in range(0, 3):
        ig += 1
        ax = _plt.subplot2grid((6, 3), (row, col), colspan=1)
        ax.set_facecolor(face_col[row-3])
        #rank, srtd = get_rank(stds/means, SHUFFLE, ig)

        ths      = _N.argsort(_N.abs(lens[0:-1, 0] - lens[SHUFFLE, 0]))
        similars = ths[0:100]
        print(len(similars))
        rank, srtd = get_rank(lvs, similars, SHUFFLE, ig)
        ranks[ig] = rank
        _plt.plot(srtd, color="grey")
        #_plt.scatter([rank], [stds[SHUFFLE, ig]/means[SHUFFLE, ig]], marker=".", color="black", s=150)
        _plt.scatter([rank], [lvs[SHUFFLE, ig]], marker=".", color="black", s=150)
        _plt.axvline(x=rank, ls=":", color="black")
        _plt.xlim(-3, len(similars)+3)

_plt.savefig("empirical_%(df)s" % {"df" : dat_fn})


