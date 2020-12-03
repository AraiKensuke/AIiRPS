import numpy as _N
import read_taisen as _rt
import matplotlib.pyplot as _plt

#dat_fn="20Aug18-1644-09"
dat_fn="20Nov21-2131-38"
dat_fn="20Jun01-0748-03"
dat_fn="20May29-1923-44"
#dat_fn="20Nov21-1959-30"
dat_fn="20Nov22-1108-25"

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

    return all, sty_consecut, dns_consecut, ups_consecut

hnd_dat = _rt.return_hnd_dat(dat_fn)

win_st, win_dn, win_up, tie_st, tie_dn, tie_up, los_st, los_dn, los_up, wins, ties, loss = _rt.get_ME_WTL(hnd_dat, 0, hnd_dat.shape[0])

win_cons, win_st_cons, win_dn_cons, win_up_cons = one_condition(_N.sort(wins), win_st, win_dn, win_up)
tie_cons, tie_st_cons, tie_dn_cons, tie_up_cons = one_condition(_N.sort(ties), tie_st, tie_dn, tie_up)
los_cons, los_st_cons, los_dn_cons, los_up_cons = one_condition(_N.sort(loss), los_st, los_dn, los_up)

fig = _plt.figure(figsize=(11, 5))
fig.add_subplot(3, 1, 1)
_plt.scatter(win_st_cons, _N.ones(len(win_st_cons)), marker=".", color="black")
_plt.scatter(win_dn_cons, _N.ones(len(win_dn_cons))-1, marker="|", color="black")
_plt.scatter(win_up_cons, _N.ones(len(win_up_cons))-2, marker="|", color="blue")
_plt.ylim(-5, 3)
fig.add_subplot(3, 1, 2)
_plt.scatter(tie_st_cons, _N.ones(len(tie_st_cons)), marker=".", color="black")
_plt.scatter(tie_dn_cons, _N.ones(len(tie_dn_cons))-1, marker="|", color="black")
_plt.scatter(tie_up_cons, _N.ones(len(tie_up_cons))-2, marker="|", color="blue")
_plt.ylim(-5, 3)
fig.add_subplot(3, 1, 3)
_plt.scatter(los_st_cons, _N.ones(len(los_st_cons)), marker=".", color="black")
_plt.scatter(los_dn_cons, _N.ones(len(los_dn_cons))-1, marker="|", color="black")
_plt.scatter(los_up_cons, _N.ones(len(los_up_cons))-2, marker="|", color="orange")
_plt.ylim(-5, 3)
