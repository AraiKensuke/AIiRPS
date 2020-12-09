import AIiRPS.utils.read_taisen as _rt
import numpy as _N
from filter import gauKer 

def empirical_NGS(dat, SHUF=0, win=20):
    _td = _rt.return_hnd_dat(dat)
    Tgame= _td.shape[0]
    cprobs = _N.zeros((SHUF+1, 9, Tgame-win))

    for shf in range(SHUF+1):
        if shf > 0:
            inds = _N.arange(_td.shape[0])
            _N.random.shuffle(inds)
            td = _N.array(_td[inds])
        else:
            td = _td

        scores_wtl1 = _N.zeros(Tgame-1, dtype=_N.int)
        scores_tr10 = _N.zeros(Tgame-1, dtype=_N.int)

        ################################# wtl 1 steps back
        wins_m1 = _N.where(td[0:Tgame-1, 2] == 1)[0]
        ties_m1 = _N.where(td[0:Tgame-1, 2] == 0)[0]
        loss_m1 = _N.where(td[0:Tgame-1, 2] == -1)[0]
        scores_wtl1[wins_m1] = 2
        scores_wtl1[ties_m1] = 1
        scores_wtl1[loss_m1] = 0
        ################################# tr from 1->0
        stays = _N.where(td[0:Tgame-1, 0] == td[1:Tgame, 0])[0]  
        scores_tr10[stays]   = 2
        dngrd = _N.where(((td[0:Tgame-1, 0] == 1) & (td[1:Tgame, 0] == 2)) |
                         ((td[0:Tgame-1, 0] == 2) & (td[1:Tgame, 0] == 3)) |
                         ((td[0:Tgame-1, 0] == 3) & (td[1:Tgame, 0] == 1)))[0]
        scores_tr10[dngrd]   = 1
        upgrd = _N.where(((td[0:Tgame-1, 0] == 1) & (td[1:Tgame, 0] == 3)) |
                         ((td[0:Tgame-1, 0] == 2) & (td[1:Tgame, 0] == 1)) |
                         ((td[0:Tgame-1, 0] == 3) & (td[1:Tgame, 0] == 2)))[0]
        scores_tr10[upgrd]   = 0
        scores    = scores_wtl1 + 3*scores_tr10
        scores_pr = scores_wtl1

        i = 0

        for i in range(0, Tgame-win):
            n_win    = len(_N.where(scores_pr[i:i+win] == 2)[0])
            n_win_st = len(_N.where(scores[i:i+win] == 8)[0])
            n_win_dn = len(_N.where(scores[i:i+win] == 5)[0])
            n_win_up = len(_N.where(scores[i:i+win] == 2)[0])
            n_tie    = len(_N.where(scores_pr[i:i+win] == 1)[0])
            n_tie_st = len(_N.where(scores[i:i+win] == 7)[0])
            n_tie_dn = len(_N.where(scores[i:i+win] == 4)[0])
            n_tie_up = len(_N.where(scores[i:i+win] == 1)[0])
            n_los    = len(_N.where(scores_pr[i:i+win] == 0)[0])
            n_los_st = len(_N.where(scores[i:i+win] == 6)[0])
            n_los_dn = len(_N.where(scores[i:i+win] == 3)[0])
            n_los_up = len(_N.where(scores[i:i+win] == 0)[0])
            if n_win > 0:
                cprobs[shf, 0, i] = n_win_st / n_win
                cprobs[shf, 1, i] = n_win_dn / n_win
                cprobs[shf, 2, i] = n_win_up / n_win
            else:
                cprobs[shf, 0, i] = cprobs[shf, 0, i-1]
                cprobs[shf, 1, i] = cprobs[shf, 1, i-1]
                cprobs[shf, 2, i] = cprobs[shf, 2, i-1]
            if n_tie > 0:
                cprobs[shf, 3, i] = n_tie_st / n_tie
                cprobs[shf, 4, i] = n_tie_dn / n_tie
                cprobs[shf, 5, i] = n_tie_up / n_tie
            else:
                cprobs[shf, 3, i] = cprobs[shf, 3, i-1]
                cprobs[shf, 4, i] = cprobs[shf, 4, i-1]
                cprobs[shf, 5, i] = cprobs[shf, 5, i-1]
            if n_los > 0:
                cprobs[shf, 6, i] = n_los_st / n_los
                cprobs[shf, 7, i] = n_los_dn / n_los
                cprobs[shf, 8, i] = n_los_up / n_los
            else:
                cprobs[shf, 6, i] = cprobs[shf, 6, i-1]
                cprobs[shf, 7, i] = cprobs[shf, 7, i-1]
                cprobs[shf, 8, i] = cprobs[shf, 8, i-1]
    return cprobs, Tgame
 

def kernel_NGS(dat, SHUF=0, kerwin=3):
    _td = _rt.return_hnd_dat(dat)
    Tgame= _td.shape[0]
    cprobs = _N.zeros((3, 3, Tgame-1))

    stay_win, dn_win, up_win, stay_tie, dn_tie, up_tie, stay_los, dn_los, up_los, win_cond, tie_cond, los_cond  = _rt.get_ME_WTL(_td, 0, Tgame)

    gk = gauKer(kerwin)
    gk /= _N.sum(gk)
    all_cnd_tr = _N.zeros((3, 3, Tgame-1))
    ker_all_cnd_tr = _N.ones((3, 3, Tgame-1))*-100

    all_cnd_tr[0, 0, stay_win] = 1
    all_cnd_tr[0, 1, dn_win] = 1
    all_cnd_tr[0, 2, up_win] = 1
    all_cnd_tr[1, 0, stay_tie] = 1
    all_cnd_tr[1, 1, dn_tie] = 1
    all_cnd_tr[1, 2, up_tie] = 1
    all_cnd_tr[2, 0, stay_los] = 1
    all_cnd_tr[2, 1, dn_los] = 1
    all_cnd_tr[2, 2, up_los] = 1

    for iw in range(3):
        if iw == 0:
            cond = _N.sort(win_cond)
        elif iw == 1:
            cond = _N.sort(tie_cond)
        elif iw == 2:
            cond = _N.sort(los_cond)

        for it in range(3):
            print(all_cnd_tr[iw, it, cond])
            ker_all_cnd_tr[iw, it, cond] = _N.convolve(all_cnd_tr[iw, it, cond], gk, mode="same")
            for n in range(1, Tgame-1):
                if ker_all_cnd_tr[iw, it, n] == -100:
                    ker_all_cnd_tr[iw, it, n] = ker_all_cnd_tr[iw, it, n-1]
            n = 0
            while ker_all_cnd_tr[iw, it, n] == -100:
                n += 1
            ker_all_cnd_tr[iw, it, 0:n] = ker_all_cnd_tr[iw, it, n]

    for iw in range(3):
        for_cond = _N.sum(ker_all_cnd_tr[iw], axis=0)
        for it in range(3):
            print(ker_all_cnd_tr[iw, it].shape)
            ker_all_cnd_tr[iw, it] /= for_cond
    
    return ker_all_cnd_tr
    
