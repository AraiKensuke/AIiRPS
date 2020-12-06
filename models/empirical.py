import AIiRPS.utils.read_taisen as _rt
import numpy as _N

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
                print("zero")
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
