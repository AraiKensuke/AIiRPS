import AIiRPS.utils.read_taisen as _rt
import numpy as _N
from filter import gauKer
import AIiRPS.constants as _AIconst

dn_win = 0
st_win = 1
up_win = 2
dn_tie = 3
st_tie = 4
up_tie = 5
dn_los = 6
st_los = 7
up_los = 8

def empirical_NGS(dat, SHUF=0, win=20, flip_human_AI=False, covariates=_AIconst._WTL, expt="EEG1", visit=None):
    _td, start_tm, end_tm, UA, cnstr, inp_meth, ini_percep, fin_percep = _rt.return_hnd_dat(dat, has_useragent=True, has_start_and_end_times=True, has_constructor=True, flip_human_AI=flip_human_AI, expt=expt, visit=visit)
    if _td is None:
        return None, None
    Tgame= _td.shape[0]
    ############  Several different dynamic conditional probabilities
    ############  We don't know what players look at, and how they think
    ############  about next move?  Do they think in terms of RPS, or
    ############  do they think in terms of upgrades, downgrades or stays?
    ############  Using a model that more closely matches the way they think
    ############  will probably better capture their behavior
    cprobs     = _N.zeros((SHUF+1, 9, Tgame-win))    # UDS | WTL
    cprobsRPS     = _N.zeros((SHUF+1, 9, Tgame-win)) # RPS | WTL
    cprobsDSURPS     = _N.zeros((SHUF+1, 9, Tgame-win)) # UDS | RPS
    cprobsSTSW = _N.zeros((SHUF+1, 6, Tgame-win))    #  Stay,Switch | WTL

    ############  Raw move game-by-game data
    all_tds = _N.empty((SHUF+1, _td.shape[0], _td.shape[1]), dtype=_N.int)
    for shf in range(SHUF+1):    ###########  allow randomly shuffling the data
        if shf > 0:
            inds = _N.arange(_td.shape[0])
            _N.random.shuffle(inds)
            td = _N.array(_td[inds])
        else:
            td = _td
        all_tds[shf] = td

        scores_wtl1 = _N.zeros(Tgame-1, dtype=_N.int)
        scores_rps0 = _N.zeros(Tgame-1, dtype=_N.int)
        scores_rps1 = _N.zeros(Tgame-1, dtype=_N.int)                
        scores_tr10 = _N.zeros(Tgame-1, dtype=_N.int)   #  transition

        ################################# wtl 1 steps back
        wins_m1 = _N.where(td[0:Tgame-1, 2] == 1)[0]
        ties_m1 = _N.where(td[0:Tgame-1, 2] == 0)[0]
        loss_m1 = _N.where(td[0:Tgame-1, 2] == -1)[0]
        ################################# rps 1 steps back
        R_m1 = _N.where(td[0:Tgame-1, 0] == 1)[0]
        S_m1 = _N.where(td[0:Tgame-1, 0] == 2)[0]
        P_m1 = _N.where(td[0:Tgame-1, 0] == 3)[0]

        scores_wtl1[wins_m1] = 2
        scores_wtl1[ties_m1] = 1
        scores_wtl1[loss_m1] = 0
        scores_rps1[R_m1] = 2
        scores_rps1[S_m1] = 1
        scores_rps1[P_m1] = 0
        
        ################################# tr from 1->0
        #####STAYS
        stays = _N.where(td[0:Tgame-1, 0] == td[1:Tgame, 0])[0]  
        scores_tr10[stays]   = 2
        #####DNGRAD        
        dngrd = _N.where(((td[0:Tgame-1, 0] == 1) & (td[1:Tgame, 0] == 2)) |
                         ((td[0:Tgame-1, 0] == 2) & (td[1:Tgame, 0] == 3)) |
                         ((td[0:Tgame-1, 0] == 3) & (td[1:Tgame, 0] == 1)))[0]
        scores_tr10[dngrd]   = 1
        #####UPGRAD        
        upgrd = _N.where(((td[0:Tgame-1, 0] == 1) & (td[1:Tgame, 0] == 3)) |
                         ((td[0:Tgame-1, 0] == 2) & (td[1:Tgame, 0] == 1)) |
                         ((td[0:Tgame-1, 0] == 3) & (td[1:Tgame, 0] == 2)))[0]
        scores_tr10[upgrd]   = 0
        #####ROCK
        rocks    = _N.where(td[1:Tgame, 0] == 1)[0]
        scores_rps0[rocks]   = 0
        #####SCISSOR
        scissors = _N.where(td[1:Tgame, 0] == 2)[0]
        scores_rps0[scissors]   = 1
        #####PAPER     
        papers   = _N.where(td[1:Tgame, 0] == 3)[0]
        scores_rps0[papers]   = 2
        #  UP | LOS  = scores 0
        #  UP | TIE  = scores 1
        #  UP | WIN  = scores 2
        #  DN | LOS  = scores 3
        #  DN | TIE  = scores 4
        #  DN | WIN  = scores 5
        #  ST | LOS  = scores 6
        #  ST | TIE  = scores 7
        #  ST | WIN  = scores 8

        scores       = scores_wtl1 + 3*scores_tr10  #  UDS | WTL
        scoresRPS    = scores_wtl1 + 3*scores_rps0  #  RPS | WTL
        scoresDSURPS = scores_rps1 + 3*scores_tr10  #  UDS | RPS
        scores_pr = scores_wtl1

        i = 0

        for i in range(0, Tgame-win):
            ######################################            
            n_win    = len(_N.where(scores_pr[i:i+win] == 2)[0])
            n_win_st = len(_N.where(scores[i:i+win] == 8)[0])
            n_win_dn = len(_N.where(scores[i:i+win] == 5)[0])
            n_win_up = len(_N.where(scores[i:i+win] == 2)[0])
            n_win_R  = len(_N.where(scoresRPS[i:i+win] == 8)[0])
            n_win_S  = len(_N.where(scoresRPS[i:i+win] == 5)[0])
            n_win_P  = len(_N.where(scoresRPS[i:i+win] == 2)[0])
            ######################################
            n_tie    = len(_N.where(scores_pr[i:i+win] == 1)[0])
            n_tie_st = len(_N.where(scores[i:i+win] == 7)[0])
            n_tie_dn = len(_N.where(scores[i:i+win] == 4)[0])
            n_tie_up = len(_N.where(scores[i:i+win] == 1)[0])
            n_tie_R  = len(_N.where(scoresRPS[i:i+win] == 7)[0])
            n_tie_S  = len(_N.where(scoresRPS[i:i+win] == 4)[0])
            n_tie_P  = len(_N.where(scoresRPS[i:i+win] == 1)[0])
            ######################################            
            n_los    = len(_N.where(scores_pr[i:i+win] == 0)[0])
            n_los_st = len(_N.where(scores[i:i+win] == 6)[0])
            n_los_dn = len(_N.where(scores[i:i+win] == 3)[0])
            n_los_up = len(_N.where(scores[i:i+win] == 0)[0])
            n_los_R  = len(_N.where(scoresRPS[i:i+win] == 6)[0])
            n_los_S  = len(_N.where(scoresRPS[i:i+win] == 3)[0])
            n_los_P  = len(_N.where(scoresRPS[i:i+win] == 0)[0])
            ######################################
            n_R      = len(_N.where(scores_rps1[i:i+win] == 2)[0])
            n_R_st = len(_N.where(scoresDSURPS[i:i+win] == 8)[0])
            n_R_dn = len(_N.where(scores[i:i+win] == 5)[0])
            n_R_up = len(_N.where(scores[i:i+win] == 2)[0])
            ######################################
            n_S      = len(_N.where(scores_rps1[i:i+win] == 1)[0])
            n_S_st = len(_N.where(scoresDSURPS[i:i+win] == 7)[0])
            n_S_dn = len(_N.where(scores[i:i+win] == 4)[0])
            n_S_up = len(_N.where(scores[i:i+win] == 1)[0])
            ######################################
            n_P      = len(_N.where(scores_rps1[i:i+win] == 0)[0])
            n_P_st = len(_N.where(scoresDSURPS[i:i+win] == 6)[0])
            n_P_dn = len(_N.where(scores[i:i+win] == 3)[0])
            n_P_up = len(_N.where(scores[i:i+win] == 0)[0])
            
            if n_win > 0:
                #cprobs[shf, 0, i] = n_win_st / n_win
                cprobs[shf, 0, i] = n_win_dn / n_win
                cprobs[shf, 1, i] = n_win_st / n_win
                cprobs[shf, 2, i] = n_win_up / n_win
                cprobsRPS[shf, 0, i] = n_win_R / n_win
                cprobsRPS[shf, 1, i] = n_win_S / n_win
                cprobsRPS[shf, 2, i] = n_win_P / n_win
                cprobsSTSW[shf, 0, i] = n_win_st / n_win
                cprobsSTSW[shf, 1, i] = (n_win_dn+n_win_up) / n_win
            else:
                cprobs[shf, 0, i] = cprobs[shf, 0, i-1]
                cprobs[shf, 1, i] = cprobs[shf, 1, i-1]
                cprobs[shf, 2, i] = cprobs[shf, 2, i-1]
                cprobsRPS[shf, 0, i] = cprobsRPS[shf, 0, i-1]
                cprobsRPS[shf, 1, i] = cprobsRPS[shf, 1, i-1]
                cprobsRPS[shf, 2, i] = cprobsRPS[shf, 2, i-1]
                cprobsSTSW[shf, 0, i] = cprobsSTSW[shf, 0, i-1] 
                cprobsSTSW[shf, 1, i] = cprobsSTSW[shf, 1, i-1] 
            if n_tie > 0:
                #cprobs[shf, 3, i] = n_tie_st / n_tie
                cprobs[shf, 3, i] = n_tie_dn / n_tie
                cprobs[shf, 4, i] = n_tie_st / n_tie
                cprobs[shf, 5, i] = n_tie_up / n_tie
                cprobsRPS[shf, 3, i] = n_tie_R / n_tie
                cprobsRPS[shf, 4, i] = n_tie_S / n_tie
                cprobsRPS[shf, 5, i] = n_tie_P / n_tie
                cprobsSTSW[shf, 2, i] = n_tie_st / n_tie
                cprobsSTSW[shf, 3, i] = (n_tie_dn+n_tie_up) / n_tie
            else:
                cprobs[shf, 3, i] = cprobs[shf, 3, i-1]
                cprobs[shf, 4, i] = cprobs[shf, 4, i-1]
                cprobs[shf, 5, i] = cprobs[shf, 5, i-1]
                cprobsRPS[shf, 3, i] = cprobsRPS[shf, 3, i-1]
                cprobsRPS[shf, 4, i] = cprobsRPS[shf, 4, i-1]
                cprobsRPS[shf, 5, i] = cprobsRPS[shf, 5, i-1]
                cprobsSTSW[shf, 2, i] = cprobsSTSW[shf, 2, i-1] 
                cprobsSTSW[shf, 3, i] = cprobsSTSW[shf, 3, i-1] 
            if n_los > 0:
                #cprobs[shf, 6, i] = n_los_st / n_los
                cprobs[shf, 6, i] = n_los_dn / n_los
                cprobs[shf, 7, i] = n_los_st / n_los                
                cprobs[shf, 8, i] = n_los_up / n_los
                cprobsRPS[shf, 6, i] = n_los_R / n_los
                cprobsRPS[shf, 7, i] = n_los_S / n_los
                cprobsRPS[shf, 8, i] = n_los_P / n_los
                cprobsSTSW[shf, 4, i] = n_los_st / n_los
                cprobsSTSW[shf, 5, i] = (n_los_dn+n_los_up) / n_los
            else:
                cprobs[shf, 6, i] = cprobs[shf, 6, i-1]
                cprobs[shf, 7, i] = cprobs[shf, 7, i-1]
                cprobs[shf, 8, i] = cprobs[shf, 8, i-1]
                cprobsRPS[shf, 6, i] = cprobsRPS[shf, 6, i-1]
                cprobsRPS[shf, 7, i] = cprobsRPS[shf, 7, i-1]
                cprobsRPS[shf, 8, i] = cprobsRPS[shf, 8, i-1]
                cprobsSTSW[shf, 4, i] = cprobsSTSW[shf, 4, i-1] 
                cprobsSTSW[shf, 5, i] = cprobsSTSW[shf, 5, i-1]
                ######################
            if n_R > 0:
                #cprobs[shf, 0, i] = n_win_st / n_win
                cprobsDSURPS[shf, 0, i] = n_R_dn / n_R
                cprobsDSURPS[shf, 1, i] = n_R_st / n_R
                cprobsDSURPS[shf, 2, i] = n_R_up / n_R
            else:
                cprobsDSURPS[shf, 0, i] = cprobsDSURPS[shf, 0, i-1]
                cprobsDSURPS[shf, 1, i] = cprobsDSURPS[shf, 1, i-1]
                cprobsDSURPS[shf, 2, i] = cprobsDSURPS[shf, 2, i-1]
            if n_S > 0:
                #cprobs[shf, 0, i] = n_win_st / n_win
                cprobsDSURPS[shf, 3, i] = n_S_dn / n_S
                cprobsDSURPS[shf, 4, i] = n_S_st / n_S
                cprobsDSURPS[shf, 5, i] = n_S_up / n_S
            else:
                cprobsDSURPS[shf, 3, i] = cprobsDSURPS[shf, 3, i-1]
                cprobsDSURPS[shf, 4, i] = cprobsDSURPS[shf, 4, i-1]
                cprobsDSURPS[shf, 5, i] = cprobsDSURPS[shf, 5, i-1]
            if n_P > 0:
                #cprobs[shf, 0, i] = n_win_st / n_win
                cprobsDSURPS[shf, 6, i] = n_P_dn / n_P
                cprobsDSURPS[shf, 7, i] = n_P_st / n_P
                cprobsDSURPS[shf, 8, i] = n_P_up / n_P
            else:
                cprobsDSURPS[shf, 6, i] = cprobsDSURPS[shf, 6, i-1]
                cprobsDSURPS[shf, 7, i] = cprobsDSURPS[shf, 7, i-1]
                cprobsDSURPS[shf, 8, i] = cprobsDSURPS[shf, 8, i-1]
                
    return cprobs, cprobsRPS, cprobsDSURPS, cprobsSTSW, all_tds, Tgame

#  down | win, stay | win, up | win
#  down | tie, stay | tie, up | tie
#  down | los, stay | los, up | los
def CRs(dat, expt="EEG1", visit=None, block=1, hnd_dat=None):
    if hnd_dat is None:
        td, start_time, end_time, UA, cnstr            = _rt.return_hnd_dat(dat, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=visit, expt=expt, block=block)        
        #td, start_tm, end_tm = _rt.return_hnd_dat(dat, has_useragent=True, has_start_and_end_times=True, has_constructor=True, flip_human_AI=False)
    else:
        td = hnd_dat
    Tgame= td.shape[0]
    CRs = _N.ones(Tgame-1, dtype=_N.int) * -1

    for g in range(Tgame-1):
        if td[g, 0] == td[g+1, 0]:   #  STAY
            if (td[g,2] == 1):
                CRs[g] = st_win
            elif (td[g,2] == 0):
                CRs[g] = st_tie
            elif (td[g,2] == -1):
                CRs[g] = st_los
        ##  from here, we're going to work with R=1, S=2, P=3
        #  1->2  R to S
        elif (((td[g, 0] == 1) and (td[g+1, 0] == 2)) or   #  DOWN
              ((td[g, 0] == 2) and (td[g+1, 0] == 3)) or
              ((td[g, 0] == 3) and (td[g+1, 0] == 1))):
            if (td[g,2] == 1):
                CRs[g] = dn_win
            elif (td[g,2] == 0):
                CRs[g] = dn_tie
            elif (td[g,2] == -1):
                CRs[g] = dn_los
        elif (((td[g, 0] == 1) and (td[g+1, 0] == 3)) or    #  UP
              ((td[g, 0] == 2) and (td[g+1, 0] == 1)) or
              ((td[g, 0] == 3) and (td[g+1, 0] == 2))):
            if (td[g,2] == 1):
                CRs[g] = up_win
            elif (td[g,2] == 0):
                CRs[g] = up_tie
            elif (td[g,2] == -1):
                CRs[g] = up_los
    return CRs

def marginalCR(hnd_dat):
    cr_games = CRs(None, hnd_dat=hnd_dat)
    margCR = _N.zeros((3, 3))
    margCR[0, 0] = len(_N.where(cr_games == dn_win)[0])
    margCR[0, 1] = len(_N.where(cr_games == st_win)[0])
    margCR[0, 2] = len(_N.where(cr_games == up_win)[0])
    margCR[1, 0] = len(_N.where(cr_games == dn_tie)[0])
    margCR[1, 1] = len(_N.where(cr_games == st_tie)[0])
    margCR[1, 2] = len(_N.where(cr_games == up_tie)[0])
    margCR[2, 0] = len(_N.where(cr_games == dn_los)[0])
    margCR[2, 1] = len(_N.where(cr_games == st_los)[0])
    margCR[2, 2] = len(_N.where(cr_games == up_los)[0])
    margCR[0] /= _N.sum(margCR[0])
    margCR[1] /= _N.sum(margCR[1])
    margCR[2] /= _N.sum(margCR[2])    
    return margCR
    
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
    
