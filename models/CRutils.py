import numpy as _N

def get_dbehv(prob_mvs, gk, cond=_N.array([0, 1, 2]), equalize=False):
    ab_d_prob_mvs = _N.abs(_N.diff(prob_mvs, axis=2))  #  time derivative
    if equalize:
        std_r = _N.std(ab_d_prob_mvs, axis=2).reshape(3, 3, 1)
        ab_d_prob_mvs /= std_r
    behv = _N.sum(_N.sum(ab_d_prob_mvs[cond], axis=1), axis=0)  #  1-D timeseries
    _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    if gk is not None:
        return _N.convolve(_dbehv, gk, mode="same")
    else:
        return _dbehv

def get_dbehv_combined(prob_mvs_list, gk, cond=_N.array([0, 1, 2]), equalize=False):
    n_diff_repr = len(prob_mvs_list)
    L = prob_mvs_list[0].shape[2]

    all_prob_mvs = _N.empty((9*n_diff_repr, prob_mvs_list[0].shape[2]))

    for nr in range(n_diff_repr):
        all_prob_mvs[nr*9:(nr+1)*9] =  prob_mvs_list[nr].reshape(9, L)
    ab_d_prob_mvs = _N.abs(_N.diff(all_prob_mvs, axis=1))  #  time derivative
    if equalize:
        std_r = _N.std(ab_d_prob_mvs, axis=1).reshape(9*n_diff_repr, 1)
        ab_d_prob_mvs /= std_r
    behv = _N.sum(ab_d_prob_mvs, axis=0)  #  1-D timeseries
    _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    if gk is not None:
        return _N.convolve(_dbehv, gk, mode="same")
    else:
        return _dbehv

def get_dbehv_combined_choose_(prob_mvs, gk, equalize=False):
    n_diff_repr = len(prob_mvs)   #  list of prob_mvs for each representation
    L = prob_mvs[0].shape[2]
    
    l_all_prob_mvs = []#_N.empty((9*n_diff_repr, prob_mvs_list[0].shape[2]))

    SHUFFLES = prob_mvs[0].shape[0]-1
    chng_pms = 0
    for nr in range(n_diff_repr):
        #l_bigchgs = []
        s = _N.std(prob_mvs[nr], axis=2)   #  std of move prob over all games
        std0 = _N.std(s[1:], axis=0)       #  std of SHUFFLED 
        m0   = _N.mean(s[1:], axis=0)
        z0   = (s[0] - m0) / std0
        bigchgs  = _N.where(z0 > 1.)[0]
        # for i in range(9):        
        #     ths = _N.where(s[0, i] > s[1:, i])[0]
        #     if len(ths) > int(SHUFFLES*0.95):
        #         l_bigchgs.append(i)
        # bigchgs = _N.array(l_bigchgs)
        chng_pms += len(bigchgs)

        for ib in bigchgs:
            l_all_prob_mvs.append(_N.array(prob_mvs[nr][0, ib]))
    if chng_pms == 0:
        return 0, None
    all_prob_mvs = _N.array(l_all_prob_mvs)
    ab_d_prob_mvs = _N.abs(_N.diff(all_prob_mvs, axis=1))  #  time derivative
    # if equalize:
    #     std_r = _N.std(ab_d_prob_mvs, axis=1).reshape(9*n_diff_repr, 1)
    #     ab_d_prob_mvs /= std_r
    behv = _N.sum(ab_d_prob_mvs, axis=0)  #  1-D timeseries
    _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    if gk is not None:
        return chng_pms, _N.convolve(_dbehv, gk, mode="same")
    else:
        return chng_pms, _dbehv
    
def entropy3(sig, N):
    cube = _N.zeros((N, N, N))   #  W T L conditions or
    iN   = 1./N

    #print(sig.shape[0])
    for i in range(sig.shape[0]):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        iz = int(sig[i, 2]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        iz = iz if iz < N else N-1
        cube[ix, iy, iz] += 1

    #print(cube)
    entropy  = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                p_ijk = cube[i, j, k] / len(sig)
                if p_ijk > 0:
                    entropy += -p_ijk * _N.log(p_ijk)
    return entropy

def entropy2(sig, N):
    #  calculate the entropy
    square = _N.zeros((N, N))
    iN   = 1./N
    for i in range(len(sig)):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        square[ix, iy] += 1

    entropy  = 0
    for i in range(N):
        for j in range(N):
                p_ij = square[i, j] / len(sig)
                if p_ij > 0:
                    entropy += -p_ij * _N.log(p_ij)
    return entropy
