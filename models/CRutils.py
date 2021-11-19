import numpy as _N

def get_dbehv(prob_mvs, gk, equalize=False):
    ab_d_prob_mvs = _N.abs(_N.diff(prob_mvs, axis=2))  #  time derivative
    if equalize:
        std_r = _N.std(ab_d_prob_mvs, axis=2).reshape(3, 3, 1)
        ab_d_prob_mvs /= std_r
    behv = _N.sum(_N.sum(ab_d_prob_mvs, axis=1), axis=0)  #  1-D timeseries
    _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    if gk is not None:
        return _N.convolve(_dbehv, gk, mode="same")
    else:
        return _dbehv

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
