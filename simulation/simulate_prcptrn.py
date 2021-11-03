import AIiRPS.simulation.prcptrnJS as _prcp
import numpy as _N

def recreate_percep_istate(hnd_dat, ini_percep_str, fin_percep_str):
    ini_as_arr = _N.array(ini_percep_str.split(" "), dtype=_N.float)
    fin_as_arr = _N.array(fin_percep_str.split(" "), dtype=_N.float)

    recr_prc_mvs = _N.zeros(hnd_dat.shape[0], dtype=_N.int)

    prc_N = 2
    weights_snap = _N.empty((3, hnd_dat.shape[0]+1, 3, 3, prc_N ))
    diff_tots     = _N.zeros(3)
    
    for init_pm in range(1, 4):
        prc = _prcp.perceptronJS(prc_N)
        prc.prc_weight[:, :, :] = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    prc.prc_weight[i, j, k] = ini_as_arr[(6*i) + 2*j + k]
        diff_tot = 0
        pm = init_pm
        for im in range(hnd_dat.shape[0]):  #  Goo choki paa
            m = hnd_dat[im, 0]
            weights_snap[init_pm-1, im] = prc.prc_weight
            pred_HP_move = prc.predict(pm)
            #  (1+1) % 3 + 1 = 3
            recr_prc_mvs[im] = (pred_HP_move+1) % 3 + 1
            #print("%(1)d  %(2)d" % {"1" : hnd_dat[im, 1], "2" : ((pred_HP_move+1) % 3 + 1)})
            diff_tot += _N.abs(hnd_dat[im, 1] - ((pred_HP_move+1) % 3 + 1))
            pm = m
        weights_snap[init_pm-1, hnd_dat.shape[0]] = prc.prc_weight            
        diff_tots[init_pm - 1] = diff_tot
    the0s = _N.where(diff_tots == _N.min(diff_tots))[0]

    return weights_snap, the0s[0]
        
