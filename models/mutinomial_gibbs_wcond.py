import numpy as _N
import scipy.stats as _ss
import pyPG as lw
import LOSTtmp.kfARlib1c as _kfar



capped = 0
l_capped = []


def sampleAR1_and_offset(it, Tm1, off_mu, off_sig2, vrnc, vrncL, B_n, offset, \
                         kappa, ws, q2_B_n, a_F0, b_F0, a_q2, B_q2, \
                         px, pV, fx, fV, K, random_walk):
    global capped, l_capped
    offset_mu = kappa / ws - B_n
    mu_w  = _N.sum(offset_mu*ws) / _N.sum(ws)   #  from likelihood
    mu  = (mu_w*off_sig2 + off_mu*vrncL) / (off_sig2 + vrncL) # lklhd & prior
    #offset[:] = mu + _N.sqrt(vrnc)*_N.random.randn()
    offset = mu + _N.sqrt(vrnc)*_N.random.randn()

    F0AA = _N.dot(B_n[0:-1], B_n[0:-1])
    F0BB = _N.dot(B_n[0:-1], B_n[1:])

    F0_B_n = 1
    if not random_walk:
        F0std= _N.sqrt(q2_B_n/F0AA)
        F0a, F0b  = (a_F0 - F0BB/F0AA) / F0std, (b_F0 - F0BB/F0AA) / F0std
        F0_B_n=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)

    #   sample q2
    a = a_q2 + 0.5*Tm1  #  N + 1 - 1
    #a = 0.5*Nm1  #  N + 1 - 1
    rsd_stp = B_n[1:] - F0_B_n*B_n[0:-1]
    BB = B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
    #BB = 0.5 * _N.dot(rsd_stp, rsd_stp)
    q2_B_n = _ss.invgamma.rvs(a, scale=BB)

    y             = kappa/ws - offset
    Rv = 1. / ws

    _kfar.armdl_FFBS_1itr_singletrial(Tm1, y, Rv, F0_B_n, q2_B_n,
                                      fx, fV, px, pV, B_n, K)    
    near1 = _N.where(B_n > 300)[0]
    if len(near1) > 0:
        B_n[near1] = 300   #  cap it   #  prevents overflow in exp
        capped += 1
        l_capped.append(it)
        
    return offset, F0_B_n, q2_B_n


def sampleAR_and_offset_conditional(it, Tm1, vrnc, vrncL, \
                        B_n, v_n, offset, \
                        B_1_n, v_1, offset_1, \
                        B_2_n, v_2, offset_2, \
                        kappa, ws, q2_B_n, a_F0, b_F0, a_q2, B_q2, px, pV, fx, fV, K, random_walk):
    global capped, l_capped
    offset_mu = (kappa / ws - (B_1_n+offset_1)*v_1 - (B_2_n+offset_2)*v_2)/v_n - B_n
    mu_w  = _N.sum(offset_mu*ws) / _N.sum(ws)   #  from likelihood
    mu  = (mu_w*off_sig2 + off_mu*vrncL) / (off_sig2 + vrncL) # lklhd & prior
    offset[:] = mu + _N.sqrt(vrnc)*_N.random.randn()

    F0AA = _N.dot(B_n[0:-1], B_n[0:-1])
    F0BB = _N.dot(B_n[0:-1], B_n[1:])

    F0_B_n = 1
    if not random_walk:
        F0std= _N.sqrt(q2_B_n/F0AA)
        F0a, F0b  = (a_F0 - F0BB/F0AA) / F0std, (b_F0 - F0BB/F0AA) / F0std
        F0_B_n=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)

    #   sample q2
    a = a_q2 + 0.5*Tm1  #  N + 1 - 1
    #a = 0.5*Nm1  #  N + 1 - 1
    rsd_stp = B_n[1:] - F0_B_n*B_n[0:-1]
    BB = B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
    #BB = 0.5 * _N.dot(rsd_stp, rsd_stp)
    q2_B_n = _ss.invgamma.rvs(a, scale=BB)

    y             = (kappa/ws - (B_1_n+offset_1)*v_1 - (B_2_n+offset_2)*v_2)/v_n - offset
    Rv = 1. / ws

    _kfar.armdl_FFBS_1itr_singletrial(Tm1, y, Rv, F0_B_n, q2_B_n,
                                      fx, fV, px, pV, B_n, K)    
    near1 = _N.where(B_n > 100)[0]
    if len(near1) > 0:
        B_n[near1] = 100   #  cap it   #  prevents overflow in exp
        capped += 1
        l_capped.append(it)
        
    return offset, F0_B_n, q2_B_n

class multinomial_gibbs:
    Tm1     = None
    K       = None
    N       = None
    kappa   = None
    N_vec   = None

    def __init__(self, N, K, N_vec, kappa, Tm1):
        oo = self
        oo.N = N
        oo.K = K
        oo.Tm1          = Tm1
        oo.kappa        = kappa
        oo.N_vec        = N_vec
        ###  FFBS variables

    def sample_posterior(self, ITER, a_F0, b_F0, a_q2, B_q2, smp_Bns, smp_offsets, smp_F0s, smp_q2s, off_mu=0, off_sig2=0.4, random_walk=False):
        w1_px        = _N.random.randn(Tm1)
        w1_pV        = _N.ones(Tm1)*0.2
        w1_fx        = _N.zeros(Tm1)
        w1_fV        = _N.ones(Tm1)*0.1
        w2_px        = _N.random.randn(Tm1)
        w2_pV        = _N.random.rand(Tm1)
        w2_fx        = _N.zeros(Tm1)
        w2_fV        = _N.ones(Tm1)*0.1

        t1_px        = _N.random.randn(Tm1)
        t1_pV        = _N.ones(Tm1)*0.2
        t1_fx        = _N.zeros(Tm1)
        t1_fV        = _N.ones(Tm1)*0.1
        t2_px        = _N.random.randn(Tm1)
        t2_pV        = _N.random.rand(Tm1)
        t2_fx        = _N.zeros(Tm1)
        t2_fV        = _N.ones(Tm1)*0.1

        l1_px        = _N.random.randn(Tm1)
        l1_pV        = _N.ones(Tm1)*0.2
        l1_fx        = _N.zeros(Tm1)
        l1_fV        = _N.ones(Tm1)*0.1
        l2_px        = _N.random.randn(Tm1)
        l2_pV        = _N.random.rand(Tm1)
        l2_fx        = _N.zeros(Tm1)
        l2_fV        = _N.ones(Tm1)*0.1

        w1_K         = _N.empty(Tm1)
        t1_K         = _N.empty(Tm1)
        l1_K         = _N.empty(Tm1)
        w2_K         = _N.empty(Tm1)
        t2_K         = _N.empty(Tm1)
        l2_K         = _N.empty(Tm1)

        o_w1        = _N.random.randn(Tm1)   #  start at 0 + u
        o_t1        = _N.random.randn(Tm1)   #  start at 0 + u
        o_l1        = _N.random.randn(Tm1)   #  start at 0 + u
        o_w2        = _N.random.randn(Tm1)   #  start at 0 + u
        o_t2        = _N.random.randn(Tm1)   #  start at 0 + u
        o_l2        = _N.random.randn(Tm1)   #  start at 0 + u

        B1wn        = _N.random.randn(Tm1)   #  start at 0 + u
        B1tn        = _N.random.randn(Tm1)   #  start at 0 + u
        B1ln        = _N.random.randn(Tm1)   #  start at 0 + u
        B2wn        = _N.random.randn(Tm1)   #  start at 0 + u
        B2tn        = _N.random.randn(Tm1)   #  start at 0 + u
        B2ln        = _N.random.randn(Tm1)   #  start at 0 + u

        q2_Bw1 = 1.
        q2_Bt1 = 1.
        q2_Bl1 = 1.
        q2_Bw2 = 1.
        q2_Bt2 = 1.
        q2_Bl2 = 1.
        F0_Bw1 = 0
        F0_Bt1 = 0
        F0_Bl1 = 0
        F0_Bw2 = 0
        F0_Bt2 = 0
        F0_Bl2 = 0

        ws1  = _N.random.rand(Tm1)
        ws2  = _N.random.rand(Tm1)

        zr2  = _N.where(N_vec[:, 1] == 0)[0]     #  dat where N_2 == 0  (only 1 PG var)
        nzr2 = _N.where(N_vec[:, 1] == 1)[0]   

        W_n = _N.zeros(Tm1, dtype=_N.int)
        T_n = _N.zeros(Tm1, dtype=_N.int)
        L_n = _N.zeros(Tm1, dtype=_N.int)

        if covariates == _WTL:
            win= _N.where(hnd_dat[0:Tm1, 2] == 1)[0]
            tie= _N.where(hnd_dat[0:Tm1, 2] == 0)[0]
            los= _N.where(hnd_dat[0:Tm1, 2] == -1)[0]
        elif covariates == _RPS:
            win= _N.where(hnd_dat[0:Tm1, 0] == 1)[0]   
            tie= _N.where(hnd_dat[0:Tm1, 0] == 2)[0]
            los= _N.where(hnd_dat[0:Tm1, 0] == 3)[0]    
        W_n[win] =  1
        T_n[win] = -1
        L_n[win] = -1
        #
        W_n[tie] = -1
        T_n[tie] = 1
        L_n[tie] = -1
        #
        W_n[los] = -1
        T_n[los] = -1
        L_n[los] = -1

        K     = 3
        N_vec = _N.zeros((Tm1, K), dtype=_N.int)     #  The N vector
        N     = 1
        kappa   = _N.empty((Tm1, K))

        #  hand n, n-1    --->  obs of 
        #  1 < 2   R < P    1%3 < 2%3   1 < 2
        #  2 < 3   P < S    2%3 < 3%3   2 < 3
        #  3 < 1   S < R    3%3 < 1%3   0 < 1
        if signal == _RELATIVE_LAST_ME:
            col_n0 = 0    #  current
            col_n1 = 0    #  previous
        elif signal == _RELATIVE_LAST_AI:
            col_n0 = 0    #  did player copy AI's last move
            col_n1 = 1    #  or did player go to move that beat (loses) to the last AI
        elif signal == _RELATIVE_LAST_OU:
            col_n0 = 0    #  did player copy AI's last move
            col_n1 = 2    #  or did player go to move that beat (loses) to the last AI

        #  RELATIVE LAST ME  -  stay or switch (2 types of switch)
        #  RELATIVE LAST AI  -  copy AI or 
        y_vec = _N.zeros((Tm1, 3), dtype=_N.int)
        y     = _N.zeros(Tm1, dtype=_N.int)   #  indices of the random var

        for n in range(1, Tobs):
            if signal != _RELATIVE_LAST_OU:
                if (hnd_dat[n, col_n0] == hnd_dat[n-1, col_n1]):
                    y[n-1] = 0  #  Goo, choki, paa   goo->choki
                                #   choki->paa
                    y_vec[n-1, 0] = 1    #  [1, 0, 0]    stay
                elif ((hnd_dat[n, col_n0] == 1) and (hnd_dat[n-1, col_n1] == 3)) or \
                     ((hnd_dat[n, col_n0] == 2) and (hnd_dat[n-1, col_n1] == 1)) or \
                     ((hnd_dat[n, col_n0] == 3) and (hnd_dat[n-1, col_n1] == 2)):
                    y[n-1] = -1
                    y_vec[n-1, 1] = 1    #  [0, 1, 0]    choose weaker
                elif ((hnd_dat[n, col_n0] == 1) and (hnd_dat[n-1, col_n1] == 2)) or \
                     ((hnd_dat[n, col_n0] == 2) and (hnd_dat[n-1, col_n1] == 3)) or \
                     ((hnd_dat[n, col_n0] == 3) and (hnd_dat[n-1, col_n1] == 1)):
                    y[n-1] = 1
                    y_vec[n-1, 2] = 1    #  [0, 0, 1]    choose stronger
            else:
                if (hnd_dat[n, col_n1] == 1):    # win
                    y[n-1] = 1
                    y_vec[n-1, 0] = 1    #  [1, 0, 0]    stay
                elif (hnd_dat[n, col_n1] == 0):    # tie
                    y[n-1] = 0
                    y_vec[n-1, 1] = 1    #  [0, 1, 0]    stay
                elif (hnd_dat[n, col_n1] == -1):    # los
                    y[n-1] = -1
                    y_vec[n-1, 2] = 1    #  [0, 0, 1]    stay

        for n in range(Tm1):
          N_vec[n, 0] = 1
          for k in range(1, K):
            N_vec[n, k] = N - _N.sum(y_vec[n, 0:k])
          for k in range(K):
            kappa[n, k] = y_vec[n, k] - 0.5 * N_vec[n, k]

        print("3")
        smp_offsets = _N.empty((6, ITER, Tm1))
        smp_Bns     = _N.empty((6, ITER, Tm1))
        smp_q2s     = _N.empty((ITER, 6))
        smp_F0s     = _N.empty((ITER, 6))

        #_d.copyData(_N.empty(N), _N.empty(N), onetrial=True)   #  dummy data copied


        off_sig2 = 0.4
        off_mu   = 0

        o_w1[:] = 0
        o_t1[:] = 0
        o_l1[:] = 0
        o_w2[:] = 0
        o_t2[:] = 0
        o_l2[:] = 0

        do_order = _N.arange(6)
        for it in range(ITER):
            if it % 1000 == 0:
                print("%(it)d   capped %(cp)d" % {"it" : it, "cp" : capped})

            vrncL1 = 1/_N.sum(ws1)   
            vrnc1  = (off_sig2*vrncL1) / (off_sig2 + vrncL1)
            vrncL2 = 1/_N.sum(ws2)
            vrnc2  = (off_sig2*vrncL2) / (off_sig2 + vrncL2)

            _N.random.shuffle(do_order)

            for di in do_order:
                #################
                if di == 0:
                    o_w1, F0_Bw1, q2_Bw1 = sampleAR_and_offset(it, Tm1, vrnc1, vrncL1,
                                                                     B1wn, W_n, o_w1, 
                                                                     B1tn, T_n, o_t1, 
                                                                     B1ln, L_n, o_l1, 
                                                                     kappa[:, 0], ws1, q2_Bw1, a_F0, b_F0, a_q2,
                                                                     B_q2, w1_px, w1_pV, w1_fx, w1_fV, w1_K, random_walk)
                    smp_offsets[0, it] = o_w1[0]
                elif di == 1:
                    #################
                    o_t1, F0_Bt1, q2_Bt1 = sampleAR_and_offset(it, Tm1, vrnc1, vrncL1,
                                                                     B1tn, T_n, o_t1, 
                                                                     B1ln, L_n, o_l1, 
                                                                     B1wn, W_n, o_w1, 
                                                                     kappa[:, 0], ws1, q2_Bt1, a_F0, b_F0, a_q2,
                                                                     B_q2, t1_px, t1_pV, t1_fx, t1_fV, t1_K, random_walk)
                    smp_offsets[1, it] = o_t1[0]
                elif di == 2:
                    #################
                    o_l1, F0_Bl1, q2_Bl1 = sampleAR_and_offset(it, Tm1, vrnc1, vrncL1,
                                                                     B1ln, L_n, o_l1, 
                                                                     B1wn, W_n, o_w1, 
                                                                     B1tn, T_n, o_t1, 
                                                                     kappa[:, 0], ws1, q2_Bl1, a_F0, b_F0, a_q2, 
                                                                     B_q2, l1_px, l1_pV, l1_fx, l1_fV, l1_K, random_walk)
                    smp_offsets[2, it] = o_l1[0]
                elif di == 3:
                    #################
                    o_w2, F0_Bw2, q2_Bw2 = sampleAR_and_offset(it, Tm1, vrnc2, vrncL2,
                                                                     B2wn, W_n, o_w2, 
                                                                     B2tn, T_n, o_t2, 
                                                                     B2ln, L_n, o_l2, 
                                                                     kappa[:, 1], ws2, q2_Bw2, a_F0, b_F0, a_q2,
                                                                     B_q2, w2_px, w2_pV, w2_fx, w2_fV, w2_K, random_walk)
                    smp_offsets[3, it] = o_w2[0]
                elif di == 4:
                    #################
                    o_t2, F0_Bt2, q2_Bt2 = sampleAR_and_offset(it, Tm1, vrnc2, vrncL2,
                                                                    B2tn, T_n, o_t2, 
                                                                    B2ln, L_n, o_l2, 
                                                                    B2wn, W_n, o_w2, 
                                                                    kappa[:, 1], ws2, q2_Bt2, a_F0, b_F0, a_q2,
                                                                    B_q2, t2_px, t2_pV, t2_fx, t2_fV, t2_K, random_walk)
                    smp_offsets[4, it] = o_t2[0]
                elif di == 5:
                    #################
                    o_l2, F0_Bl2, q2_Bl2 = sampleAR_and_offset(it, Tm1, vrnc2, vrncL2,
                                                                     B2ln, L_n, o_l2, 
                                                                     B2wn, W_n, o_w2, 
                                                                     B2tn, T_n, o_t2, 
                                                                     kappa[:, 1], ws2, q2_Bl2, a_F0, b_F0, a_q2, 
                                                                     B_q2, l2_px, l2_pV, l2_fx, l2_fV, l2_K, random_walk)
                    smp_offsets[5, it] = o_l2[0]

            smp_Bns[0, it] = B1wn
            smp_Bns[1, it] = B1tn
            smp_Bns[2, it] = B1ln    
            smp_Bns[3, it] = B2wn
            smp_Bns[4, it] = B2tn
            smp_Bns[5, it] = B2ln    

            smp_q2s[it]  = q2_Bw1, q2_Bt1, q2_Bl1, q2_Bw2, q2_Bt2, q2_Bl2
            #if random_walk:
            #    F0_Bw1 = F0_Bt1 = F0_Bl1 = F0_Bw2 = F0_Bt2 = F0_Bl2 = 1
            smp_F0s[it]  = F0_Bw1, F0_Bt1, F0_Bl1, F0_Bw2, F0_Bt2, F0_Bl2

            lw.rpg_devroye(N_vec[:, 0], o_w1*W_n + o_t1*T_n + o_l1*L_n, out=ws1)
            lw.rpg_devroye(N_vec[:, 1], o_w2*W_n + o_t2*T_n + o_l2*L_n, out=ws2)

            ws2[zr2] = 1e-20#1e-20


        pklme = {}
        smp_every =  50
        pklme["smp_Bns"] = smp_Bns[:, ::smp_every]
        pklme["smp_q2s"] = smp_q2s[::smp_every]
        pklme["smp_F0s"] = smp_F0s[::smp_every]
        pklme["smp_offsets"] = smp_offsets[:, ::smp_every]
        pklme["smp_every"] = smp_every
        pklme["Wn"] = W_n
        pklme["Tn"] = T_n
        pklme["Ln"] = L_n
        pklme["hnd_dat"]   = hnd_dat
        pklme["y_vec"]     = y_vec
        pklme["N_vec"]     = N_vec
        pklme["a_q2"]      = a_q2
        pklme["B_q2"]      = B_q2
        pklme["l_capped"]      = l_capped
        dmp = open("%(dir)s/%(rel)s,%(cov)s%(ran)s2.dmp" % {"rel" : ssig, "cov" : scov, "ran" : sran, "dir" : out_dir}, "wb")
        pickle.dump(pklme, dmp, -1)
        dmp.close()
        print("capped:  %d" % capped)
