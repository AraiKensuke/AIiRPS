import scipy.stats as _ss
import pyPG as lw
import LOSTtmp.kfARlib1c as _kfar
import numpy as _N


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
        oo = self

        w1_px        = _N.random.randn(oo.Tm1)
        w1_pV        = _N.ones(oo.Tm1)*0.2
        w1_fx        = _N.zeros(oo.Tm1)
        w1_fV        = _N.ones(oo.Tm1)*0.1
        w2_px        = _N.random.randn(oo.Tm1)
        w2_pV        = _N.random.rand(oo.Tm1)
        w2_fx        = _N.zeros(oo.Tm1)
        w2_fV        = _N.ones(oo.Tm1)*0.1

        w1_K         = _N.empty(oo.Tm1)
        w2_K         = _N.empty(oo.Tm1)

        o_w1        = _N.random.randn(oo.Tm1)   #  start at 0 + u
        o_w2        = _N.random.randn(oo.Tm1)   #  start at 0 + u
        B1wn        = _N.random.randn(oo.Tm1)   #  start at 0 + u
        B2wn        = _N.random.randn(oo.Tm1)   #  start at 0 + u
        q2_Bw1 = 1.
        q2_Bw2 = 1.
        F0_Bw1 = 0
        F0_Bw2 = 0
        ws1  = _N.random.rand(oo.Tm1)
        ws2  = _N.random.rand(oo.Tm1)

        zr2  = _N.where(oo.N_vec[:, 1] == 0)[0]     #  dat where N_2 == 0  (only 1 PG var)
        nzr2 = _N.where(oo.N_vec[:, 1] == 1)[0]   

        do_order = _N.arange(2)

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
                    o_w1, F0_Bw1, q2_Bw1 \
                        = sampleAR1_and_offset(it, oo.Tm1, off_mu, off_sig2, 
                                               vrnc1, vrncL1,B1wn, o_w1, 
                                               oo.kappa[:, 0], ws1, q2_Bw1, 
                                               a_F0, b_F0, a_q2, B_q2, 
                                               w1_px, w1_pV, w1_fx, w1_fV, 
                                               w1_K, random_walk)
                    smp_offsets[0, it] = o_w1
                elif di == 1:
                    #################
                    o_w2, F0_Bw2, q2_Bw2 \
                        = sampleAR1_and_offset(it, oo.Tm1, off_mu, off_sig2, 
                                               vrnc2, vrncL2, B2wn, o_w2, 
                                               oo.kappa[:, 1], ws2, q2_Bw2, 
                                               a_F0, b_F0, a_q2, B_q2, 
                                               w2_px, w2_pV, w2_fx, w2_fV, 
                                               w2_K, random_walk)
                    smp_offsets[1, it] = o_w2

            smp_Bns[0, it] = B1wn
            smp_Bns[1, it] = B2wn

            #smp_q2s[it, 2*cond:2*cond+2]  = q2_Bw1, q2_Bw2y
            smp_q2s[it]  = q2_Bw1, q2_Bw2
            #if random_walk:
            #    F0_Bw1 = F0_Bt1 = F0_Bl1 = F0_Bw2 = F0_Bt2 = F0_Bl2 = 1
            smp_F0s[it]  = F0_Bw1, F0_Bw2

            lw.rpg_devroye(oo.N_vec[:, 0], B1wn+o_w1, out=ws1)
            lw.rpg_devroye(oo.N_vec[:, 1], B2wn+o_w2, out=ws2)

            ws2[zr2] = 1e-20#1e-20
