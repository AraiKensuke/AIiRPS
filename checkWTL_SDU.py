import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
#

PCS = 5

def rand3(type):
    if type == 0:
        return _N.random.rand(3)
    elif type == 1:
        ths = _N.random.rand(4)
        ths3= _N.array([ths[0], ths[1], ths[2]+ths[3]])
        _N.random.shuffle(ths3)
        return ths3
    elif type == 2:
        ths = _N.random.rand(5)
        ths4= _N.array([ths[0], ths[1], ths[2]+ths[3] + ths[4]])
        _N.random.shuffle(ths4)
        return ths4
    elif type == 3:
        ths = _N.random.rand(6)
        ths5= _N.array([ths[0], ths[1], ths[2]+ths[3] + ths[4] + ths[5]])
        _N.random.shuffle(ths5)
        return ths5
    
def entropy3(_sig, N, repeat=None, nz=0):
    cube = _N.zeros((N, N, N))   #  W T L conditions or
    iN   = 1./N

    #print(sig.shape[0])

    if repeat is not None:
        newlen = _sig.shape[0]*repeat
        sig = _N.empty((newlen, 3))
        sig[:, 0] = _N.repeat(_sig[:, 0], repeat) + nz*_N.random.randn(newlen)
        sig[:, 1] = _N.repeat(_sig[:, 1], repeat) + nz*_N.random.randn(newlen)
        sig[:, 2] = _N.repeat(_sig[:, 2], repeat) + nz*_N.random.randn(newlen)
    else:
        sig = _sig
    
    for i in range(sig.shape[0]):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        iz = int(sig[i, 2]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        iz = iz if iz < N else N-1
        cube[ix, iy, iz] += 1

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

REPS   = 300
TRLS   = 300
rand3typ=2

all_prob_mvs = _N.empty((REPS, 3, 3, TRLS))
entropyD = _N.empty(REPS)   #  how different are D across WTL conditions
entropyS = _N.empty(REPS)
entropyU = _N.empty(REPS)
entropyW2 = _N.empty(REPS)   #  
entropyT2 = _N.empty(REPS)
entropyL2 = _N.empty(REPS)
entropyW = _N.empty(REPS)   #  
entropyT = _N.empty(REPS)
entropyL = _N.empty(REPS)

#  if p(UP) and p(DN) are similar

aabbs = _N.empty((REPS, TRLS, 3, 2))
for rep in range(REPS):
    prob_mvs = _N.empty((3, 3, TRLS))
    X = _N.random.rand()
    Y = _N.random.rand()    
    if _N.random.rand() < 0.5:
        r = 0.8*(X-0.5)
    else:
        r = 0.8*(0.5-X)
    #r=0   #  r=0, p(U) p(D) are always the same
    #r>0   #  p(U) p(D) are always the same
    T0 = _N.random.randint(6, 10)
    T = int(T0 + 4*_N.random.rand())
    tr = 0
    while tr < TRLS:
        tuntil = TRLS if (tr + T) > TRLS else tr + T
        if tr == 0:
            cmpsChng = _N.array([0, 1, 2])
        else:
            cmpsChng = _N.array([_N.random.randint(0, 3)])
        for wtl in cmpsChng:
            pDSU = rand3(rand3typ)
            # pDSU = pDSU / _N.sum(pDSU)
            # DU   = pDSU[0] + pDSU[2]
            # a    = (DU/2)*(1+r)   #  unbalance of 
            # b    = (DU/2)*(1-r)
            # pDSU[0] = a
            # pDSU[2] = b
            # aabbs[rep, tr, wtl] = a, b            
            # pDSU /= _N.sum(pDSU)

            
            prob_mvs[wtl, :, tr:] = pDSU.reshape((3, 1))
        tr = tuntil
        T = int(T0 + 4*_N.random.rand())        
            
    all_prob_mvs[rep] = prob_mvs
        
    entsDSU = _N.array([entropy3(prob_mvs[:, 0].T, PCS),
                        entropy3(prob_mvs[:, 1].T, PCS),
                        entropy3(prob_mvs[:, 2].T, PCS)])

    pW_stsw = _N.array([prob_mvs[0, 0] + prob_mvs[0, 2], prob_mvs[0, 1]])
    pT_stsw = _N.array([prob_mvs[1, 0] + prob_mvs[1, 2], prob_mvs[1, 1]])
    pL_stsw = _N.array([prob_mvs[2, 0] + prob_mvs[2, 2], prob_mvs[2, 1]])
    entsWTL2 = _N.array([entropy2(pW_stsw.T, PCS), entropy2(pT_stsw.T, PCS), entropy2(pL_stsw.T, PCS)])
    entsWTL3 = _N.array([entropy3(prob_mvs[0].T, PCS),
                         entropy3(prob_mvs[1].T, PCS),
                         entropy3(prob_mvs[2].T, PCS)])


    entropyD[rep] = entsDSU[0]
    entropyS[rep] = entsDSU[1]
    entropyU[rep] = entsDSU[2]
                        
    entropyW2[rep] = entsWTL2[0]
    entropyT2[rep] = entsWTL2[1]
    entropyL2[rep] = entsWTL2[2]

    entropyW[rep] = entsWTL3[0]
    entropyT[rep] = entsWTL3[1]
    entropyL[rep] = entsWTL3[2]


###  The idea is that when people make a decision whether to stay, up or down,
###  the decision to stay or switch might be a lot more calculated than
###  the decision to UP or DN.
###  Including UP and DN separately in entropy might introduce variability
###  that isn't reflective of cognitive thought, but noise

fig = _plt.figure(figsize=(10, 10))
if1 = -1
for sfeat1 in ["entropyS", "entropyD", "entropyU"]:
    feat1 = eval(sfeat1)
    if1 += 1
    if2 = -1
    for sfeat2 in ["entropyW2", "entropyT2", "entropyL2"]:
        feat2 = eval(sfeat2)
        if2 += 1
        fig.add_subplot(3, 3, if1*3 + if2 + 1)
        pc, pv = _ss.pearsonr(feat1, feat2)
        _plt.title("%(pc).2f  %(pv).1e" % {"pc" : pc, "pv" : pv}) 
        _plt.scatter(feat1, feat2, color="black", s=5)
        if if2 == 0:
            _plt.ylabel(sfeat1, fontsize=18)
        if if1 == 2:
            _plt.xlabel(sfeat2, fontsize=18)
        _plt.xticks(fontsize=13)
        _plt.yticks(fontsize=13)        
fig.subplots_adjust(wspace=0.25, hspace=0.25)
_plt.savefig("corr_btwn_ent_comps_sim2")

fig = _plt.figure(figsize=(10, 10))
if1 = -1
for sfeat1 in ["entropyS", "entropyD", "entropyU"]:
    feat1 = eval(sfeat1)
    if1 += 1
    if2 = -1
    for sfeat2 in ["entropyW", "entropyT", "entropyL"]:
        feat2 = eval(sfeat2)
        if2 += 1
        fig.add_subplot(3, 3, if1*3 + if2 + 1)
        pc, pv = _ss.pearsonr(feat1, feat2)
        _plt.title("%(pc).2f  %(pv).1e" % {"pc" : pc, "pv" : pv}) 
        _plt.scatter(feat1, feat2, color="black", s=5)
        if if2 == 0:
            _plt.ylabel(sfeat1, fontsize=18)
        if if1 == 2:
            _plt.xlabel(sfeat2, fontsize=18)
        _plt.xticks(fontsize=13)
        _plt.yticks(fontsize=13)        
fig.subplots_adjust(wspace=0.25, hspace=0.25)
_plt.savefig("corr_btwn_ent_comps_sim")

print("--------")
print(_N.mean(entropyU))
print(_N.mean(entropyD))

# _plt.plot(all_prob_mvs[0, 0, 0], lw=3)
# _plt.plot(all_prob_mvs[0, 0, 1])
# _plt.plot(all_prob_mvs[0, 0, 2])
