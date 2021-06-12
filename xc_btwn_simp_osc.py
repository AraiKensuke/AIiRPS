import numpy as _N
import matplotlib.pyplot as _plt
from LOST.kflib import createDataAR
from LOST.ARcfSmplFuncs import initF
import numpy.polynomial.polynomial as _Npp

#  what kind of cross correlation to expect if we have a binary signal with
#  an oscillatory autocorrelation, and a continuous signal?

Fs     = 1000

T_sec  = 1
N      = T_sec*Fs

f_01   = 25  #  Hz   frequency of binary state appearance
f_cont = 25  #
T_smp  = (1/25)*Fs   #  avg. period of binary state in unit of samples

###  build binary state sequence
t      = 0
sts01     = _N.zeros(N)
while t < N:
    t += int(T_smp*(1 + 0.15*_N.random.randn()))
    ups = _N.where(_N.random.rand(6) < 0.5)[0]

    for i in range(len(ups)):
        if t + ups[i] < N:
            sts01[t + ups[i]] = 1

stoch_T = 20
nz_ts = _N.where(_N.random.rand(N) < 1/stoch_T)[0]
sts01[nz_ts] = 1

#_plt.xcorr(ts - _N.mean(ts), cont - _N.mean(cont), maxlags=100)

roots = initF(0, 1, 0, ifs=[(_N.pi/(Fs*0.5))*f_cont], ir=0.95)

alfa  = []

for irt in range(len(roots)//2):
    alfa.append([roots[irt*2], roots[irt*2+1]])

nRhythms = len(alfa)
ARcoeff = _N.empty((nRhythms, 2))

lat_obs  = _N.empty(N+1)
for n in range(nRhythms):
    # [0.1, 0.1] (x - 0.1) * (x - 0.1)
    ARcoeff[n]          = (-1*_Npp.polyfromroots(alfa[n])[::-1][1:]).real

    xy = createDataAR(N, ARcoeff[0], 0.001, N+1, trim=0)

cont = xy[0]


fig = _plt.figure(figsize=(7, 13))
fig.add_subplot(3, 1, 1)
_plt.acorr(sts01 - _N.mean(sts01), maxlags=100)
fig.add_subplot(3, 1, 2)
_plt.acorr(cont - _N.mean(cont), maxlags=100)
fig.add_subplot(3, 1, 3)
_plt.xcorr(cont - _N.mean(cont), sts01 - _N.mean(sts01), maxlags=100)

