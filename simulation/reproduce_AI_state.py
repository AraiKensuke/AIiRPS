import AIiRPS.simulation.prcptrn2dw as prcptrn
import AIiRPS.simulation.samk as samk
import numpy as _N
import time as _tm
import AIiRPS.simulation.janken_switch_hands_multi as _jsh
import matplotlib.pyplot as _plt
import datetime
import pickle

_NME = 0
_MC1 = 1
_MC2 = 2
_PRC = 3

month_str = ["Jan","Feb", "Mar", "Apr", "May", "Jun",
             "Jul","Aug", "Sep", "Oct", "Nov", "Dec"]

nohist_crats   = _N.array([0, 0.3333333333, 0.6666666666, 1])

###  [Win_stay, Lose_switch, ]
#T0           = build_T([3., 0.3], [3., .3], [3., 0.3])
#   T0 basically says if I lose, don't switch
cyclic_strat_chg = False
cyclic_jmp       = -1   #  or -1

####  WEAK rules
#     stay, change, change   -->
#     change, change, change   -->  If perc. knows my current move, it is quite certain next move is going to be different
#    change, change, stay
####  STRONG rules
#     stay, change, change
#  [win_stay, win_change]    [tie_stay, tie_change],   [lose_stay, lose_change]

#  [p(stay | win) p(go_to_weaker | win) p(go_to_stronger | win)]
#  [p(stay | tie) p(go_to_weaker | tie) p(go_to_stronger | tie)]
#  [p(stay | los) p(go_to_weaker | los) p(go_to_stronger | los)]


###############################################
T1           = ([[0.1, 0.8, 0.1],  #  prob stay, down, up | WIN
                 [0.2, 0.7, 0.1],  #  prob stay, down, up | TIE
                 [0.1, 0.8, 0.1]]) #  prob stay, down, up | LOSE
T2           = ([[0.8, 0.1, 0.1],  #  win-stay-lose-switch, tie different
                 [0.7, 0.2, 0.1],
                 [0.1, 0.1, 0.8]])

T3           = ([[0.1, 0.7, 0.2],  #  win-stay-lose-switch, tie different
                 [0.1, 0.2, 0.7],
                 [0.7, 0.2, 0.1]])
T4           = ([[0.6, 0.2, 0.2],  #  win-stay-lose-switch, tie different
                 [0.7, 0.2, 0.4],
                 [0.1, 0.3, 0.8]])

TA           = ([[1, 0, 0],  #  win-stay-lose-switch, tie different
                 [0, 1, 0.],
                 [0, 0, 1]])

"""
##############################################
T1           = ([[0.9, 0.05, 0.05],  #  
                 [0.2, 0.7, 0.1],
                 [0.1, 0.7, 0.2]])
T2           = ([[0.9, 0.05, 0.05],  #  
                 [0.7, 0.1, 0.2],
                 [0.1, 0.2, 0.7]])
T3           = ([[0.2, 0.5, 0.3],  #
                 [0.2, 0.7, 0.1],
                 [0.3, 0.3, 0.4]])
T4           = ([[0.1, 0.1, 0.8],  
                 [0.2, 0.1, 0.7],
                 [0.1, 0.8, 0.1]])
###############################################
D1           = ([[0.7, 0.2, 0.15],  #  win-stay-lose-switch, tie different
                 [0.7, 0.2, 0.15],
                 [0.7, 0.2, 0.15]])
D2           = ([[0.1, 0.4, 0.5],  #  win-stay-lose-switch, tie different
                 [0.1, 0.4, 0.5],
                 [0.1, 0.4, 0.5]])
D3           = ([[0.1, 0.9, 0.05],  #  win-stay-lose-switch, tie different
                 [0.1, 0.9, 0.05],
                 [0.1, 0.9, 0.05]])

RND1           = ([[0.2, 0.4, 0.4],  #  win-stay-lose-switch, tie different
                   [0.5, 0.25, 0.25],
                   [0.4, 0.3, 0.3]])
"""
RND_sam1108_25           = ([[0.28, 0.36, 0.36],  #  win-stay-lose-switch, tie different
                       [0.35, 0.44, 0.21],
                       [0.31, 0.345, 0.345]])


Ts           = _N.array([T1, T2, T3])#, T4])
#Ts           = _N.array([T2.T, T4.T])#, T4])
#Ts           = _N.array([T2, T3, T4])#, T4])
#Ts           = _N.array([RND_sam1108_25])#, T4])
#Ts           = _N.array([RND])#, T4])
#Ts           = _N.array([C2])#, T4])

#  next_hand(T, wtl, last_hand)

switch_T_shrt = 15
switch_T_long = 25

max_hands  = 500

strt_chg_intvs    = _N.random.randint(switch_T_shrt, switch_T_long, size=max_hands)
strt_chg_times    = _N.cumsum(strt_chg_intvs)
uptohere          = len(_N.where(strt_chg_times < max_hands)[0])
strt_chg_times01  = _N.zeros(max_hands+1, dtype=_N.int)
strt_chg_times01[strt_chg_times[0:uptohere]] = 1

vs_human   = False     #  is there dynamics in the human hand selection?
hist_dep_hands = True   # if vs_human is False, does AI play random or vs rule
comp       = _MC2  #  _MC1, _MC2, _PRC
mc_decay   = 0.1

#   percept vs human
#   percept vs computer (hist_dep)
#   percept vs computer (not hist_dep)
#   Nash_eq vs human
#   Nash_eq vs computer (hist_dep)
#   Nash_eq vs computer (not hist_dep)


REPS       = 100

chg        = _N.zeros(REPS)
fws        = _N.zeros((REPS, 3), dtype=_N.int)

obs_go_prbs = _N.ones((max_hands+1, 3))*-1

Ts_timeseries = []

now     = datetime.datetime.now()
day     = "%02d" % now.day
mnthStr = month_str[now.month-1]
year    = "%d" % (now.year-2000)
hour    = "%02d" % now.hour
minute  = "%02d" % now.minute
second  = "%02d" % now.second
jh_fn_mod = "rpsm_%(yr)s%(mth)s%(dy)s-%(hr)s%(min)s-%(sec)s" % {"yr" : year, "mth" : mnthStr, "dy" : day, "hr" : hour, "min" : minute, "sec" : second}

iCurrStrat = 0
for rep in range(REPS):
    ######  int main 
    #int i,pred,m,v[3],x[3*N+1],w[9*N+3],fw[3];

    N = 5
    #  initialize
    # v = _N.zeros(3)                  # inputs into predictive units
    # x = _N.zeros(3*N+1)              # past moves by player 
    # x[3*N]=-1                        # threshold      x[3*N] never changes
    # w = _N.zeros(9*N+3)              # weights
    fw= _N.zeros(3, dtype=_N.int)    #  cum win, draw, lose

    if comp != _NME:
        if comp == _PRC:
            HAL9000 = prcptrn.perceptron(N)
        else:
            HAL9000 = samk.MarkovChain(1, mc_decay, transform=(comp == _MC2))

    m=1
    pairs = []

    quit  = False
    hds   = -1

    t00    = _tm.time()

    prev_w = True
    prev_m = 1

    iCurrStrat = 0
    N_strategies = Ts.shape[0]

    prevM = 1
    prevWTL = 1

    #prev_gcp_wtl = _N.zeros((9, 1), dtype=_N.int)
    prev_gcp_wtl = _N.zeros(9, dtype=_N.int)  #  prev goo chok paa win tie los
    prev_gcp_wtl_unob = _N.zeros(9, dtype=_N.int)

    iSinceLastStrChg = 0

    switch_ts = []

    initial = _N.random.choice(['1','2','3'])    
    pair = "11"

    while not quit:
        hds += 1

        switched = 0
        #  first, perceptron prediction of player move

        #pred=_N.random.randint(1, 4) if vs_NME else HAL9000.predict(m,x,w,v, update=(hds%20==0))   # pred is prediction of user move (1,2,3) */
        pred=_N.random.randint(1, 4) if comp == _NME else int(HAL9000.predict(pair))   # pred is prediction of user move (1,2,3) */)

        #pred=_N.random.randint(1, 4) if vs_NME else HAL9000.predict(m,x,w,v)   # pred is prediction of user move (1,2,3) */
        #pred=_N.random.randint(1, 4) if vs_NME else HAL9000.predict(m,x,w,v, update=True, uw=0.0001)   # pred is prediction of user move (1,2,3) */

        #  user inputs move
        if vs_human:
            print("\n\n---------\nmake your move {1(goo),2(choki),3(paa)}:")
            ###  RPS   rock scissor paper

            got_input = False
            while not got_input:
                m = raw_input()
                if not m.isdigit():
                    if m != '':
                        quit = True
                        got_input = True
                    else:
                        print("???????????????????????????????????  try again (non-number input quits janken)")                
                else:
                    m = int(m)
                    if (m < 1) or (m > 3):
                        print("???????????????????????????????????  try again (non-number input quits janken)")
                    else:
                        got_input = True
        else:   #  not vs. human
            if N_strategies > 1:
                if strt_chg_times01[hds] == 1:
                    candidate = _N.random.randint(0, Ts.shape[0])
                    while candidate == iCurrStrat:
                        candidate = _N.random.randint(0, Ts.shape[0])
                    iCurrStrat = candidate
                    switch_ts.append(hds)
                        
            #print("prevM  %d" % prevM)
            prev_gcp_wtl[:] = 0
            #  if prev was goo     did we win tie lose
            #  if prev was choki   did we win tie lose
            prev_gcp_wtl[(prevM-1)*3+prevWTL-1] = 1  #
            #  the latent state is p(change | win), p(change | tie), p(change | lose)
            #  p(change | win) = p(change | win)  == 
            #  p(R | S, W) + p(P | S, W)
            #  p(S | R, W) + p(P | R, W)
            #  p(S | P, W) + p(R | P, W)

            #  p(change | tie)

            if hist_dep_hands:
                #  the probs is 
                m = _jsh.next_hand(Ts[iCurrStrat], prevWTL, prevM)  #  m is 1, 2, 3     #  prob of 3 hands
                
                Ts_timeseries.append(Ts[iCurrStrat])
            else:  #  not hist_dep_hands
                rnd = _N.random.rand()
                m = _N.where((rnd >= nohist_crats[0:-1]) & (rnd < nohist_crats[1:]))[0][0]+1
            prevM = m                        
            #print("m: %(m)d    currStr %(cs)d" % {"m" : m, "cs" : iCurrStrat})

            if hds >= max_hands:
                quit = True
            pair="%(prd)d%(m)d" % {"prd" : int(pred), "m" : m}

        if not quit:
            # show perceptron's move

            #print("<->%d:   " % ((pred+1)%3+1)) 

            #   m = (pred+1)%3+1 would beat the perceptron's prediction
            #   pred=1(goo)   :(pred+1)%3+1=3(paa)
            #   pred=2(choki) :(pred+1)%3+1=1(goo)
            #   pred=3(paa)   :(pred+1)%3+1=2(choki)*/

            #  who won, lost or tied?
            
            if pred==m:  #  human move predicted, so perceptron wins
                if vs_human:
                    print("[perceptron wins]")  
                fw[2] += 1
                prevWTL = 3
                #pairs.append([m, (pred+1)%3+1, -1, switched])
                pairs.append([m, (pred+1)%3+1, -1, iCurrStrat])
            elif (pred%3) == (m-1):
                if vs_human:
                    print("[!!!       YOU WIN   !!!!!!!   YOU WIN        !!!]")
                    print("[!!!       YOU WIN   !!!!!!!   YOU WIN        !!!]")
                fw[0] += 1
                prevWTL = 1
                #  human player wins because
                #   pred%3=0(paa predicted) so percep outputs choki, but player goo
                #   pred%3=1(goo predicted) so percep outputs paa, but player choki
                #   pred%3=2(choki predicted) so percep outputs goo, but player paa
                #   Toda comment 2003/05/28 */
                pairs.append([m, (pred+1)%3+1, 1, iCurrStrat])
            else:
                if vs_human:
                    print("[tie XXXXX]")
                fw[1] += 1
                pairs.append([m, (pred+1)%3+1, 0, iCurrStrat])
                prevWTL = 2            


            if vs_human:
                print("    PER %(pc)d,  TIE %(tie)d, [[HUMAN]] %(hum)d   UpOrDown %(updn)d" % {"pc" : fw[2], "tie" : fw[1], "hum" : fw[0], "updn" : (fw[0] - fw[2])})

    t01 = _tm.time()
    #print("game duration  %.1f" % (t01-t00))    

    fws[rep] = fw
    #print("    PER %(pc)d,  TIE %(tie)d, [[HUMAN]] %(hum)d   UpOrDown %(updn)d" % {"pc" : fw[2], "tie" : fw[1], "hum" : fw[0], "updn" : (fw[0] - fw[2])})

    if rep == 0:   #  only save 1st one
        file_nm = "/Users/arai/nctc/Workspace/AIiRPS_SimDAT/%s.dat" % jh_fn_mod
        gt_file_nm = "/Users/arai/nctc/Workspace/AIiRPS_SimDAT/%s_GT.dat" % jh_fn_mod
        #u_fnm = uniqFN("SimDAT/%s" % file_nm, serial=True)
        #u_fnm_gt = uniqFN("SimDAT/%s" % gt_file_nm, serial=True)
        hnd_dat = _N.array(pairs, dtype=_N.int)
        _N.savetxt(file_nm, hnd_dat, fmt="%d %d % d %d")
        print("janken match data: %s" % file_nm)
        #if hist_dep_hands and (not vs_human):

        pklme = {}
        pklme["hnd_dat"] = hnd_dat
        pklme["Ts_timeseries"] = _N.array(Ts_timeseries)
        dmp = open("/Users/arai/nctc/Workspace/AIiRPS_SimDAT/%s.dmp" % jh_fn_mod, "wb")
        pickle.dump(pklme, dmp, -1)
        dmp.close()

if REPS > 1:
    print("mean human wins, mean ties, mean computer wins")
    print(_N.mean(fws, axis=0))


    if (comp == _MC1) or (comp == _MC2):
        print("vs MC%d" % comp)
        print("decay:  %.3f" % mc_decay)
    else:
        print("vs perceptron")

    print("human number strategies  %d" % Ts.shape[0])
    print("human change:  %(1)d  %(2)d" % {"1" : switch_T_shrt, "2" : switch_T_long})
