import numpy as _N
import re
import os

try:
    if os.environ["AIiRPS_on_colab"] == "1":
        print("found AIiRPS_on_colab")
        simulation_data_dir="AIiRPS/sampledata/simu_vs_AI"
        data_dir="AIiRPS/sampledata/HP_vs_AI"
except KeyError:   #####  SET THIS IF RUNNING LOCALLY
    print("didn't find AIiRPS_on_colab")
    simulation_data_dir="/Users/arai/nctc/Workspace/AIiRPS_SimDAT"
    data_dir="/Users/arai/Sites/janken/taisen_data"

def return_hnd_dat(ufn, tr0=0, tr1=None, know_gt=False, flip_human_AI=False):
    global simulation_data_dir, data_dir
    baseDir = data_dir if not know_gt else simulation_data_dir
    with open('%(bd)s/rpsm_%(fn)s.dat' % {"bd" : baseDir, "fn" : ufn}, 'r') as f:
        lines = f.read().splitlines()

    iCommOffset = 1 if lines[0][0] == "#" else 0
    rec_hands     = lines[iCommOffset].rstrip()
    rec_per_hands = lines[iCommOffset+1].rstrip()
    rec_reaction_times = lines[iCommOffset+2].rstrip()    

    hh  = re.split(" +", rec_hands)
    if flip_human_AI:
        human_hands   = _N.array(re.split(" +", rec_per_hands), dtype=_N.int)
        per_hands     = _N.array(re.split(" +", rec_hands), dtype=_N.int)
    else:
        human_hands   = _N.array(re.split(" +", rec_hands), dtype=_N.int)
        per_hands     = _N.array(re.split(" +", rec_per_hands), dtype=_N.int)
    reaction_times = _N.array(re.split(" +", rec_reaction_times), dtype=_N.int)
    #  First reaction_time is time from page load to 1st player response

    N             = human_hands.shape[0]
    hnd_dat = _N.empty((N, 4), dtype=_N.int)
    wtl           = _N.empty(N, dtype=_N.int)

    tr0     = 0 if tr0 is None else tr0
    tr1     = N if tr1 is None else tr1
    for i in range(N):
        if (human_hands[i] == 1):   #  HUMAN GOO
            if per_hands[i] == 1:
                wtl[i] = 0
            if per_hands[i] == 2:
                wtl[i] = 1
            if per_hands[i] == 3:   #  paa
                wtl[i] = -1
        elif (human_hands[i] == 2):   #  HUMAN CHOKI
            if per_hands[i] == 1:   #  per goo
                wtl[i] = -1
            if per_hands[i] == 2:
                wtl[i] = 0
            if per_hands[i] == 3:   #  paa
                wtl[i] = 1
        elif (human_hands[i] == 3): #  HUMAN PAA
            if per_hands[i] == 1:   
                wtl[i] = 1
            if per_hands[i] == 2:
                wtl[i] = -1
            if per_hands[i] == 3:   #  paa
                wtl[i] = 0
    hnd_dat[0, 3] = reaction_times[0]
    for i in range(1, N):
        hnd_dat[i, 3] = hnd_dat[i-1, 3] + reaction_times[i]

    hnd_dat[:, 0] = human_hands
    hnd_dat[:, 1] = per_hands
    hnd_dat[:, 2] = wtl
    #hnd_dat[:, 3] = 0

    return hnd_dat


def get_consecutive_conditioned_hands(clpd_hnd_dat, h1_1=0, h0_1=0, h1_2=0, h0_2=0, h1_3=0, h0_3=0, conditional_col=2, wtl_rps=None):
    """
    get events where t-1th wtl is wtl, the t-1th and tth hands are either
    (h0_1 and h1_1) or
    (h0_2 and h1_2) or
    (h0_3 and h1_3)
    and return the index of the t-1th point
    """
    tr1 = _N.where((clpd_hnd_dat[1:, 0] == h1_1) & (clpd_hnd_dat[0:-1, 0] == h0_1) & (clpd_hnd_dat[0:-1, conditional_col] == wtl_rps))[0]
    tr2 = _N.where((clpd_hnd_dat[1:, 0] == h1_2) & (clpd_hnd_dat[0:-1, 0] == h0_2) & (clpd_hnd_dat[0:-1, conditional_col] == wtl_rps))[0]
    tr3 = _N.where((clpd_hnd_dat[1:, 0] == h1_3) & (clpd_hnd_dat[0:-1, 0] == h0_3) & (clpd_hnd_dat[0:-1, conditional_col] == wtl_rps))[0]
    return tr1.tolist() + tr2.tolist() + tr3.tolist()

def get_ME_WTL(hnd_dat, tr0, tr1):
     stay_win = _N.where((hnd_dat[1+tr0:tr1, 0] == hnd_dat[tr0:tr1-1, 0]) & (hnd_dat[tr0:tr1-1, 2] == 1))[0]
     #  p(str | W)   #  R=0 < P=1,   P=1 < S=2,   S=2 < R=0
     strg_win = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=2, h1_2=2, h0_2=3, h1_3=3, h0_3=1, conditional_col=2, wtl_rps=1)
     #  p(str | W)   #  R=0 > S=2,   P=1 > R=0,   S=2 > P=1
     wekr_win = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=3, h1_2=2, h0_2=1, h1_3=3, h0_3=2, conditional_col=2, wtl_rps=1)
     
     #  p(stay | T)
     stay_tie = _N.where((hnd_dat[1+tr0:tr1, 0] == hnd_dat[tr0:tr1-1, 0]) & (hnd_dat[tr0:tr1-1, 2] == 0))[0]
     #  p(wkr | T)   #  R=0 < P=1,   P=1 < S=2,   S=2 < R=0
     strg_tie = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=2, h1_2=2, h0_2=3, h1_3=3, h0_3=1, conditional_col=2, wtl_rps=0)
     #  p(str | T)   #  R=0 > S=2,   P=1 > R=0,   S=2 > P=1
     wekr_tie = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=3, h1_2=2, h0_2=1, h1_3=3, h0_3=2, conditional_col=2, wtl_rps=0)
     
     #  p(stay | L)
     stay_los = _N.where((hnd_dat[tr0+1:tr1, 0] == hnd_dat[tr0:tr1-1, 0]) & (hnd_dat[tr0:tr1-1, 2] == -1))[0]
     #  p(wkr | L)   #  R=0 < P=1,   P=1 < S=2,   S=2 < R=0
     strg_los = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=2, h1_2=2, h0_2=3, h1_3=3, h0_3=1, conditional_col=2, wtl_rps=-1)
     #  p(str | L)   #  R=0 > S=2,   P=1 > R=0,   S=2 > P=1
     wekr_los = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=3, h1_2=2, h0_2=1, h1_3=3, h0_3=2, conditional_col=2, wtl_rps=-1)

     win_cond= _N.array(stay_win.tolist() + wekr_win + strg_win)
     tie_cond= _N.array(stay_tie.tolist() + wekr_tie + strg_tie)
     los_cond= _N.array(stay_los.tolist() + wekr_los + strg_los)

     return _N.array(stay_win), _N.array(wekr_win), _N.array(strg_win), \
         _N.array(stay_tie), _N.array(wekr_tie), _N.array(strg_tie), \
         _N.array(stay_los), _N.array(wekr_los), _N.array(strg_los), \
         win_cond, tie_cond, los_cond


def get_ME_RPS(hnd_dat, tr0, tr1):
     stay_R = _N.where((hnd_dat[1+tr0:tr1, 0] == hnd_dat[tr0:tr1-1, 0]) & (hnd_dat[tr0:tr1-1, 0] == 1))[0]
     #  p(str | W)   #  R=0 < P=1,   P=1 < S=2,   S=2 < R=0
     strg_R = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=2, h1_2=2, h0_2=3, h1_3=3, h0_3=1, conditional_col=0, wtl_rps=1)
     #  p(str | W)   #  R=0 > S=2,   P=1 > R=0,   S=2 > P=1
     wekr_R = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=3, h1_2=2, h0_2=1, h1_3=3, h0_3=2, conditional_col=0, wtl_rps=1)
     
     #  p(stay | T)
     stay_P = _N.where((hnd_dat[1+tr0:tr1, 0] == hnd_dat[tr0:tr1-1, 0]) & (hnd_dat[tr0:tr1-1, 0] == 3))[0]
     #  p(wkr | T)   #  R=0 < P=1,   P=1 < S=2,   S=2 < R=0
     strg_P = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=2, h1_2=2, h0_2=3, h1_3=3, h0_3=1, conditional_col=0, wtl_rps=3)
     #  p(str | T)   #  R=0 > S=2,   P=1 > R=0,   S=2 > P=1
     wekr_P = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=3, h1_2=2, h0_2=1, h1_3=3, h0_3=2, conditional_col=0, wtl_rps=3)
     
     #  p(stay | L)
     stay_S = _N.where((hnd_dat[tr0+1:tr1, 0] == hnd_dat[tr0:tr1-1, 0]) & (hnd_dat[tr0:tr1-1, 0] == 2))[0]
     #  p(wkr | L)   #  R=0 < P=1,   P=1 < S=2,   S=2 < R=0
     strg_S = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=2, h1_2=2, h0_2=3, h1_3=3, h0_3=1, conditional_col=0, wtl_rps=2)
     #  p(str | L)   #  R=0 > S=2,   P=1 > R=0,   S=2 > P=1
     wekr_S = get_consecutive_conditioned_hands(hnd_dat[tr0:tr1], h1_1=1, h0_1=3, h1_2=2, h0_2=1, h1_3=3, h0_3=2, conditional_col=0, wtl_rps=2)

     R_cond= _N.array(stay_R.tolist() + wekr_R + strg_R)
     P_cond= _N.array(stay_P.tolist() + wekr_P + strg_P)
     S_cond= _N.array(stay_S.tolist() + wekr_S + strg_S)

     return _N.array(stay_R), _N.array(strg_R), _N.array(wekr_R), _N.array(stay_P), _N.array(strg_P), _N.array(wekr_P), _N.array(stay_S), _N.array(strg_S), _N.array(wekr_S), R_cond, P_cond, S_cond


"""
#  Goo = 0, Choki = 1, Paa = 2
#  R   = 0, S     = 1, P   = 2
def get_ME_RPS(hnd_dat, tr0, tr1):
     R_R = _N.where((hnd_dat[1+tr0:tr1, 0] == 1) & (hnd_dat[tr0:tr1-1, 0] == 1))[0]
     P_R = _N.where((hnd_dat[1+tr0:tr1, 0] == 3) & (hnd_dat[tr0:tr1-1, 0] == 1))[0]
     S_R = _N.where((hnd_dat[1+tr0:tr1, 0] == 2) & (hnd_dat[tr0:tr1-1, 0] == 1))[0]

     R_P = _N.where((hnd_dat[1+tr0:tr1, 0] == 1) & (hnd_dat[tr0:tr1-1, 0] == 3))[0]
     P_P = _N.where((hnd_dat[1+tr0:tr1, 0] == 3) & (hnd_dat[tr0:tr1-1, 0] == 3))[0]
     S_P = _N.where((hnd_dat[1+tr0:tr1, 0] == 2) & (hnd_dat[tr0:tr1-1, 0] == 3))[0]

     R_S = _N.where((hnd_dat[1+tr0:tr1, 0] == 1) & (hnd_dat[tr0:tr1-1, 0] == 2))[0]
     P_S = _N.where((hnd_dat[1+tr0:tr1, 0] == 3) & (hnd_dat[tr0:tr1-1, 0] == 2))[0]
     S_S = _N.where((hnd_dat[1+tr0:tr1, 0] == 2) & (hnd_dat[tr0:tr1-1, 0] == 2))[0]

     R_cond= _N.array(R_R.tolist() + P_R.tolist() + S_R.tolist())
     P_cond= _N.array(R_P.tolist() + P_P.tolist() + S_P.tolist())
     S_cond= _N.array(R_S.tolist() + P_S.tolist() + S_S.tolist())

     return R_R, P_R, S_R, R_P, P_P, S_P, R_S, P_S, S_S, R_cond, P_cond, S_cond
"""

def write_hnd_dat(hnd_dat, fn):
    global simulation_data_dir, data_dir
    fp = open("%(dd)s/rpsm_%{fn}s.dat" % {"dd" : data_dir, "fn" : fn}, "w")
    fp.write("#  player hands, AI hands, mv times, inp method, ini_weight, fin_weights, paced_or_free, AI_or_RNG\n")
    dat_strng = str(hnd_dat[:, 0]).replace("\n", "")
    fp.write("%s\n" % dat_strng[1:-1])
    dat_strng = str(hnd_dat[:, 1]).replace("\n", "")
    fp.write("%s\n" % dat_strng[1:-1])
    dat_strng = "0 " * hnd_dat.shape[0]
    fp.write("%s\n" % dat_strng)
    dat_strng = "0 " * hnd_dat.shape[0]
    fp.write("%s\n" % dat_strng)
    dat_strng = "0\n0\n0\n0\n0\n"
    fp.write("%s" % dat_strng)
    fp.close()
