import numpy as _N
import re
import os
import glob
import pandas as pd
import AIiRPS.utils.misc as misc

_TRUE_ONLY_  = 0
_FALSE_ONLY_ = 1
_TRUE_AND_FALSE_ = 2
try:
    if os.environ["AIiRPS_on_colab"] == "1":
        print("found AIiRPS_on_colab")
        _simulation_data_dir="AIiRPS/sampledata/simu_vs_AI"
        _data_dir="AIiRPS/sampledata/HP_vs_AI"
except KeyError:   #####  SET THIS IF RUNNING LOCALLY
    print("didn't find AIiRPS_on_colab")
    _simulation_data_dir="/Users/arai/nctc/Workspace/AIiRPS_SimDAT"
    _data_dir="/Users/arai/Sites/taisen/DATA"
    _data_dir_base = "/Users/arai/Sites/taisen/DATA"

def date_range(start=None, end=None):
    date_strs = []    
    if (start is not None) and (end is not None):
        dates = pd.date_range(start=start, end=end)

        for date_tmstmp in dates:
            _date = "%(yr)d%(mn)2d%(dy)2d" % {"yr" : date_tmstmp.year, "mn" : date_tmstmp.month, "dy" : date_tmstmp.day}
            date  = _date.replace(" ", "0")
            date_strs.append(date)
    return date_strs
    

def return_hnd_dat(day_time, tr0=0, tr1=None, know_gt=False, flip_human_AI=False, has_useragent=False, has_start_and_end_times=False, has_constructor=False, block=1, expt="EEG1", visit=None, ai_states=False):
    """
    starttime and endtime needed because filename date is when userID was crafted, but not the time RPS game was started.
    """
    data_dir = "%(dd)s/%(ex)s" % {"dd" : _data_dir, "ex" : expt}
    baseDir = data_dir if not know_gt else _simulation_data_dir
    day = day_time[0:8]

    look_in_dir = "%(dd)s/%(dy)s/%(dt)s" % {"dd" : data_dir, "dt" : day_time, "dy" : day}

    if visit is None:
        s = "%s/*.dat" % look_in_dir
    else:
        s = "%(lid)s/%(v)d/*.dat" % {"lid" : look_in_dir, "v" : visit}
    if know_gt:
        import pickle
        with open("%s/block1_AI.dmp" % look_in_dir, "rb") as f:
            gt_dmp = pickle.load(f)

    dat_files = glob.glob(s)
    #
    rpsm_fn = None
    if len(dat_files) == 1:
        rpsm_fn = dat_files[0][len(look_in_dir)+1:]
    else:
        if block is not None:
            for ib in range(len(dat_files)):
                base_filename = dat_files[ib][0:-4]
                bfnl          = len(base_filename)
                
                str_bl = str(block)
                len_blstr = len(str_bl)   #  length of block # as a string
                if base_filename[bfnl-len_blstr-3:bfnl-3] == str_bl:
                    rpsm_fn = dat_files[ib][len(look_in_dir)+1:]
                    #rpsm_fn = os.path.basename(dat_files[ib])

    if rpsm_fn is None:
        print("returning None      %d" % block)
        return None, None, None, None, None, None, None, None

    #print("----")
    #print('%(bd)s/%(day)s/%(partID)s/%(rpsmfn)s' % {"day" : day, "partID" : day_time, "bd" : baseDir, "rpsmfn" : rpsm_fn})
    
    with open('%(bd)s/%(day)s/%(partID)s/%(rpsmfn)s' % {"day" : day, "partID" : day_time, "bd" : baseDir, "rpsmfn" : rpsm_fn}) as f:
        lines = f.read().splitlines()

    start = -1
    end   = -1    
    iCommOffset = 1 if lines[0][0] == "#" else 0
    iDataOffset = iCommOffset
    if has_useragent:
        UA = lines[iDataOffset].rstrip()
        iDataOffset += 1
    if has_start_and_end_times:
        start = lines[iDataOffset].rstrip()
        iDataOffset += 1
        end = lines[iDataOffset].rstrip()
        iDataOffset += 1
    if has_constructor:
        cnstr = lines[iDataOffset].rstrip()
        iDataOffset += 1

    _rec_hands     = lines[iDataOffset].rstrip()
    iDataOffset += 1        
    _rec_per_hands = lines[iDataOffset].rstrip()
    iDataOffset += 1            

    ##  from here, we're going to work with R=1, S=2, P=3
    rec_hands = _rec_hands.replace("R", "1").replace("S", "2").replace("P", "3")
    rec_per_hands = _rec_per_hands.replace("R", "1").replace("S", "2").replace("P", "3")    
    rec_reaction_times = lines[iDataOffset].rstrip()
    iDataOffset += 1
    rec_input_method = lines[iDataOffset].rstrip()
    iDataOffset += 1
    AI_arch = lines[iDataOffset].rstrip()
    iDataOffset += 1
    if expt != "TMB1":
        rec_ini_percep = lines[iDataOffset].rstrip()
        iDataOffset += 1    
        rec_fin_percep = lines[iDataOffset].rstrip()
        iDataOffset += 1    

    if flip_human_AI:
        human_hands   = _N.array(re.split(" +", rec_per_hands), dtype=int)
        per_hands     = _N.array(re.split(" +", rec_hands), dtype=int)
    else:
        human_hands   = _N.array(re.split(" +", rec_hands), dtype=int)
        per_hands     = _N.array(re.split(" +", rec_per_hands), dtype=int)
    reaction_times = _N.array(re.split(" +", rec_reaction_times), dtype=int)
    #  First reaction_time is time from page load to 1st player response

    N             = human_hands.shape[0]
    hnd_dat = _N.empty((N, 4), dtype=int)
    wtl           = _N.empty(N, dtype=int)

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
    if expt != "TMB1":
        return hnd_dat, start, end, UA, cnstr, rec_input_method, rec_ini_percep, rec_fin_percep, gt_dmp
    else:
        return hnd_dat, start, end, UA, cnstr, rec_input_method, None, None, None


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

def write_hnd_dat(hnd_dat, data_dir, fn):
    #global simulation_data_dir, data_dir
    fp = open("%(dd)s/%(fn)s.dat" % {"dd" : data_dir, "fn" : fn}, "w")
    fp.write("#  player hands, AI hands, mv times, inp method, ini_weight, fin_weights, paced_or_free, AI_or_RNG\n")
    fp.write("Notzilla/5.0\n")
    fp.write("01Jan01-0000-00\n")
    fp.write("01Jan01-0000-01\n")
    fp.write("Perceptron(2)\n")    
    dat_strng = str(hnd_dat[:, 0]).replace("\n", "")
    dat_strng = dat_strng.replace("1", "R").replace("2", "S").replace("3", "P")
    #  goo choki paa
    fp.write("%s\n" % dat_strng[1:-1])
    dat_strng = str(hnd_dat[:, 1]).replace("\n", "")
    dat_strng = dat_strng.replace("1", "R").replace("2", "S").replace("3", "P")
    fp.write("%s\n" % dat_strng[1:-1])
    dat_strng = "0 " * hnd_dat.shape[0]
    fp.write("%s\n" % dat_strng)
    dat_strng = "0 " * hnd_dat.shape[0]
    fp.write("%s\n" % dat_strng)
    dat_strng = "0\n0\n0\n0\n0\n"
    fp.write("%s" % dat_strng)
    fp.close()

def filterRPSdats(expt, dates, visits=[1], domainQ=_TRUE_AND_FALSE_, demographic=_TRUE_AND_FALSE_, mentalState=_TRUE_AND_FALSE_, maxIGI=20000, minIGI=0, min_meanIGI=1000, max_meanIGI=10000, MinWinLossRat=0, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1, ngames=None, print_warn=False):
    """
    visit:   1, 2, 3    
    works like a filter
    """
    files = []
    hnd_dats = {}          # key is partID
    constructors = {}      # key is partID
    constructors0 = []
    datn         = -1
    for date in dates:
        datdir = "%(bd)s/%(e)s/%(dt)s" % {"bd" : _data_dir_base, "e" : expt, "dt" : date}

        if os.access(datdir, os.F_OK):
            dats4date = os.listdir(datdir)

            for time in dats4date:   #  the data sets for this day
                dattmdir = "%(dd)s/%(tm)s" % {"dd" : datdir, "tm" : time}

                warnings = ["", ""]
                datfilewarnings = ["not found", "not found"]        
                
                demoG = False
                domQ  = False
                goods = [False] * len(visits)
                if os.access("%s/DQ1.txt" % dattmdir, os.F_OK):
                    demoG = True
                    if os.access("%s/AQ29.txt" % dattmdir, os.F_OK):
                        domQ = True

                for visit in visits:   #  multiple visits for the day
                    inclDemo = False
                    inclDom  = False
                    allblocks= False
                    igiCondMet = _N.ones(blocks, dtype=int)
                    all_igiCondMet = _N.zeros(blocks, dtype=int)
                    winCondMet = _N.ones(blocks, dtype=int)      
                    ngameCondMet = _N.ones(blocks, dtype=int)      

                    fn_for_block = [] # [dat-tm, dat-tm, dat-tm, dat-tm]
                    dat_for_block = []
                    cnstr_for_block = []                                        
                    for blk in range(1, blocks+1):
                        datfn = "%(dd)s/%(v)d/block%(bl)d_AI.dat" % {"dd" : dattmdir, "v" : visit, "bl" : blk}

                        if os.access(datfn, os.F_OK):
                            datfilewarnings[visit-1] = "found"

                            if ((demoG == True) and ((demographic == _TRUE_AND_FALSE_) or (demographic == _TRUE_ONLY_))) or \
                               ((demoG == False) and ((demographic == _TRUE_AND_FALSE_) or (demographic == _FALSE_ONLY_))):
                                inclDemo = True
                            if ((domQ == True) and ((domainQ == _TRUE_AND_FALSE_) or (domainQ == _TRUE_ONLY_))) or \
                               ((domQ == False) and ((domainQ == _TRUE_AND_FALSE_) or (domainQ == _FALSE_ONLY_))):
                                inclDom = True
                            if (inclDemo and inclDom):
                                theDat, tStrt, tEnd, UA, cnstr, inp_meth, ini_percep, fin_percep = return_hnd_dat(time, tr0=0, tr1=None, know_gt=False, flip_human_AI=False, has_useragent=has_useragent, has_start_and_end_times=has_start_and_end_times, has_constructor=has_constructor, block=blk, visit=visit, expt=expt)

                                if theDat is None:
                                    print("theDat is None!!!!!!!!!!!")
                                    print(datfn)
                                all_avgIGI = _N.mean(_N.diff(theDat[:, 3]))
                                if (all_avgIGI > min_meanIGI) and (all_avgIGI < max_meanIGI):
                                    all_igiCondMet[blk-1]= 1
                                if (minIGI > 0):
                                    L = theDat.shape[0]
                                    L5 = L//5

                                    for pc in range(5):
                                        mult = 1
                                        avgIGI = _N.mean(_N.diff(theDat[pc*L5:(pc+1)*L5, 3]))
                                        #print("%(a).3f   %(m)d" % {"a" : avgIGI, "m" : maxIGI})

                                        if (avgIGI < minIGI*mult) or (avgIGI > maxIGI):
                                            igiCondMet[blk-1] = 0
                                    
                                if (MinWinLossRat > 0):
                                    wins = len(_N.where(theDat[:, 2] == 1)[0])
                                    loss = len(_N.where(theDat[:, 2] == -1)[0])
                                    if (wins / loss) < MinWinLossRat:
                                        ###  losing too many times
                                        winCondMet[blk-1] = 0
                                if (ngames is not None):
                                    if ngames == theDat.shape[0]:
                                        ngameCondMet[blk-1] = 1
                                    else:
                                        ngameCondMet[blk-1] = 0
                                else:
                                        ngameCondMet[blk-1] = 1

                                warnings[visit-1] = "IGI %(1)d  WinCon %(2)d  ngame %(3)d  allIGI %(4)d" % {"1" : igiCondMet[blk-1], "2" : winCondMet[blk-1], "3" : ngameCondMet[blk-1], "4" : all_igiCondMet[blk-1]}
                                        
                                if (igiCondMet[blk-1] == 1) and (winCondMet[blk-1] == 1) and (ngameCondMet[blk-1] == 1) and (all_igiCondMet[blk-1] == 1):
                                    fn_for_block.append(time)
                                    dat_for_block.append(theDat)
                                    cnstr_for_block.append(cnstr)
                                
                                    #hnd_dats[time] = theDat
                                    #files.append(time)
                                    #hnd_dats[time] = theDat
                                    #files.append(time)
                                    
                    if len(fn_for_block) == blocks:    #  complete
                        datn += 1
                        goods[visit-1] = True   #  we have 4 blocks
                        if datn == 0:
                            #  test that all constructors are unique
                            if no_duplicates(cnstr_for_block):
                                constructors0 = cnstr_for_block
                                hnd_dats[time] = dat_for_block
                                constructors[time] = cnstr_for_block
                            else:
                                print("!!! duplicate constructor found as constr0 %s" % time)
                                goods[visit-1] = False
                                datn -= 1  
                        else:
                            #print(dattmdir)
                            #print(fn_for_block)
                            if no_duplicates(cnstr_for_block):
                                dat_order_like_0 = orderByConstructors(constructors0, cnstr_for_block, dat_for_block)
                                hnd_dats[time] = dat_order_like_0
                                constructors[time] = constructors0
                            else:
                                print("!!! duplicate constructor found as constr %s" % time)                                
                                goods[visit-1] = False      

                good = True
                for iv in range(len(visits)):
                    if goods[iv] == False:
                        good = False
                if print_warn:
                    print(time)
                    print(warnings)
                    print(datfilewarnings)                
                if good:
                    files.append(time)
                        

    return files, hnd_dats, constructors

def orderByConstructors(useThisOrder, datorder, dats):
    """
    useThisOrder is a list of constructors
    order0dat = orderByConstrcutors(cnstrs[partIDs[0]], cnstrs[partIDs[i]], dats[partIDs[i]])
    """
    keyedDat = {}
    i = -1
    for cn in datorder:
        i+=1 
        keyedDat[cn] = dats[i]
    reordered = []

    i = -1
    for cn in datorder:
        i+=1
        reordered.append(keyedDat[useThisOrder[i]])
    return reordered

        
    

def AQ28(aq28fn):
    #“Definitely agree” or “slightly agree” responses scored 1 point, on the following items:
    #  1, 2, 4, 5, 6, 7, 9, 12, 13, 16, 18, 19, 20, 21, 22, 23, 26, 33, 35, 39, 41, 42, 43, 45, 46.
    #“Definitely disagree” or “slightly dis- agree” responses scored 1 point, on the following items:
    #  3, 8, 10, 11, 14, 15, 17, 24, 25, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 40, 44, 47, 48, 49, 50.

    #A   Agree has high score
    #C
    #--- SOCIAL SKILLS    
    #1   1C  "I prefer to do things with others rather than on my own.",
    #2   11C "I find social situations easy.",
    #3   13A "I would rather go to a library than to a party.",
    #4   15C "I find myself drawn more strongly to people than to things.",
    #5   22A "I find it hard to make new friends.",
    
    #6   44C "I enjoy social occasions.",
    #7   47C "I enjoy meeting new people.",
    #--- ROUTINE    
    #8   2A  "I prefer to do things the same way over and over again.",
    #9   25C "It does not upset me if my daily routine is disturbed.",
    #10  34C"I enjoy doing things spontaneously.",
    
    #11  46A"New situations make me anxious.",
    #--- SWITCHING
    #12  4A "I frequently get strongly absorbed in one thing.",
    #13  10C"I can easily keep track of several different people's conversations.",
    #14  32C"I find it easy to do more than one thing at once.",
    #15  37C"If there is an interruption, I can switch back very quickly.",
    #--- IMAG
    
    #16  3C "Trying to imagine something, I find it easy to create a picture in my mind.",
    #17  8C "Reading a story, I can easily imagine what the characters might look like.",
    #18  14C"I find making up stories easy.",
    #19  20A"Reading a story, I find it difficult to work out the character's intentions.",
    #20  36C"I find it easy to work out what someone is thinking or feeling.",
    #21  42A"I find it difficult to imagine what it would be like to be someone else.",
    #22  45A"I find it difficult to work out people's intentions.",
    #23  50C"I find it easy to play games with children that involve pretending.",
    #--- FACT NUMB AND PATT    
    #24  6A "I usually notice car number plates or similar strings of information.",
    #25  9A "I am fascinated by dates.",
    
    #26  19A"I am fascinated by numbers.",
    #27  23A"I notice patterns in things all the time.",
    #28  41A"I like to collect information about categories of things."

    flip = _N.array([1, 1, 0, 1, 0,
                     1, 1, 0, 1, 1,
                     0, 0, 1, 1, 1,
                     1, 1, 1, 0, 1,
                     0, 0, 1, 0, 0,
                     0, 0, 0])
    #1A   #  strongly agree    = autism
    #11C  #  strongly disagree = autism
    #13A  #  strongly agree    = autism
    #15C
    #22A
    
    #44C
    #47C
    #2A 
    #25C
    #34C
    
    #46A
    #4A 
    #10C
    #32C
    #37C
                 
    #3C
    #8C 
    #14C
    #20A
    #36C
    
    #42A
    #45A
    #50C
    #6A 
    #9A
                 
    #19A
    #23A
    #41A

    answers = _N.loadtxt(aq28fn)

    # tot1 = 0
    # for i in range(28):
    #     if flip[i] == 0:
    #         tot1 += answers[i]
    #     elif flip[i] == 1:
    #         tot1 += 5 - answers[i]   # 5-1, 5-2, 5-3, 5-4

    ind_scrs = flip*5 + -1*flip*answers + (1-flip)*answers

    return _N.sum(ind_scrs), _N.sum(ind_scrs[0:7]), _N.sum(ind_scrs[7:11]), _N.sum(ind_scrs[11:15]), _N.sum(ind_scrs[15:23]), _N.sum(ind_scrs[23:])

def no_duplicates(lst):
    duplicate = False
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j]:
                duplicate = True
    return (not duplicate)

def AQ28ans(aq28fn):
    #“Definitely agree” or “slightly agree” responses scored 1 point, on the following items:
    #  1, 2, 4, 5, 6, 7, 9, 12, 13, 16, 18, 19, 20, 21, 22, 23, 26, 33, 35, 39, 41, 42, 43, 45, 46.
    #“Definitely disagree” or “slightly dis- agree” responses scored 1 point, on the following items:
    #  3, 8, 10, 11, 14, 15, 17, 24, 25, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 40, 44, 47, 48, 49, 50.

    #A   Agree has high score
    #C
    #--- SOCIAL SKILLS     [0, 1, 2, 3, 4, 5, 6]
    #1   1C  "I prefer to do things with others rather than on my own.",
    #2   11C "I find social situations easy.",
    #3   13A "I would rather go to a library than to a party.",
    #4   15C "I find myself drawn more strongly to people than to things.",
    #5   22A "I find it hard to make new friends.",
    
    #6   44C "I enjoy social occasions.",
    #7   47C "I enjoy meeting new people.",
    #--- ROUTINE    [7, 8, 9, 10]
    #8   2A  "I prefer to do things the same way over and over again.",
    #9   25C "It does not upset me if my daily routine is disturbed.",
    #10  34C"I enjoy doing things spontaneously.",
    
    #11  46A"New situations make me anxious.",
    #--- SWITCHING  [11, 12, 13, 14]
    #12  4A "I frequently get strongly absorbed in one thing.",
    #13  10C"I can easily keep track of several different people's conversations.",
    #14  32C"I find it easy to do more than one thing at once.",
    #15  37C"If there is an interruption, I can switch back very quickly.",
    #--- IMAG       [15, 16, 17, 18, 19, 20, 21, 22]
    #  16, 17 <- imagingn  15, 16,
    #  18, 23 <--- pretending  
    #  19, 20, 21, 22  <-- other's intentions  18, 19, 20, 21
    #  20, 22  <-- other's intentions  19, 21
    #16  3C "Trying to imagine something, I find it easy to create a picture in my mind.",
    #17  8C "Reading a story, I can easily imagine what the characters might look like.",
    #18  14C"I find making up stories easy.",
    #19  20A"Reading a story, I find it difficult to work out the character's intentions.",
    #20  36C"I find it easy to work out what someone is thinking or feeling.",
    

    #21  42A"I find it difficult to imagine what it would be like to be someone else.",
    #22  45A"I find it difficult to work out people's intentions.",
    #23  50C"I find it easy to play games with children that involve pretending.    #--- FACT NUMB AND PATT",     [23, 24, 25, 26, 27]
    #24  6A "I usually notice car number plates or similar strings of information.",
    #25  9A "I am fascinated by dates.",
    
    #26  19A"I am fascinated by numbers.",
    #27  23A"I notice patterns in things all the time.",
    #28  41A"I like to collect information about categories of things."

    flip = _N.array([1, 1, 0, 1, 0,
                     1, 1, 0, 1, 1,
                     0, 0, 1, 1, 1,
                     1, 1, 1, 0, 1,
                     0, 0, 1, 0, 0,
                     0, 0, 0])
    #1A   #  strongly agree    = autism
    #11C  #  strongly disagree = autism
    #13A  #  strongly agree    = autism
    #15C
    #22A
    
    #44C
    #47C
    #2A 
    #25C
    #34C
    
    #46A
    #4A 
    #10C
    #32C
    #37C
                 
    #3C
    #8C 
    #14C
    #20A
    #36C
    
    #42A
    #45A
    #50C
    #6A 
    #9A
                 
    #19A
    #23A
    #41A

    answers = _N.loadtxt(aq28fn)

    # tot1 = 0
    # for i in range(28):
    #     if flip[i] == 0:
    #         tot1 += answers[i]
    #     elif flip[i] == 1:
    #         tot1 += 5 - answers[i]   # 5-1, 5-2, 5-3, 5-4

    ind_scrs = flip*5 + -1*flip*answers + (1-flip)*answers

    return ind_scrs[_N.array([0, 1, 2, 3, 4, 5, 6])], ind_scrs[_N.array([7, 8, 9, 10])], ind_scrs[_N.array([11, 12, 13, 14])], ind_scrs[_N.array([15, 16, 17, 18, 19, 20, 21, 22])], ind_scrs[_N.array([23, 24, 25, 26, 27])]

def no_duplicates(lst):
    duplicate = False
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j]:
                duplicate = True
    return (not duplicate)

def Demo(demo_fn):
    fp = open(demo_fn, "r")
    answers = fp.readlines()

    age = -1
    if answers[1][:-1] == "<18":
        age = 1
    elif answers[1][:-1] == "18-24":
        age = 2
    elif answers[1][:-1] == "25-29":
        age = 3
    elif answers[1][:-1] == "30-34":
        age = 4
    elif answers[1][:-1] == "35-39":
        age = 5
    elif answers[1][:-1] == "40-44":
        age = 6
    elif answers[1][:-1] == "45-49":
        age = 7
    elif answers[1][:-1] == "50-54":
        age = 8
    elif answers[1][:-1] == "55-59":
        age = 9
    elif answers[1][:-1] == "60-64":
        age = 10
    elif answers[1][:-1] == "65-69":
        age = 11
    elif answers[1][:-1] == "70-74":
        age = 12
    elif answers[1][:-1] == "75-79":
        age = 13
    elif answers[1][:-1] == "80-84":
        age = 14
    elif answers[1][:-1] == "85-89":
        age = 15
    elif answers[1][:-1] == ">90":
        age = 16
    gen = -1
    if answers[2][:-1] == "Male":
        gen = 0
    elif answers[2][:-1] == "Female":
        gen = 1
    elif answers[2][:-1] == "Non-binary":
        gen = 2
    Eng = -1        
    if answers[3][:-1] == "Yes":
        Eng = 1
    elif answers[3][:-1] == "No":
        Eng = 0

    return age, gen, Eng

def repeated_keys(hnd_dat):
    """
    return me 
    """
    return misc.repeated_array_entry(hnd_dat[:, 0])


    # longest_repeats = []
    # i = 0
    # L = hnd_dat.shape[0]
    # while i < L-1:
    #     if hnd_dat[i, 0] == hnd_dat[i+1, 0]:  #  Found a repeat
    #         j = i
    #         keep_going = True
    #         while (j < L-1) and keep_going:
    #             if hnd_dat[j, 0] != hnd_dat[j+1, 0]:
    #                 longest_repeats.append(j - i+1)
    #                 keep_going = False
    #             j += 1
    #         if keep_going:  #  hit end of loop while a repeat
    #             longest_repeats.append(j - i+1)                
    #         i = j-1  #  j+1 is not equal
    #     else:     #  Not a repeat
    #         longest_repeats.append(1)
    #     i += 1
    # if hnd_dat[L-2, 0] != hnd_dat[L-1, 0]:   #  last 2 are not repeats
    #     longest_repeats.append(1)

    # return longest_repeats

