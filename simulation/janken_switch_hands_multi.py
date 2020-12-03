import numpy as _N

def next_hand(Tmat, prev_hnd_wtl, prev_hnd):
    rats = Tmat[prev_hnd_wtl-1]
    crats= _N.zeros(4)
    crats[1:4] = _N.cumsum(rats)

    rnd   = _N.random.rand()
    nxt_mv = _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0]
    
    #  nxt_mv == 0   stay 
    #  nxt_mv == 1   go_to_weaker
    #  nxt_mv == 2   go_to_stronger

    if nxt_mv == 0:
        return prev_hnd
    elif nxt_mv == 1:       #  go to weaker hnd
        if prev_hnd == 1:   #  goo(1)    ->  go to choki(2)
            return 2
        elif prev_hnd == 2: #  choki(2)  ->  paa(3)
            return 3
        elif prev_hnd == 3: #  paa    ->  goo
            return 1
        else:
            print("shouldn't be here")
    elif nxt_mv == 2:       #  go to stronger hnd
        if prev_hnd == 1:   #  goo(1)    ->  paa(3)
            return 3
        elif prev_hnd == 2: #  choki(2)  ->  goo(1)
            return 1
        elif prev_hnd == 3: #  paa(3)    ->  choki(2)
            return 2
        else:
            print("shouldn't be here")
