#!/usr/bin/python

import sys
import os
import pandas as pd
import AIiRPS.utils.read_taisen as _rt

#  take the stuff from TMB experiments online
#  and make the filenames easier

if len(sys.argv) < 2:
    print("python TMB_easy_dirnames.py <TMB1_o, TMB1_n or TMB2>")
    sys.exit()
if (sys.argv[1] != "TMB1_o") and (sys.argv[1] != "TMB1_n") and (sys.argv[1] != "TMB2"):
    print("unknown exptname")
    sys.exit()
if sys.argv[1] == "TMB2":
    expt = sys.argv[1]
else:
    expt = "TMB1"
    old_TMB1 = False    
    if sys.argv[1] == "TMB1_o":
        old_TMB1 = True

zipped_dir = "/Users/arai/Sites/taisen/DATA/%s" % expt

#dates      = ["20210723"]

dates = _rt.date_range(start='7/13/2021', end='11/30/2021')
if (expt == "TMB1"):
    if old_TMB1:
        dates = _rt.date_range(start='7/13/2021', end='8/17/2021')
    else:
        dates = _rt.date_range(start='8/18/2021', end='11/30/2021')
        
#dates = pd.date_range(start='7/1/2021', end='7/31/2021')

for date in dates:
#    _date = "%(yr)d%(mn)2d%(dy)2d" % {"yr" : date_tmstmp.year, "mn" : date_tmstmp.month, "dy" : date_tmstmp.day}
#    date  = _date.replace(" ", "0")

    hr = 0
    mn = 0
    
    just_datetimes = []

    flder = "%(zd)s/%(dt)s" % {"zd" : zipped_dir, "dt" : date}
    if os.access(flder, os.F_OK):
        pIDdirs = os.listdir(flder)

        for pIDdir in pIDdirs:
            syr = pIDdir[0:8]

            print(pIDdir[17:21])#26])
            shr = "%d" % hr
            shr = shr.rjust(2, '0')
            smn = "%d" % mn
            smn = smn.rjust(2, '0')

            if os.access("%(fld)s/%(dt)s_%(hr)s%(mn)s-00" % {"dt" : syr, "hr" : shr, "mn" : smn, "fld" : flder}, os.F_OK):
                print("Destination %(fld)s/%(dt)s_%(hr)s%(mn)s-00 exists." % {"dt" : syr, "hr" : shr, "mn" : smn, "fld" : flder})
            #print("%(fld)s/%(pIDdir)s"  % {"fld" : flder, "pIDdir" : pIDdir})
            os.rename("%(fld)s/%(pIDdir)s"  % {"fld" : flder, "pIDdir" : pIDdir}, "%(fld)s/%(dt)s_%(hr)s%(mn)s-00" % {"dt" : syr, "hr" : shr, "mn" : smn, "fld" : flder})
            mn += 5
            if mn > 55:
                mn = 0
                hr += 1


#  now go to read_taisen.py
if expt == "TMB1":
    fns, dats, cnstrctrs = _rt.filterRPSdats(expt, dates, visits=[1], domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=20, maxIGI=90000, MinWinLossRat=0, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=4)
else:
    fns, dats, cnstrs = _rt.filterRPSdats(expt, dates, visits=[1], domainQ=(_rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=300, max_meanIGI=20000, minIGI=10, maxIGI=30000, MinWinLossRat=0., has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)    
    #fns, dats, cnstrctrs = _rt.filterRPSdats(expt, dates, visits=[1, ], domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, minIGI=20, maxIGI=90000, MinWinLossRat=0, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

fp = open("%sfns.txt" % expt, "w")

for fn in fns:
    fp.write("%s\n" % fn)
fp.close()
    
print("Wrote %sfns.txt" % expt)

if expt == "TMB2":
    print("Run mv_Ken_Sam to add 2 old datasets to %sfns.txt to check repeatability." % expt)
