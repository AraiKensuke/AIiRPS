from AIiRPS.utils.read_taisen import return_hnd_dat
import numpy as _N
#james=May 26 13:25:53 2021
#mavi =May 26 14:14:28 2021
#Sarah=May 26 15:00:54 2021
#Ali=  May 26 15:18:58 2021

def give_me_the_time(partID, fields):
    hnd_dat, start, end = return_hnd_dat(partID, has_useragent=True, has_constructor=True)

    since_page_load = hnd_dat[-1, 3]
    all_seconds             = int(_N.round(since_page_load/1000))
    minutes                 = all_seconds // 60
    seconds                 = all_seconds - minutes*60

    fields[1] -= minutes
    fields[2] -= seconds
    if fields[2] < 0:
        fields[1] -= 1
        fields[2] += 60
    if fields[1] < 0:
        fields[0] -= 1
        fields[1] += 60

james=[13,25,53]
mavi =[14,14,28]
sarah =[15,00,54]
ali=[15,18,58]

#  21Jun09-1230-47   <-- this format in .dat file
give_me_the_time("20210526_1318-12", james)
give_me_the_time("20210526_1358-27", mavi)
give_me_the_time("20210526_1416-25", sarah)
give_me_the_time("20210526_1503-39", ali)
                 


