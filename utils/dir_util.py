resultDir = "/Users/arai/nctc/Workspace/AIiRPS_Results"

def getResultFN(fn):
    return "%(rd)s/%(fn)s" % {"rd" : resultDir, "fn" : fn}
