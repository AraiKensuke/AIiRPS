def gauKer(w):
    """
    1-D gaussian kernel.  Use with numpy.convolve
    """
    wf = _N.empty(8*w+1)

    for i in range(-4*w, 4*w+1):
        wf[i+4*w] = _N.exp(-0.5*(i*i)/(w*w))
    return wf

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm
