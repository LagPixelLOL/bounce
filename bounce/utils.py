import numpy as np
import scipy as sp

def unit_diff(p1, p2):
    diff = p2 - p1
    return diff / np.sqrt(diff[0] ** 2 + diff[1] ** 2)

def overlapped_volume(f1, f2):
    return sp.integrate.nquad(lambda y, x: min(f1(x, y), f2(x, y)), np.array([[-np.inf, np.inf], [-np.inf, np.inf]]), opts={"limit": 256})[0]
