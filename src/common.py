import numpy as np

def g(iter, k, l):
    gamma = 2.2
    ans = np.power(iter / k, gamma) * l
    return ans