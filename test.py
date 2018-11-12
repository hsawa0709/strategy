import numpy as np
import entity
import math
from logging import (getLogger, StreamHandler, INFO, Formatter)

import time
import threading


def f(x):
    x = np.array(x)
    alpha = 100
    sigma = 10
    mu = np.array([10, 5])
    y =  (alpha / (math.sqrt(2*math.pi) * sigma) ) * math.exp(-np.dot(x-mu,x-mu) / 2 * sigma )
    return y

def g(x):
    x = np.array(x)
    mu = np.array([10,5])
    y = -np.dot(x-mu, x-mu)
    return y

if __name__ == "__main__":
    x = np.array([9,4])
    print(f(x))