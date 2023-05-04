import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from tqdm import trange

def GrdGen(L, h,
           n = 200, k = 600,
           a = 1, b = 1):
    sumX, sumY = 0, 0
    for ii in range(n-2):
        sumX += a ** ii
    for ii in range(k-2):
        sumY += b ** ii
    dx = L / sumX
    dy = h / sumY
    xx = np.zeros((n), float)
    yy = np.zeros((k), float)
    for ii in trange(1, n, desc='dx'):
        if (a ** (ii - 2)) * dx <= 1.5e-6:
            xx[ii] = xx[ii-1] + (a ** (ii - 2)) * dx
        elif (a ** (ii - 2)) * dx > 1.5e-6:
            xx[ii] = xx[ii-1] + 1.5e-6
    for ii in trange(1, k, desc='dy'):
        yy[ii] = yy[ii-1] + (b ** (ii - 2)) * dy
    return xx, yy

def GrdGenSimpleY(L, h,
           n = 200, k = 600,
           a = 1, b = 1):
    sumX = 0
    for ii in range(n-2):
        sumX += a ** ii
    dx = L / sumX
    xx = np.zeros((n), float)
    yy = np.zeros((k), float)
    for ii in trange(1, n, desc='dx'):
        if (a ** (ii - 2)) * dx <= 1e-6:
            xx[ii] = xx[ii-1] + (a ** (ii - 2)) * dx
        elif (a ** (ii - 2)) * dx > 1e-6:
            xx[ii] = xx[ii-1] + 1e-6
    for ii in trange(1, k, desc='dy'):
        yy[ii] = yy[ii-1] + 1e-4
    return xx, yy

@njit
def quality_chek(X, X_it, key):
    accurasy = 1e-5 #1e-8
    for j in range(len(X_it)):
            delta = np.abs(X_it[j] - X[1,j]) / (X[1,j] + 10e-10)
            if (delta > accurasy):
                key = 1
                break
    return key

def AirProp(T, T0, P):
    vis0 = 1.85e-5
    lam = 0.026
    R = 8.314e3 / 29
    cp = 1005
    try:
        den = P / R / T
    except ZeroDivisionError:
        raise Exception('Division by zero')
    vis = vis0 * (T / (T0 + 1e-10)) ** 75e-2
    return [den, vis, lam, cp]

def graph(Rex, Cf, Cf_analit, Cf_analit_turb):
    plt.plot(Rex, Cf, marker = 'o', markersize = 2, linewidth = 1, color='r')
    plt.plot(Rex, Cf_analit, linewidth = 1, color='black')
    plt.plot(Rex, Cf_analit_turb, linewidth = 1, color='black')
    plt.xlabel('Rex')
    plt.ylabel('Cf/2')
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

def isNaN(num):
    return num != num