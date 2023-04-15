import numpy as np
import matplotlib.pyplot as plt
import math as mt
import time
import csv

from numba import njit, jit
from tqdm import trange

from solver import Solver
import functions as func

fieldnames = ["Rex", "Cf", "Cf_analit", "Cf_analit_turb"]
with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

P = 101325
L = 5
U0 = 10
T0 = 300
Tw = 300

PrT = 0.9

properties0 = func.AirProp(T0, T0, P)

den0 = properties0[0]
vis0 = properties0[1]
lam0 = properties0[2]
cp = properties0[3]

n = 2000
k = 300

Rel = den0 * U0 * L / vis0
Pr = vis0 * cp / lam0

# scale = 2
# h = scale * max(0.37 * L / (Rel ** 0.2),
#                 0.37 * L / (Rel ** 0.2 * Pr ** 0.6))

h = 0.5

xx, yy = func.GrdGen(L, h, n, k, a=1.01, b=1.026)

U = np.zeros((2,k), float)
Uit = np.zeros((k), float)

V = np.zeros((2,k), float)
Vit = np.zeros((k), float)

T = np.zeros((2,k), float)
Tit = np.zeros((k), float)

ka = np.zeros((2,k), float)
kait = np.zeros((k), float)

omega = np.zeros((2,k), float)
omegait = np.zeros((k), float)

gamma = np.zeros((2,k), float)
gammait = np.zeros((k), float)

den  = np.zeros((2,k), float)
denit = np.zeros((k), float)

vis = np.zeros((2,k), float)
visit = np.zeros((k), float)

visT = np.zeros((2,k), float)
visTit = np.zeros((k), float)

lam = np.zeros((2,k), float)
lamit = np.zeros((k), float)

S1 = np.zeros((k), float)
S2 = np.zeros((k), float)

for ii in trange(k):
    den[:,ii] = float(den0)

for ii in trange(k):
    vis[:,ii] = float(vis0)

for ii in trange(k):
    visT[:,ii] = 10 * vis[:,ii]

for ii in trange(k):
    lam[:,ii] = float(lam0)

Tu = 0.065

x0 = 1e-10

for jj in range(0, k):
    U[0,jj]            = U0 #mt.erf(0.313 * yy[jj] * np.sqrt(den0 * U0 / vis0 / x0))
    V[0,jj]            = 0.
    T[0,jj]            = T0
    # ka[0,jj] = 1e-5
    # omega[0,jj] = 1e-5
    ka[0,jj] = 1.5 * U0 * Tu * Tu
    Ret = den[0,jj] / vis[0,jj] * ((ka[0,jj]) ** 0.5) * yy[jj]
    Dq = 1. - np.exp(-0.022 * Ret)
    omega[0,jj] = 0.09 * Dq * den[0,jj] * ka[0,jj] / visT[0,jj]

    if yy[jj] / h <= 0.3:
        gamma[0,jj] = 0.
    else:
        gamma[0,jj] = 1.

U[0,0] = 0.
T[0,0] = Tw

Rex       = []
Cf        = []
Cf_analit = []
Cf_analit_turb = []
# Nu        = []
# Nu_analit = []

x = 0
i = 0
iternum = 200
counter = 0

sigmaom = 2.
Com1 = 5 / 9
Com2 = 3 / 40
gammamax = 1.1
sigmal = 5.
sigmagam = 0.2
sigmak = 2.
Cmu = 0.09

while x <= L:
    if i < n-1:
        dx = xx[i+1] - xx[i]
    else:
        dx = 1e-6
    x += dx
    i += 1

    start = time.time()

    for jj in range(k):
        U[1,jj] = U[0,jj]
        V[1,jj] = V[0,jj]
        T[1,jj] = T[0,jj]
        ka[1,jj] = ka[0,jj]
        omega[1,jj] = omega[0,jj]
        gamma[1,jj] = gamma[0,jj]
        den[1,jj] = den[0,jj]
        vis[1,jj] = vis[0,jj]
        visT[1,jj] = visT[0,jj]

    for q in range(iternum):
        for jj in range(k):
            Uit[jj] = U[1,jj]
            Vit[jj] = V[1,jj]
            Tit[jj] = T[1,jj]
            kait[jj] = ka[1,jj]
            omegait[jj] = omega[1,jj]
            gammait[jj] = gamma[1,jj]
            denit[jj] = den[1,jj]
            visit[jj] = vis[1,jj]
            visTit[jj] = visT[1,jj]

        # Momentum equation
        fcond = [0., 1., 0., 0.]
        lcond = [0., 1., 0., U0]
        for jj in range(1, k):
            S1[jj] = 0.
            S2[jj] = 0.
        U, V = Solver(U, den, vis, visT, fcond, lcond, U, V, dx, yy, S1, S2, theta=1, motion_eq=True)
        for kk in range(k):
            if func.isNaN(U[1,kk]):
                print('Programm iterrupt at Rex = ', U[1,k-1] * x * den0 / vis0)
                print('dx = ', dx)
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('U = Nan')
        for kk in range(k):
            if func.isNaN(V[1,kk]):
                print('Programm iterrupt at Rex = ', U[1,k-1] * x * den0 / vis0)
                print('dx = ', dx)
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('V = Nan')

        # # Energy equation
        # fcond = [0, 1, 0, T[1,0]]
        # lcond = [0, 1, 0, T[1,len(T[0])-1]]
        # for jj in range(1, len(S1)):
        #     S1[jj] = (vis[1,jj] + visT[1,jj]) * (U[1,jj] - U[1,jj-1]) * (U[1,jj] - U[1,jj-1]) / (yy[jj] - yy[jj-1]) / (yy[jj] - yy[jj-1])
        #     S2[jj] = 0
        # T = Solver(T, cp*den, lam, cp*visT/PrT, fcond, lcond, U, V, dx, yy, S1, S2)
        # for kk in range(len(T[1])):
        #     if isNaN(T[1,kk]):
        #         raise Exception('T = Nan')

        # omega equation
        ones = np.ones((2,k), float)
        fcond = [0., 1., 0., 6.*vis[1,0]/den[1,0]/Com2/yy[1]/yy[1]]
        # fcond = [0, -1/(yy[1]), 1/(yy[1]), 0]
        lcond = [-1./(yy[k-1] - yy[k-2]), 1./(yy[k-1] - yy[k-2]), 0., 0.]
        for jj in range(1, k-1):
            dyp = yy[jj+1] - yy[jj]
            dym = yy[jj]   - yy[jj-1]
            S = (0.5 ** 0.5) * ((dyp / dym * (U[1,jj] - U[1,jj-1]) + dym / dyp * (U[1,jj+1] - U[1,jj])) / (dym + dyp)) #(U[1,jj] - U[1,jj-1]) / (yy[jj] - yy[jj-1])
            S1[jj] = 2. * den[1,jj] * Com1 * S * S
            S2[jj] = - den[1,jj] * Com2 * omega[1,jj] * omega[1,jj]
            if func.isNaN(S1[jj]):
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('S1 in omega')
            if func.isNaN(S2[jj]):
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('S2 in omega')
        omega = Solver(omega, den, vis, visT/sigmaom, fcond, lcond, U, V, dx, yy, S1, S2, theta=0.04)
        for kk in range(k):
            if func.isNaN(omega[1,kk]):
                print('Programm iterrupt at Rex = ', U[1,k-1] * x * den0 / vis0)
                print('dx = ', dx)
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('omega = Nan')

        #gamma equation
        fcond = [0., -1./yy[1], 1./yy[1], 0.]
        lcond = [0., 1., 0., 1.]
        for jj in range(1, k-1):
            Rt = visT[1,jj] / vis[1,jj]
            Vort = np.abs((U[1,jj] - U[1,jj-1]) / (yy[jj] - yy[jj-1]))
            Tom = Rt * Vort / omega[1,jj]
            Rc = 400. - 360. * min(Tom / 2., 1.)
            Rnu = den[1,jj] * yy[jj] * yy[jj] * Vort / 2.188 / vis[1,jj]
            if (Rnu <= Rc) or (Rnu >= 100. / 0.7):
                Fgam = 0
            if (Rnu > Rc + 4.) and (Rnu <= 100./0.7 - 1):
                Fgam = 8.
            # Fgam = 2. * max(0., min(200. - Rnu, 1.)) * min(max(Rnu - Rc, 0.), 4.)
            Pgam = Fgam * Vort * (gammamax - gamma[1,jj]) * gamma[1,jj] ** 0.5
            Fturb = np.exp(-((Rnu * Rt) ** 1.2))
            if (Rnu <= 18.) or (Rnu >= 100.):
                Ggam = 0.
            if (Rnu > 19.) and (Rnu <= 99.):
                Ggam = 7.5
            # Ggam = 7.5 * max(0, min(100. - Rnu, 1.)) * min(max(Rnu - 18., 0.), 1.)
            Egam = Ggam * Fturb * Vort * gamma[1,jj] ** 1.5
            S1[jj] = den[1,jj] * (Pgam - Egam)
            S2[jj] = 0.
            if func.isNaN(S1[jj]):
                #graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('S1 in gamma')
            if func.isNaN(S2[jj]):
                #graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('S2 in gamma')
        gamma = Solver(gamma, den, vis/sigmal, visT/sigmagam, fcond, lcond, U, V, dx, yy, S1, S2, theta=0.5)
        for kk in range(k):
            if gamma[1,kk] < 0:
                gamma[1,kk] = 0.
            gamma[1,kk] = min(gamma[1,kk], 1.)
        for kk in range(k):
            if func.isNaN(gamma[1,kk]):
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('gamma = Nan')

        # ka equation
        fcond = [0., 1., 0., 0.]
        lcond = [-1./(yy[k-1] - yy[k-2]), 1./(yy[k-1] - yy[k-2]), 0., 0.]
        for jj in range(1, k-1):
            dyp = yy[jj+1] - yy[jj]
            dym = yy[jj]   - yy[jj-1]
            S = (0.5 ** 0.5) * ((dyp / dym * (U[1,jj] - U[1,jj-1]) + dym / dyp * (U[1,jj+1] - U[1,jj])) / (dym + dyp))
            Pk = gamma[1,jj] * min(2. * visT[1,jj] * S * S / den[1,jj], ka[1,jj] * np.abs(S) / (3. ** 0.5))
            S1[jj] = den[1,jj] * Pk
            S2[jj] = - Cmu * den[1,jj] * ka[1,jj] * omega[1,jj]
            if func.isNaN(S1[jj]):
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('S1 in ka')
            if func.isNaN(S2[jj]):
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('S2 in ka')
        ka = Solver(ka, den, vis, visT/sigmak, fcond, lcond, U, V, dx, yy, S1, S2, theta=0.05)
        for kk in range(k):
            # if ka[1,kk] < 0:
            #     ka[1,kk] = 1e-10
            if func.isNaN(ka[1,kk]):
                print('Programm iterrupt at Rex = ', U[1,k-1] * x * den0 / vis0)
                print('dx = ', dx)
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('ka = Nan')

        for jj in range(k):
            den[1,jj] = func.AirProp(T[1,jj], T0, P)[0]

        for jj in range(k):
            vis[1,jj] = func.AirProp(T[1,jj], T0, P)[1]

        for jj in range(k):
            visT[1,jj] = den[1,jj] * ka[1,jj] / omega[1,jj]
            if visT[1,jj] < 0:
                print('visT < 0')
                print('ka, omega = ', ka[1,jj], omega[1,jj])
                plt.subplot(1,3,1)
                plt.plot(ka[0], yy)
                plt.plot(ka[1], yy)
                plt.subplot(1,3,2)
                plt.plot(omega[0], yy)
                plt.plot(omega[1], yy)
                plt.subplot(1,3,3)
                plt.plot(gamma[0], yy)
                plt.plot(gamma[1], yy)
                plt.show()
                raise Exception('VisT < 0')
                visT[1,jj] = 0
            if func.isNaN(visT[1,jj]):
                print('ka, omega = ', ka[1,jj], omega[1,jj])
                func.graph(Rex, Cf, Cf_analit, Cf_analit_turb)
                raise Exception('VisT takes NaN')

        key1 = 0
        key1 = func.quality_chek(U, Uit, key1)

        key2 = 0
        key2 = func.quality_chek(V, Vit, key2)

        key3 = 0
        key3 = func.quality_chek(T, Tit, key3)

        key4 = 0
        key4 = func.quality_chek(den, denit, key4)

        key5 = 0
        key5 = func.quality_chek(vis, visit, key5)

        key6 = 0
        key6 = func.quality_chek(visT, visTit, key6)

        key6 = 0
        key6 = func.quality_chek(visT, visTit, key6)

        key7 = 0
        key7 = func.quality_chek(ka, kait, key7)

        key8 = 0
        key8 = func.quality_chek(omega, omegait, key8)

        key9 = 0
        key9 = func.quality_chek(gamma, gammait, key9)

        if (key1 == 0 and key2 == 0 and key3 == 0\
                      and key4 == 0 and key5 == 0\
                      and key6 == 0 and key7 == 0\
                      and key8 == 0 and key9 == 0):
            break

    for jj in range(k):
        U[0,jj] = U[1,jj]
        V[0,jj] = V[1,jj]
        T[0,jj] = T[1,jj]
        ka[0,jj] = ka[1,jj]
        omega[0,jj] = omega[1,jj]
        gamma[0,jj] = gamma[1,jj]
        den[0,jj] = den[1,jj]
        vis[0,jj] = vis[1,jj]
        visT[0,jj] = visT[1,jj]

    end = time.time()

    if q >= iternum * 0.75:
        counter += 1

    # if i % 50 == 0:
    print('|', 'step:', i, '; x:', round(x, 3), '; Re:', round(U[1,k-1] * x * den0 / vis0), '; step time:', round((end - start)*1000), 'ms', '; iter:', q, '; dx: ', dx, '; errors:', counter, '; progress', round(x * 100 / L, 3), '% |')

    Rex.append(U[1,k-1] * x * den0 / vis0)
    Cf.append(vis0 * ((U[1,1] - U[1,0]) / (yy[1] - yy[0]))/(den0 * U[1,k-1] * U[1,k-1]))
    Cf_analit.append(0.332/np.sqrt(U[1,k-1] * x * den0 / vis0))
    Cf_analit_turb.append(0.0296 * (U[1,k-1] * x * den0 / vis0) ** (-0.2))
    # Nu.append(-lam0 * x * ((T[1,1] - T[1,0]) / (yy[1] - yy[0])) / ((Tw - T[1,k-1]) * lam0))
    # Nu_analit.append(0.332 * (U[1,k-1] * x * den0 / vis0)**(0.5) * Pr**(0.333))

    if i % 10 == 0:
        with open('data.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                "Rex": (U[1,k-1] * x * den0 / vis0),
                "Cf": (vis0 * ((U[1,1] - U[1,0]) / (yy[1] - yy[0]))/(den0 * U[1,k-1] * U[1,k-1])),
                "Cf_analit": (0.332/np.sqrt(U[1,k-1] * x * den0 / vis0)),
                "Cf_analit_turb": (0.0296 * (U[1,k-1] * x * den0 / vis0) ** (-0.2)),
            }
            
            csv_writer.writerow(info)