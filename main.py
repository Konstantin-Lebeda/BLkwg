# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import time
import csv

from numba import njit, jit
from tqdm import trange

from solver import Solver
import functions as func

# Начальные условия среды
P = 101325.
U0 = 25.
T0 = 300.
Tw = 301.
PrT = 0.9

properties0 = func.AirProp(T0, T0, P) # Расчёт вектора начальных свойств
den0 = properties0[0]
vis0 = properties0[1]
lam0 = properties0[2]
cp = properties0[3]

# Геометрия и параметры сетки
L = 5.
h = 0.4
n = 2000
k = 500

# По х сетка строится таким образом, чтобы за указанное число элементов
# (n) выйти на фиксированные шаг (в данном случае 5е-7).
# Сетка по у генерируется стандартным способом по кол-ву элементов и степени
# сжатия к нижней стенке
xx, yy = func.GrdGen(L, h, n, k, a=1.01, b=1.023)

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

den[:,:] = float(den0)

vis[:,:] = float(vis0)

visT[:,:] = 100 * vis[:,:]

lam[:,:] = float(lam0)

Tu = 0.065


U[0,:] = U0
V[0,:] = 0.
T[0,:] = T0

U[0,0] = 0.
T[0,0] = Tw


ka[0,:] = 1.5 * U0 * Tu * Tu
omega[0,:] = 0.09 * (1. - np.exp(-0.022 * (den[0,:] / vis[0,:] * ((ka[0,:]) ** 0.5) * yy[:]))) * den[0,:] * ka[0,:] / visT[0,:]

for jj in range(k):
    if yy[jj] / h <= 0.3:
        gamma[0,jj] = 0.
    else:
        gamma[0,jj] = 1.

# Инициализация списков для хранения некоторых рассчитываемых параметров
Rex       = []
Cf        = []
Cf_analit = []
Cf_analit_turb = []

x = 0.
i = 0
iternum = 1000
counter = 0

# Константы источниковых членов ур-ний турбулентности
sigmaom = 2.
Com1 = 5 / 9
Com2 = 3 / 40
gammamax = 1.1
sigmal = 5.
sigmagam = 0.2
sigmak = 2.
Cmu = 0.09

# Инициализация и очистка файла для мониторинга (script.py)
fieldnames = ["Rex", "Cf", "Cf_analit", "Cf_analit_turb"]
with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

Ures = open('res/U.txt', 'w')
Vres = open('res/V.txt', 'w')
kares = open('res/ka.txt', 'w')
wres = open('res/w.txt', 'w')
gres = open('res/g.txt', 'w')

ypres = open('res/yp.txt', 'w')
kapres = open('res/kap.txt', 'w')
epspres = open('res/epsp.txt', 'w')

while x <=L:
    if i < n-1:
        dx = xx[i+1] - xx[i]
    else:
        dx = 5e-8
    # До определённого момента (i < n-1) сетка посчитана с плавным
    # переходом на постоянный шаг.
    # Далее шаг всегда постоянный

    x += dx # Текущее значение
    i += 1 # Счётчик для расчёта dx

    start = time.time() # Для оценки времени расчёта одного шага

    # Перенос значений с предыдущего шага на текущий
    U[1,:] = U[0,:]
    V[1,:] = V[0,:]
    T[1,:] = T[0,:]
    ka[1,:] = ka[0,:]
    omega[1,:] = omega[0,:]
    gamma[1,:] = gamma[0,:]
    den[1,:] = den[0,:]
    vis[1,:] = vis[0,:]
    visT[1,:] = visT[0,:]

    # Цикл расчёта текущего шага
    for q in range(iternum):
        # Формирование массивов для проверки сходимости
        Uit[:] = U[1,:]
        Vit[:] = V[1,:]
        Tit[:] = T[1,:]
        kait[:] = ka[1,:]
        omegait[:] = omega[1,:]
        gammait[:] = gamma[1,:]
        denit[:] = den[1,:]
        visit[:] = vis[1,:]
        visTit[:] = visT[1,:]

        # Momentum equation
        fcond = np.array([0., 1., 0., 0.], float) # Граничные условия на стенке
        lcond = np.array([0., 1., 0., U0], float) # Граничные условия в потоке
        S1[1:] = 0. # Источниковый член
        S2[1:] = 0. # Источниковый член
        U = Solver(np.array(U, float), den, vis, visT, fcond, lcond, np.array(U, float), np.array(V, float), dx, yy, S1, S2, theta=1)
        for jj in range(1, k):
            dym = yy[jj]-yy[jj-1]
            V[1,jj] = - (0.5 * (U[1,jj]   - U[0,jj])   / dx + \
                         0.5 * (U[1,jj-1] - U[0,jj-1]) / dx) * dym + V[1,jj-1]
        V[1,:] = 0.5 * V[1,:] + 0.5 * Vit[:]
        U = np.array(U, float)
        V = np.array(V, float)
        # Для уточнения см. solver.py
        # Далее формат вызова функции Solver() будет происходит аналогично.

        # # Energy equation
        # fcond = [0, 1, 0, T[1,0]]
        # lcond = [0, 1, 0, T[1,len(T[0])-1]]
        # for jj in range(1, len(S1)):
        #     S1[jj] = (vis[1,jj] + visT[1,jj]) * (U[1,jj] - U[1,jj-1]) * (U[1,jj] - U[1,jj-1]) / (yy[jj] - yy[jj-1]) / (yy[jj] - yy[jj-1])
        #     S2[jj] = 0
        # T = Solver(T, cp*den, lam, cp*visT/PrT, fcond, lcond, U, V, dx, yy, S1, S2)
        # for kk in range(len(T[1])):
        #     if isNaN(T[1,kk]):ka

        # omega equation
        fcond = np.array([0., 1., 0., 6.*vis[1,1]/den[1,1]/Com2/yy[1]/yy[1]], float)
        lcond = np.array([-1./(yy[k-1] - yy[k-2]), 1./(yy[k-1] - yy[k-2]), 0., 0.], float)
        for jj in range(1, k-1):
            dyp = yy[jj+1] - yy[jj]
            dym = yy[jj]   - yy[jj-1]
            S = (0.5 ** 0.5) * ((dyp / dym * (U[1,jj] - U[1,jj-1]) + dym / dyp * (U[1,jj+1] - U[1,jj])) / (dym + dyp))
            S1[jj] = 2. * den[1,jj] * Com1 * S * S
            S2[jj] = - den[1,jj] * Com2 * omega[1,jj] * omega[1,jj]
        omega = Solver(np.array(omega, float), den, vis, visT/sigmaom, fcond, lcond, U, V, dx, yy, S1, S2, theta=1)
        relax = 0.04
        omega[1,:] = relax * omega[1,:] + (1. - relax) * omegait[:]

        #gamma equation
        fcond = np.array([0., -1./yy[1], 1./yy[1], 0.], float)
        lcond = np.array([0., 1., 0., 1.], float)
        for jj in range(1, k-1):
            Rt = visT[1,jj] / vis[1,jj]
            Vort = np.abs((dyp / dym * (U[1,jj] - U[1,jj-1]) + dym / dyp * (U[1,jj+1] - U[1,jj])) / (dym + dyp))
            Tom = Rt * Vort / omega[1,jj]
            Rc = 400. - 360. * min(Tom / 2., 1.)
            Rnu = den[1,jj] * yy[jj] * yy[jj] * Vort / 2.188 / vis[1,jj]
            if (Rnu <= Rc) or (Rnu >= 100. / 0.7):
                Fgam = 0
            if (Rnu > Rc + 4.) and (Rnu <= 100./0.7 - 1):
                Fgam = 8.
            Pgam = Fgam * Vort * (gammamax - gamma[1,jj]) * gamma[1,jj] ** 0.5
            Fturb = np.exp(-((Rnu * Rt) ** 1.2))
            if (Rnu <= 18.) or (Rnu >= 100.):
                Ggam = 0.
            if (Rnu > 19.) and (Rnu <= 99.):
                Ggam = 7.5
            Egam = Ggam * Fturb * Vort * (gamma[1,jj]) ** 1.5
            S1[jj] = den[1,jj] * (Pgam - Egam)
            S2[jj] = 0.
        gamma = Solver(np.array(gamma, float), den, vis/sigmal, visT/sigmagam, fcond, lcond, U, V, dx, yy, S1, S2, theta=1)
        for kk in range(k):
            if gamma[1,kk] < 0:
                gamma[1,kk] = 0.
            gamma[1,kk] = min(gamma[1,kk], 1.)
        relax = 0.5
        gamma[1,:] = relax * gamma[1,:] + (1. - relax) * gammait[:]
        
            
        # ka equation
        fcond = np.array([0., 1., 0., 0.], float)
        lcond = np.array([-1./(yy[k-1] - yy[k-2]), 1./(yy[k-1] - yy[k-2]), 0., 0.], float)
        for jj in range(1, k-1):
            dyp = yy[jj+1] - yy[jj]
            dym = yy[jj]   - yy[jj-1]
            S = (0.5 ** 0.5) * ((dyp / dym * (U[1,jj] - U[1,jj-1]) + dym / dyp * (U[1,jj+1] - U[1,jj])) / (dym + dyp))
            Pk = gamma[1,jj] * min(2. * visT[1,jj] * S * S / den[1,jj], ka[1,jj] * np.abs(S) / (3. ** 0.5))
            S1[jj] = den[1,jj] * Pk
            S2[jj] = - Cmu * den[1,jj] * ka[1,jj] * omega[1,jj] #0.
        ka = Solver(np.array(ka, float), den, vis, visT/sigmak, fcond, lcond, U, V, dx, yy, S1, S2, theta=1)
        relax = 0.05
        ka[1,:] = relax * ka[1,:] + (1. - relax) * kait[:]


        for jj in range(k):
            den[1,jj] = func.AirProp(T[1,jj], T0, P)[0]

        for jj in range(k):
            vis[1,jj] = func.AirProp(T[1,jj], T0, P)[1]

        visT[1,:] = den[1,:] * ka[1,:] / omega[1,:]

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

    # Перенос значений с текущего шага на предыдущий
    U[0,:] = U[1,:]
    V[0,:] = V[1,:]
    T[0,:] = T[1,:]
    ka[0,:] = ka[1,:]
    omega[0,:] = omega[1,:]
    gamma[0,:] = gamma[1,:]
    den[0,:] = den[1,:]
    vis[0,:] = vis[1,:]
    visT[0,:] = visT[1,:]

    end = time.time()

    # Вывод некоторых параметров в консоль
    # if i % 50 == 0:
    print('|', 'step:', i, '; x:', round(x, 3), '; Re:', round(U[1,k-1] * x * den0 / vis0), '; mSPI:', round((end - start)*1000 / q), '; iter:', q, '; dx: ', dx, '; progress', round(x * 100 / L, 3), '% |')
    
    if i % 5000 == 0:
        for jj in range(k):
            UU = U[1,jj]
            VV = V[1,jj]
            kaka = ka[1,jj]
            ww = omega[1,jj]
            gg = gamma[1,jj]
            Ures.write(str(UU))
            Ures.write(', ')
            Vres.write(str(VV))
            Vres.write(', ')
            kares.write(str(kaka))
            kares.write(', ')
            wres.write(str(ww))
            wres.write(', ')
            gres.write(str(gg))
            gres.write(', ')
        Ures.write('\n')
        Vres.write('\n')
        kares.write('\n')
        wres.write('\n')
        gres.write('\n')

    Rex.append(U[1,k-1] * x * den0 / vis0)
    Cf.append(vis0 * ((U[1,1] - U[1,0]) / (yy[1] - yy[0]))/(den0 * U[1,k-1] * U[1,k-1]))
    Cf_analit.append(0.332/np.sqrt(U[1,k-1] * x * den0 / vis0))
    Cf_analit_turb.append(0.0296 * (U[1,k-1] * x * den0 / vis0) ** (-0.2))

    # Запись в файл параметров для мониторинга (script.py)
    if i == 1:
        with open('data.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                "Rex": (U[1,k-1] * x * den0 / vis0),
                "Cf": (vis0 * ((U[1,1] - U[1,0]) / (yy[1] - yy[0]))/(den0 * U[1,k-1] * U[1,k-1])),
                "Cf_analit": (0.332/np.sqrt(U[1,k-1] * x * den0 / vis0)),
                "Cf_analit_turb": (0.0296 * (U[1,k-1] * x * den0 / vis0) ** (-0.2)),
            }
            
            csv_writer.writerow(info)

    if i % 100 == 0:
        with open('data.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                "Rex": (U[1,k-1] * x * den0 / vis0),
                "Cf": (vis0 * ((U[1,1] - U[1,0]) / (yy[1] - yy[0]))/(den0 * U[1,k-1] * U[1,k-1])),
                "Cf_analit": (0.332/np.sqrt(U[1,k-1] * x * den0 / vis0)),
                "Cf_analit_turb": (0.0296 * (U[1,k-1] * x * den0 / vis0) ** (-0.2)),
            }
            
            csv_writer.writerow(info)


for jj in range(k):
    UU = U[1,jj]
    VV = V[1,jj]
    kaka = ka[1,jj]
    ww = omega[1,jj]
    gg = gamma[1,jj]
    Ures.write(str(UU))
    Ures.write(', ')
    Vres.write(str(VV))
    Vres.write(', ')
    kares.write(str(kaka))
    kares.write(', ')
    wres.write(str(ww))
    wres.write(', ')
    gres.write(str(gg))
    gres.write(', ')
Ures.write('\n')
Vres.write('\n')
kares.write('\n')
wres.write('\n')
gres.write('\n')