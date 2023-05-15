import math as mt
import time
import numpy as np

from numba import njit, jit

@jit
def Solver(Psi, alpha, phi, phiT, fcond, lcond, U, V, dx, yy, S1, S2,
           theta=0.49, omega1=0.99, omega2=0.99,
           motion_eq=False):

    first_index = 0
    last_index  = len(Psi[0])

    # Повторное уточнение типа данных принятых массивов и переменных
    # (необходимо для корректной работы jit, не изменяет сами массивы)
    Psi = np.double(Psi)
    alpha = np.double(alpha)
    phi = np.double(phi)
    phiT = np.double(phiT)
    fcond = np.double(fcond)
    lcond = np.double(lcond)
    U = np.double(U)
    V = np.double(V)
    dx = np.double(dx)
    yy = np.double(yy)
    S1 = np.double(S1)
    S2 = np.double(S2)
    theta = np.double(theta)
    omega1 = np.double(omega1)
    omega2 = np.double(omega2)

    # Инициализация массивов
    a = np.zeros(last_index, float)
    b = np.zeros(last_index, float)
    c = np.zeros(last_index, float)
    d = np.zeros(last_index, float)

    # Граничные условия на стенке
    a[0] = fcond[0]
    b[0] = fcond[1]
    c[0] = fcond[2]
    d[0] = fcond[3]

    # Ур-я, зашитые в коэффициенты трёхдиагональной матрицы описаны в Module solver.odt
    for jj in range(first_index+1, last_index-1):
        dyp = yy[jj+1] - yy[jj]
        dym = yy[jj]   - yy[jj-1]

        a[jj] = - alpha[1,jj] * V[1,jj] * dyp / dym / (dym + dyp) -\
                theta * ((phi[1,jj]   + phiT[1,jj]) + (phi[1,jj-1] + phiT[1,jj-1])) / dym / (dym + dyp)

        b[jj] = alpha[1,jj] * U[1,jj] / dx +\
                alpha[1,jj] * V[1,jj] * dyp / dym / (dym + dyp) -\
                alpha[1,jj] * V[1,jj] * dym / dyp / (dym + dyp) +\
                theta * ((phi[1,jj+1] + phiT[1,jj+1]) + (phi[1,jj]   + phiT[1,jj]  )) / dyp / (dym + dyp) +\
                theta * ((phi[1,jj]   + phiT[1,jj]  ) + (phi[1,jj-1] + phiT[1,jj-1])) / dym / (dym + dyp) +\
                    (1. - omega1) * S1[jj] / (Psi[1,jj] + 1e-32) +\
                    (1. - omega2) * S2[jj] / (Psi[1,jj] + 1e-32)

        c[jj] =   alpha[1,jj] * V[1,jj] * dym / dyp / (dym + dyp) -\
                theta * ((phi[1,jj+1] + phiT[1,jj+1]) + (phi[1,jj]   + phiT[1,jj]  )) / dyp / (dym + dyp)

        d[jj] = alpha[1,jj] * U[1,jj] / dx * Psi[0,jj] +\
                    (1. - theta) *\
                (((phi[0,jj+1] + phiT[0,jj+1]) + (phi[0,jj]   + phiT[0,jj]  )) * (U[0,jj+1] - U[0,jj]  ) / dyp -\
                 ((phi[0,jj]   + phiT[0,jj]  ) + (phi[0,jj-1] + phiT[0,jj-1])) * (U[0,jj]   - U[0,jj-1]) / dym) +\
                    omega1 * S1[jj] +\
                    omega2 * S2[jj]

    # Граничные условия в потоке
    a[last_index-1] = lcond[0]
    b[last_index-1] = lcond[1]
    c[last_index-1] = lcond[2]
    d[last_index-1] = lcond[3]

    # TDMA
    for jj in range(first_index+1, last_index):
        b[jj] -= a[jj] * c[jj-1] / (b[jj-1] + 1e-32)
        d[jj] -= a[jj] * d[jj-1] / (b[jj-1] + 1e-32)
        a[jj]  = 0.

    d[last_index-1] /= (b[last_index-1] + 1e-32)
    b[last_index-1]  = 1.

    for jj in range(last_index-2, first_index-1, -1):
        d[jj] = (d[jj] - c[jj] * d[jj+1]) / (b[jj] + 1e-32)
        c[jj] = 0.
        b[jj] = 1.

    # Присвоение решения TDMA массиву исходной величины с учётом релаксации
    for jj in range(first_index, last_index):
        Psi[1,jj] = d[jj]

    # Ур-е неразрывности
    if motion_eq:
        for jj in range(first_index+1, last_index):
            dym = yy[jj]-yy[jj-1]

            V[1,jj] = 1. / alpha[1,jj] * (- dym / 2. / dx * \
                        (alpha[1,jj]   * Psi[1,jj]   - alpha[0,jj]   * Psi[0,jj]    + \
                         alpha[1,jj-1] * Psi[1,jj-1] - alpha[0,jj-1] * Psi[0,jj-1]) + alpha[1,jj-1] * V[1,jj-1])
            
        return Psi, V

    else:
        return Psi
    
def GrdGen(L, h,
           n = 200, k = 600,
           a = 1., b = 1.):
    sumX, sumY = 0, 0
    for ii in range(n-2):
        sumX += a ** ii
    for ii in range(k-2):
        sumY += b ** ii
    dx = L / sumX
    dy = h / sumY
    xx = [0. for _ in range(n)]
    yy = [0. for _ in range(k)]
    for ii in range(1, n):
        if (a ** (ii - 2)) * dx <= 1.5e-6:
            xx[ii] = xx[ii-1] + (a ** (ii - 2)) * dx
        elif (a ** (ii - 2)) * dx > 1.5e-6:
            xx[ii] = xx[ii-1] + 1.5e-6
    for ii in range(1, k):
        yy[ii] = yy[ii-1] + (b ** (ii - 2)) * dy
    return xx, yy

@njit
def quality_chek(X, X_it, key):
    accurasy = 1e-5 #1e-8
    for j in range(len(X_it)):
            delta = abs(X_it[j] - X[1][j]) / (X[1][j] + 10e-10)
            if (delta > accurasy):
                key = 1
                break
    return key

def AirProp(T, T0, P):
    vis0 = 1.85e-5
    lam = 0.026
    R = 8.314e3 / 29
    cp = 1005
    den = P / R / (T + 1e-10)
    vis = vis0 * (T / (T0 + 1e-10)) ** 75e-2
    return [den, vis, lam, cp]

P = 101325
U0 = 10
T0 = 300
Tw = 301
PrT = 0.9

properties0 = AirProp(T0, T0, P) # Расчёт вектора начальных свойств
den0 = properties0[0]
vis0 = properties0[1]
lam0 = properties0[2]
cp = properties0[3]

# Геометрия и параметры сетки
L = 5
h = 0.4
n = 2000
k = 500

xx, yy = GrdGen(L, h, n, k, a=1.01, b=1.023)

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

for ii in range(k):
    den[0][ii] = float(den0)
    den[1][ii] = float(den0)

for ii in range(k):
    vis[0][ii] = float(vis0)
    vis[1][ii] = float(vis0)

for ii in range(k):
    visT[0][ii] = 100 * vis[0][ii]
    visT[1][ii] = 100 * vis[1][ii]

for ii in range(k):
    lam[0][ii] = float(lam0)
    lam[1][ii] = float(lam0)

Tu = 0.065

for jj in range(0, k):
    U[0][jj]            = U0
    V[0][jj]            = 0.
    T[0][jj]            = T0

U[0][0] = 0.
T[0][0] = Tw

for jj in range(0, k):
    ka[0][jj] = 1.5 * U0 * Tu * Tu
    Ret = den[0][jj] / vis[0][jj] * ((ka[0][jj]) ** 0.5) * yy[jj]
    Dq = 1. - mt.exp(-0.022 * Ret)
    omega[0][jj] = 0.09 * Dq * den[0][jj] * ka[0][jj] / visT[0][jj]

    if yy[jj] / h <= 0.4:
        gamma[0][jj] = 0.
    else:
        gamma[0][jj] = 1.

# Инициализация списков для хранения некоторых рассчитываемых параметров
Rex       = []
Cf        = []
Cf_analit = []
Cf_analit_turb = []
# Nu        = []
# Nu_analit = []

x = 0
i = 0
iternum = 1000
counter = 0
q = 0

# Константы источниковых членов ур-ний турбулентности
sigmaom = 2.
Com1 = 5 / 9
Com2 = 3 / 40
gammamax = 1.1
sigmal = 5.
sigmagam = 0.2
sigmak = 2.
Cmu = 0.09

while x <=L:
    if i < n-1:
        dx = xx[i+1] - xx[i]
    else:
        dx = 1.5e-6
    # До определённого момента (i < n-1) сетка посчитана с плавным
    # переходом на постоянный шаг.
    # Далее шаг всегда постоянный

    x += dx # Текущее значение
    i += 1 # Счётчик для расчёта dx

    start = time.time() # Для оценки времени расчёта одного шага

    # Перенос значений с предыдущего шага на текущий
    for jj in range(k):
        U[1][jj] = U[0][jj]
        V[1][jj] = V[0][jj]
        T[1][jj] = T[0][jj]
        ka[1][jj] = ka[0][jj]
        omega[1][jj] = omega[0][jj]
        gamma[1][jj] = gamma[0][jj]
        den[1][jj] = den[0][jj]
        vis[1][jj] = vis[0][jj]
        visT[1][jj] = visT[0][jj]

    # Цикл расчёта текущего шага
    for q in range(iternum):
        # Формирование массивов для проверки сходимости
        for jj in range(k):
            Uit[jj] = U[1][jj]
            Vit[jj] = V[1][jj]
            Tit[jj] = T[1][jj]
            kait[jj] = ka[1][jj]
            omegait[jj] = omega[1][jj]
            gammait[jj] = gamma[1][jj]
            denit[jj] = den[1][jj]
            visit[jj] = vis[1][jj]
            visTit[jj] = visT[1][jj]

        # Momentum equation
        fcond = [0., 1., 0., 0.] # Граничные условия на стенке
        lcond = [0., 1., 0., U0] # Граничные условия в потоке
        for jj in range(1, k):
            S1[jj] = 0. # Источниковый член
            S2[jj] = 0. # Источниковый член
        U, V = Solver(U, den, vis, visT, fcond, lcond, U, V, dx, yy, S1, S2, theta=1, motion_eq=True)
        for jj in range(k):
            V[1][jj] = 0.5 * V[1][jj] + 0.5 * Vit[jj]
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
        fcond = [0., 1., 0., 6.*vis[1][1]/den[1][1]/Com2/yy[1]/yy[1]]
        lcond = [-1./(yy[k-1] - yy[k-2]), 1./(yy[k-1] - yy[k-2]), 0., 0.]
        for jj in range(1, k-1):
            dyp = yy[jj+1] - yy[jj]
            dym = yy[jj]   - yy[jj-1]
            S = (0.5 ** 0.5) * ((dyp / dym * (U[1][jj] - U[1][jj-1]) + dym / dyp * (U[1][jj+1] - U[1][jj])) / (dym + dyp))
            S1[jj] = 2. * den[1][jj] * Com1 * S * S
            S2[jj] = - den[1][jj] * Com2 * omega[1][jj] * omega[1][jj]
        omega = Solver(omega, den, vis, visT/sigmaom, fcond, lcond, U, V, dx, yy, S1, S2, theta=1)
        relax = 0.04
        for jj in range(k):
            omega[1][jj] = relax * omega[1][jj] + (1 - relax) * omegait[jj]

        #gamma equation
        fcond = [0., -1./yy[1], 1./yy[1], 0.]
        lcond = [0., 1., 0., 1.]
        for jj in range(1, k-1):
            dyp = yy[jj+1] - yy[jj]
            dym = yy[jj]   - yy[jj-1]
            Rt = visT[1][jj] / vis[1][jj]
            Vort = abs((dyp / dym * (U[1][jj] - U[1][jj-1]) + dym / dyp * (U[1][jj+1] - U[1][jj])) / (dym + dyp))
            Tom = Rt * Vort / omega[1][jj]
            Rc = 400. - 360. * min(Tom / 2., 1.)
            Rnu = den[1][jj] * yy[jj] * yy[jj] * Vort / 2.188 / vis[1][jj]
            Fgam = 2. * max(0., min(200. - Rnu, 1.)) * min(max(Rnu - Rc, 0.), 4.)
            if (Rnu <= Rc) or (Rnu >= 100. / 0.7):
                Fgam = 0.
            if (Rnu > Rc + 4.) and (Rnu <= 100./0.7 - 1.):
                Fgam = 8.
            Pgam = Fgam * Vort * (gammamax - gamma[1][jj]) * gamma[1][jj] ** 0.5
            Fturb = mt.exp(-((Rnu * Rt) ** 1.2))
            Ggam = 7.5 * max(0, min(100. - Rnu, 1.)) * min(max(Rnu - 18., 0.), 1.)
            if (Rnu <= 18.) or (Rnu >= 100.):
                Ggam = 0.
            if (Rnu > 19.) and (Rnu <= 99.):
                Ggam = 7.5
            Egam = Ggam * Fturb * Vort * (gamma[1][jj]) ** 1.5
            S1[jj] = den[1][jj] * (Pgam - Egam)
            S2[jj] = 0.
        for jj in range(k):
            jabu[0][jj] = vis[0][jj] / sigmal
            jabu[1][jj] = vis[1][jj] / sigmal
        for jj in range(k):
            kuva[0][jj] = visT[0][jj] / sigmagam
            kuva[1][jj] = visT[1][jj] / sigmagam
        gamma = Solver(gamma, den, jabu, kuva, fcond, lcond, U, V, dx, yy, S1, S2, theta=1)
        for kk in range(k):
            if gamma[1][kk] < 0:
                gamma[1][kk] = 0.
            gamma[1][kk] = min(gamma[1][kk], 1.)
        relax = 0.5
        for jj in range(k):
            gamma[1][jj] = relax * gamma[1][jj] + (1 - relax) * gammait[jj]
        
            
        # ka equation
        fcond = [0., 1., 0., 0.]
        lcond = [-1./(yy[k-1] - yy[k-2]), 1./(yy[k-1] - yy[k-2]), 0., 0.]
        for jj in range(1, k-1):
            dyp = yy[jj+1] - yy[jj]
            dym = yy[jj]   - yy[jj-1]
            S = (0.5 ** 0.5) * ((dyp / dym * (U[1][jj] - U[1][jj-1]) + dym / dyp * (U[1][jj+1] - U[1][jj])) / (dym + dyp))
            Pk = gamma[1][jj] * min(2. * visT[1][jj] * S * S / den[1][jj], ka[1][jj] * abs(S) / (3. ** 0.5))
            S1[jj] = den[1][jj] * Pk
            S2[jj] = - Cmu * den[1][jj] * ka[1][jj] * omega[1][jj]
        for jj in range(k):
            kuva[0][jj] = visT[0][jj] / sigmak
            kuva[1][jj] = visT[1][jj] / sigmak
        ka = Solver(ka, den, vis, kuva, fcond, lcond, U, V, dx, yy, S1, S2, theta=1)
        relax = 0.05
        for jj in range(k):
            ka[1][jj] = relax * ka[1][jj] + (1 - relax) * kait[jj]
            # if ka[1,jj] < 0:
            #     ka[1,jj] = 0.

        for jj in range(k):
            den[1][jj] = AirProp(T[1][jj], T0, P)[0]

        for jj in range(k):
            vis[1][jj] = AirProp(T[1][jj], T0, P)[1]

        for jj in range(k):
            visT[1][jj] = den[1][jj] * ka[1][jj] / omega[1][jj]
            if visT[1][jj] < 0:
                print('visT < 0')
                print('ka, omega = ', ka[1][jj], omega[1][jj])
                raise Exception('VisT < 0')

        key1 = 0
        key1 = quality_chek(U, Uit, key1)

        key2 = 0
        key2 = quality_chek(V, Vit, key2)

        key3 = 0
        key3 = quality_chek(T, Tit, key3)

        key4 = 0
        key4 = quality_chek(den, denit, key4)

        key5 = 0
        key5 = quality_chek(vis, visit, key5)

        key6 = 0
        key6 = quality_chek(visT, visTit, key6)

        key6 = 0
        key6 = quality_chek(visT, visTit, key6)

        key7 = 0
        key7 = quality_chek(ka, kait, key7)

        key8 = 0
        key8 = quality_chek(omega, omegait, key8)

        key9 = 0
        key9 = quality_chek(gamma, gammait, key9)

        if (key1 == 0 and key2 == 0 and key3 == 0\
                      and key4 == 0 and key5 == 0\
                      and key6 == 0 and key7 == 0\
                      and key8 == 0 and key9 == 0):
            break

    # Перенос значений с текущего шага на предыдущий
    for jj in range(k):
        U[0][jj] = U[1][jj]
        V[0][jj] = V[1][jj]
        T[0][jj] = T[1][jj]
        ka[0][jj] = ka[1][jj]
        omega[0][jj] = omega[1][jj]
        gamma[0][jj] = gamma[1][jj]
        den[0][jj] = den[1][jj]
        vis[0][jj] = vis[1][jj]
        visT[0][jj] = visT[1][jj]

    end = time.time()

    # Вывод некоторых параметров в консоль
    # if i % 50 == 0:
    print('|', 'step:', i, '; x:', round(x, 3), '; Re:', round(U[1][k-1] * x * den0 / vis0), '; step time:', round((end - start)*1000), 'ms', '; iter:', q, '; dx: ', dx, '; progress', round(x * 100 / L, 3), '% |')

    Rex.append(U[1][k-1] * x * den0 / vis0)
    Cf.append(vis0 * ((U[1][1] - U[1][0]) / (yy[1] - yy[0]))/(den0 * U[1][k-1] * U[1][k-1]))
    Cf_analit.append(0.332/mt.sqrt(U[1][k-1] * x * den0 / vis0))
    Cf_analit_turb.append(0.0296 * (U[1][k-1] * x * den0 / vis0) ** (-0.2))