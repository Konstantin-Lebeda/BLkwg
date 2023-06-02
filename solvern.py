import numpy as np

from numba import jit, njit

# Функция SolVner() принимает на вход следующие параметры:
# Psin - переменная, относительно которой решается уравнение (искомая величина)
# alphan - свойства не зависящие от вязкости или турбулентной вязкости (в случае ур-я энергии и от теплопроводности)
# phin - свойства зависящие от вязкости (в случае ур-я энергии от теплопроводности)
# phinT - свойства зависящие от турбулентной вязкости
# fcondn, lcondn - граничные условия для формирования трёхдиагональной матрицы на стенке и в потоке соотвестственно
# Un, Vn - поля скоростей соответственно
# dxn - текущий шаг по х
# уу - массив значений у в каждом элементе сетки
# S1n, S2n - источниковые члены
# thetan - параметр явно-неявной схемы дискретизации
# omega1n, omega2n - коэффициенты линеаризации источниковых членов (defaUnlt = 0.99)
# motion_eq - флаг, используемый при решении ур-я движения, позволяющий совместно решить ур-е неразрывности. (defaUnlt = False)
# При решении !только! ур-я движения необходимо передать значение motion_eq=TrUne 

# Функция возвращает расчитанное значение на текущем шаге искомой величины Psin, и, в случае решения уравнения движения
# (если motion_eq=TrUne), вторым элеменетом возращает расчитанное значение Vn.

@njit
def Solver(Psi, alpha, phi, phiT, fcond, lcond, U, V, dx, yy, S1, S2,
           theta=0.49, omega1=0.99, omega2=0.99,
           motion_eq=False):

    first_index = 0
    last_index  = int(len(Psi[0]))

    # Повторное уточнение типа данных принятых массивов и переменных
    # (необходимо для корректной работы jit, не изменяет сами массивы)
    Psin = float(Psi)
    alphan = float(alpha)
    phin = float(phi)
    phinT = float(phiT)
    fcondn = float(fcond)
    lcondn = float(lcond)
    Un = float(U)
    Vn = float(V)
    dxn = float(dx)
    yyn = float(yy)
    S1n = float(S1)
    S2n = float(S2)
    thetan = float(theta)
    omega1n = float(omega1)
    omega2n = float(omega2)

    # Инициализация массивов
    a = np.zeros(last_index, float)
    b = np.zeros(last_index, float)
    c = np.zeros(last_index, float)
    d = np.zeros(last_index, float)

    # Граничные условия на стенке
    a[0] = fcondn[0]
    b[0] = fcondn[1]
    c[0] = fcondn[2]
    d[0] = fcondn[3]

    # Ур-я, зашитые в коэффициенты трёхдиагональной матрицы описаны в ModUnle solVner.odt
    for jj in range(first_index+1, last_index-1):
        dyp = yyn[jj+1] - yyn[jj]
        dym = yyn[jj]   - yyn[jj-1]

        a[jj] = - alphan[1,jj] * Vn[1,jj] * dyp / dym / (dym + dyp) -\
                thetan * ((phin[1,jj]   + phinT[1,jj]) + (phin[1,jj-1] + phinT[1,jj-1])) / dym / (dym + dyp)

        b[jj] = alphan[1,jj] * Un[1,jj] / dxn +\
                alphan[1,jj] * Vn[1,jj] * dyp / dym / (dym + dyp) -\
                alphan[1,jj] * Vn[1,jj] * dym / dyp / (dym + dyp) +\
                thetan * ((phin[1,jj+1] + phinT[1,jj+1]) + (phin[1,jj]   + phinT[1,jj]  )) / dyp / (dym + dyp) +\
                thetan * ((phin[1,jj]   + phinT[1,jj]  ) + (phin[1,jj-1] + phinT[1,jj-1])) / dym / (dym + dyp) +\
                    (1. - omega1n) * S1n[jj] / (Psin[1,jj] + 1e-32) +\
                    (1. - omega2n) * S2n[jj] / (Psin[1,jj] + 1e-32)

        c[jj] =   alphan[1,jj] * Vn[1,jj] * dym / dyp / (dym + dyp) -\
                thetan * ((phin[1,jj+1] + phinT[1,jj+1]) + (phin[1,jj]   + phinT[1,jj]  )) / dyp / (dym + dyp)

        d[jj] = alphan[1,jj] * Un[1,jj] / dxn * Psin[0,jj] +\
                    (1. - thetan) *\
                (((phin[0,jj+1] + phinT[0,jj+1]) + (phin[0,jj]   + phinT[0,jj]  )) * (Un[0,jj+1] - Un[0,jj]  ) / dyp -\
                 ((phin[0,jj]   + phinT[0,jj]  ) + (phin[0,jj-1] + phinT[0,jj-1])) * (Un[0,jj]   - Un[0,jj-1]) / dym) +\
                    omega1n * S1n[jj] +\
                    omega2n * S2n[jj]

    # Граничные условия в потоке
    a[last_index-1] = lcondn[0]
    b[last_index-1] = lcondn[1]
    c[last_index-1] = lcondn[2]
    d[last_index-1] = lcondn[3]

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
        Psin[1][jj] = d[jj]

    # Ур-е неразрывности
    if motion_eq:
        for jj in range(first_index+1, last_index):
            dym = yyn[jj]-yyn[jj-1]

            # Vn[1,jj] = 1. / alphan[1,jj] * (- dym / 2. / dxn * \
            #             (alphan[1,jj]   * Psin[1,jj]   - alphan[0,jj]   * Psin[0,jj]    + \
            #              alphan[1,jj-1] * Psin[1,jj-1] - alphan[0,jj-1] * Psin[0,jj-1]) + alphan[1,jj-1] * Vn[1,jj-1])

            Vn[1][jj] = - (0.5 * (Psin[1,jj]   - Psin[0,jj])   / dxn + \
                         0.5 * (Psin[1,jj-1] - Psin[0,jj-1]) / dxn) * dym + Vn[1,jj-1]
            
        return Psin, Vn

    else:
        return Psin