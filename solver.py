import numpy as np

from numba import jit

# Функция Solver() принимает на вход следующие параметры:
# Psi - переменная, относительно которой решается уравнение (искомая величина)
# alpha - свойства не зависящие от вязкости или турбулентной вязкости (в случае ур-я энергии и от теплопроводности)
# phi - свойства зависящие от вязкости (в случае ур-я энергии от теплопроводности)
# phiT - свойства зависящие от турбулентной вязкости
# fcond, lcond - граничные условия для формирования трёхдиагональной матрицы на стенке и в потоке соотвестственно
# U, V - поля скоростей соответственно
# dx - текущий шаг по х
# уу - массив значений у в каждом элементе сетки
# S1, S2 - источниковые члены
# theta - параметр явно-неявной схемы дискретизации
# omega1, omega2 - коэффициенты линеаризации источниковых членов (default = 0.99)
# motion_eq - флаг, используемый при решении ур-я движения, позволяющий совместно решить ур-е неразрывности. (default = False)
# При решении !только! ур-я движения необходимо передать значение motion_eq=True 

# Функция возвращает расчитанное значение на текущем шаге искомой величины Psi, и, в случае решения уравнения движения
# (если motion_eq=True), вторым элеменетом возращает расчитанное значение V.

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