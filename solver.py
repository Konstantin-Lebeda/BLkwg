import numpy as np
from numba import jit

@jit
def Solver(Psi, alpha, phi, phiT, fcond, lcond, U, V, dx, yy, S1, S2,
           theta=0.49, omega1=0.99, omega2=0.99,
           motion_eq=False):

    first_index = 0
    last_index  = len(Psi[0])

    Psi = np.float64(Psi)
    alpha = np.float64(alpha)
    phi = np.float64(phi)
    phiT = np.float64(phiT)
    fcond = np.float64(fcond)
    lcond = np.float64(lcond)
    U = np.float64(U)
    V = np.float64(V)
    dx = np.float64(dx)
    yy = np.float64(yy)
    S1 = np.float64(S1)
    S2 = np.float64(S2)
    theta = np.float64(theta)
    omega1 = np.float64(omega1)
    omega2 = np.float64(omega2)

    a = np.zeros(last_index, float)
    b = np.zeros(last_index, float)
    c = np.zeros(last_index, float)
    d = np.zeros(last_index, float)

    a[0] = fcond[0]
    b[0] = fcond[1]
    c[0] = fcond[2]
    d[0] = fcond[3]

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

    a[last_index-1] = lcond[0]
    b[last_index-1] = lcond[1]
    c[last_index-1] = lcond[2]
    d[last_index-1] = lcond[3]

    for jj in range(first_index+1, last_index):
        b[jj] -= a[jj] * c[jj-1] / b[jj-1]
        d[jj] -= a[jj] * d[jj-1] / b[jj-1]
        a[jj]  = 0.

    d[last_index-1] /= b[last_index-1]
    b[last_index-1]  = 1.

    for jj in range(last_index-2, first_index-1, -1):
        d[jj] = (d[jj] - c[jj] * d[jj+1]) / b[jj]
        c[jj] = 0.
        b[jj] = 1.

    for jj in range(first_index, last_index):
        Psi[1,jj] = d[jj]

    if motion_eq:
        for jj in range(first_index+1, last_index):
            dym = yy[jj]-yy[jj-1]

            V[1,jj] = 1. / alpha[1,jj] * (- dym / 2. / dx * \
                        (alpha[1,jj]   * Psi[1,jj]   - alpha[0,jj]   * Psi[0,jj]    + \
                         alpha[1,jj-1] * Psi[1,jj-1] - alpha[0,jj-1] * Psi[0,jj-1]) + alpha[1,jj-1] * V[1,jj-1])

        return Psi, V

    else:
        return Psi