import numpy as np
import matplotlib.pyplot as plt 
from numpy import interp



x0 = 0.
xf = 1.
t0 = 0.
tf = 1.
c = 1.
alpha = 0.5


def F(t, u, n):
    Phi = u[:, 0]
    Pi = u[:, 1]

    Phi_ghost = np.zeros(n + 2)
    Pi_ghost = np.zeros(n + 2)

    Phi_ghost[1:-1] = Phi
    Pi_ghost[1:-1] = Pi

    Phi_ghost[-1] = Phi_ghost[1]
    Phi_ghost[0] = Phi_ghost[-2]

    Pi_ghost[-1] = Pi_ghost[1]
    Pi_ghost[0] = Pi_ghost[-2]

    d2Phi_dx2 = np.zeros(n)
    for i in range(n):
        d2Phi_dx2[i] = (Phi_ghost[i + 2] - 2. * Phi_ghost[i + 1] + Phi_ghost[i]) / ((xf/n)**2)
    
    dPi_dt = c**2 * d2Phi_dx2

    return np.array([Pi, dPi_dt]).T


def Runge_Kutta(F, t, u, dt, n):
    K1 = F(t, u, n)
    K2 = F(t + (dt/2.), u + ((dt * K1)/2.), n)
    K3 = F(t + (dt/2.), u + ((dt * K2)/2.), n)
    K4 = F(t + dt, u + (dt * K3), n)
    u_np1 = u + (dt / 6.) * (K1 + 2. * K2 + 2. * K3 + K4)
    return u_np1


def solver(n):
    h = xf / n
    dt = (alpha * h) / c

    x = np.linspace(x0, xf, n, endpoint=False)

    t = np.arange(t0, tf + dt, dt)
    Phi0 = np.sin(12. * np.pi * x)
    Pi0 = 0.
    u = np.zeros((len(t), n, 2))
    u[0, :, 0] = Phi0
    u[0, :, 1] = Pi0

    for i in range(len(t) - 1):
        u[i + 1] = Runge_Kutta(F, t[i], u[i], dt, n)
    return x, u[-1, :, 0]

resolution = [100, 200, 400]
solutions = []
grids = []

for n in resolution:
    x, Phi = solver(n)
    grids.append(x)
    solutions.append(Phi)

u_h, u_h2, u_h4 = solutions
x_h, x_h2, x_h4 = grids

u_h2_interp = np.interp(x_h, x_h2, u_h2)
u_h4_interp = np.interp(x_h, x_h4, u_h4)

Error_12 = np.sqrt(np.mean((u_h - u_h2_interp)**2))
Error_24 = np.sqrt(np.mean((u_h2_interp - u_h4_interp)**2))

p = np.log2(Error_12/Error_24)
print(p)


plt.figure(figsize=(8,4))
plt.plot(x_h, u_h,  label=f'n={resolution[0]}')
plt.plot(x_h2, u_h2, '--', label=f'n={resolution[1]}')
plt.plot(x_h4, u_h4, ':',  label=f'n={resolution[2]}')
plt.xlabel('x', fontsize=14)
plt.ylabel(r'$\phi$(x,t)', fontsize=14)
plt.legend(frameon=False, fontsize=12)
plt.show()