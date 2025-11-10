import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation


n = 400
x0 = 0.
xf = 1.5
t0 = 0.
tf = 2.
c = 1.
h = xf / n
alpha = 1.
dt = (alpha * h) / c


x = np.linspace(x0, xf, n, endpoint=False)


def F(t, u):
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
        d2Phi_dx2[i] = (Phi_ghost[i + 2] - 2. * Phi_ghost[i + 1] + Phi_ghost[i]) / (h**2)
    
    dPi_dt = c**2 * d2Phi_dx2

    return np.array([Pi, dPi_dt]).T


def Runge_Kutta(F, t, u, dt):
    K1 = F(t, u)
    K2 = F(t + (dt/2.), u + ((dt * K1)/2.))
    K3 = F(t + (dt/2.), u + ((dt * K2)/2.))
    K4 = F(t + dt, u + (dt * K3))
    u_np1 = u + (dt / 6.) * (K1 + 2. * K2 + 2. * K3 + K4)
    return u_np1


t = np.arange(t0, tf + dt, dt)
Phi0 = np.sin(12. * np.pi * x)
Pi0 = 0.
u = np.zeros((len(t), n, 2))
u[0, :, 0] = Phi0
u[0, :, 1] = Pi0

for i in range(len(t) - 1):
    u[i + 1] = Runge_Kutta(F, t[i], u[i], dt)


plt.figure(figsize=(10, 8))
plt.plot(x, u[0, :, 0], label=r"$\Phi$")
plt.plot(x, u[0, :, 1], label=r"$\Pi$")
plt.legend(frameon=False, fontsize=12)
plt.xlabel("x", fontsize=14)
plt.title("t=0", fontsize=14)
plt.savefig("Method_of_lines_t0.png")
plt.show()