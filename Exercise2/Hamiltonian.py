import numpy as np
import matplotlib.pyplot as plt 

p0 = 0.
x0 = 1.
m = 1.
omega = 1.0

def F(t, u):
    x, p = u
    return np.array([p/m, -omega**2 * x * m])

def Runge_Kutta(F, t, u, dt):
    K1 = F(t, u)
    K2 = F(t + (dt/2.), u + ((dt * K1)/2.))
    K3 = F(t + (dt/2.), u + ((dt * K2)/2.))
    K4 = F(t + dt, u + (dt * K3))
    u_np1 = u + (dt / 6.) * (K1 + 2. * K2 + 2. * K3 + K4)
    return u_np1

dt = 0.1
t = np.arange(0, 20 + dt, dt)
u = np.zeros((len(t), 2))
u[0] = [x0, p0]

for n in range(len(t) - 1):
    u[n + 1] = Runge_Kutta(F, t[n], u[n], dt)

plt.figure(figsize=(10, 8))
plt.plot(t, u[:,0], label="position x")
plt.plot(t, u[:,1], label="momentum p")
plt.legend(frameon=False, loc = 4)
plt.ylim([-1.2, 1.2])
plt.xlabel("t", fontsize=14)
plt.savefig("Hamiltonian_solution.png")
plt.show()