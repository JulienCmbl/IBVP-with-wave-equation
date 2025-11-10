import numpy as np
import matplotlib.pyplot as plt 


def Runge_Kutta(F, t, u, dt):
    K1 = F(t, u)
    K2 = F(t + (dt/2.), u + ((dt * K1)/2.))
    K3 = F(t + (dt/2.), u + ((dt * K2)/2.))
    K4 = F(t + dt, u + (dt * K3))
    u_np1 = u + (dt / 6.) * (K1 + 2. * K2 + 2. * K3 + K4)
    return u_np1

dt = 0.1
t = np.arange(0, 10 + dt, dt)
u = np.zeros(len(t))
u[0] = 1

def F(t, u):
    return -u

for n in range(len(t) - 1):
    u[n + 1] = Runge_Kutta(F, t[n], u[n], dt)


plt.plot(t, u, label="Runge-Kutta")
plt.plot(t, np.exp(-t), label="Analytical", linestyle='dashed')
plt.legend(frameon=False)
plt.xlabel("t", fontsize=14)
plt.ylabel("u(t)", fontsize=14)
plt.savefig("RK4_simplefunction.png")
plt.show()