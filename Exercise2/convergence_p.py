import numpy as np 
import matplotlib.pyplot as plt 


def f(x):
    return np.exp(-1. * (x**2))

def df_exact(x):
    return -2. * x * np.exp(-1. * (x**2))

def df2_exact(x):
    return (4. * (x**2) - 2) * np.exp(-1. * (x**2))

def finite_diff(f, x, h):
    u_plus = f(x + h)
    u_minus = f(x - h)
    return (u_plus - u_minus) / (2. * h)

def finite_diff_second(f, x, h):
    u_plus = f(x + h)
    u_minus = f(x - h)
    u = f(x)
    return (u_plus - (2. * u) + u_minus) / (h**2.)

def Convergence_ratio(f, df_exact, df2_exact, h):
    x_h = np.arange(1e-5, 1 + h, h)
    du_h = finite_diff(f, x_h, h)
    du2_h = finite_diff_second(f, x_h, h)
    err_h = (df_exact(x_h) - du_h)
    err2_h = (df2_exact(x_h) - du2_h)


    h2 = h/2
    x_h2 = np.arange(1e-5, 1 + h2, h2)
    du_h2 = finite_diff(f, x_h2, h2)
    du2_h2 = finite_diff_second(f, x_h2, h2)
    err_h2 = (df_exact(x_h2) - du_h2)
    err2_h2 = (df2_exact(x_h2) - du2_h2)
    

    err_h2_interp = np.interp(x_h, x_h2, err_h2)
    err2_h2_interp = np.interp(x_h, x_h2, err2_h2)
    eps = 1e-15
    

    Q = (err_h + eps) / (err_h2_interp + eps)
    Q2 = (err2_h + eps) / (err2_h2_interp + eps)
    p = np.log2(Q)
    p2 = np.log2(Q2)

    return x_h, Q, p, Q2, p2

h = 0.1
x, Q, p, Q2, p2 = Convergence_ratio(f, df_exact, df2_exact, h)

plt.figure(figsize=(9, 8))
plt.plot(x, p, label="p(x) = First derivative")
plt.plot(x, p2, label="p(x) = Second derivative")
plt.xlabel("x", fontsize=14)
plt.ylabel("Convergence order p", fontsize=14)
plt.ylim([1.5, 2.5])
plt.legend(frameon=False, fontsize=12)
plt.savefig("p.png")
plt.show()

# In order to do the proper self convergence test, maybe try to do a loop over h and plot h(x).