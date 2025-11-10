import numpy as np 
import matplotlib.pyplot as plt 



def f(x):
    return (x-0.5)**3 + (x-0.5)**2 + x

def df_exact(x):
    return 3*(x-0.5)**2 + 2*(x-0.5) + 1

def finite_diff(f, x, h):
    u_plus = f(x + h)
    u_minus = f(x - h)
    return (u_plus - u_minus) / (2 * h)

def finite_diff_second(f, x, h):
    u_plus = f(x + h)
    u_minus = f(x - h)
    u = f(x)
    return (u_plus - 2*u + u_minus) / (h**2)

def C(f, df_exact, h):
    x_h = np.arange(0, 1 + h, h)
    du_h = finite_diff(f, x_h, h)
    err_h = (df_exact(x_h) - du_h)

    h2 = h/2
    x_h2 = np.arange(0, 1 + h2, h2)
    du_h2 = finite_diff(f, x_h2, h2)
    err_h2 = (df_exact(x_h2) - du_h2)

    Value_C = err_h 
    Value_Ch2 = err_h2 

    return x_h, x_h2, Value_C, Value_Ch2


h = 0.6
x, x2, Value_C, Value_Ch2 = C(f, df_exact, h)

plt.figure(figsize=(7, 4))
plt.plot(x, np.abs(Value_C), label="h=0.6")
plt.plot(x2, np.abs(Value_Ch2), label="h/2=0.3")
plt.xlabel("x")
plt.ylim([0,0.38])
plt.ylabel(r"$|f'(x) - f'^{(h)}(x)|$")
plt.legend(frameon=False, loc=4)
plt.show()