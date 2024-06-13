import numpy as np
from scipy.integrate import solve_ivp
def explicit_euler(f, t0, y0, h, n):
    t = np.empty(n + 1)
    t[0] = t0
    y = np.empty((n + 1, len(y0)))  # geändert
    y[0] = y0
    for k in range(n):
        t[k + 1] = t[k] + h
        y[k + 1] = y[k] + h * f(t[k], y[k])
    return t, y

M = np.array([[0.0, 1.0], [-2.0, 1.0]])
f = lambda t, y: np.dot(M, y)
t0 = 0.0
y0 = np.array([1.0, 0.0])
h = 0.1
n = 2

t, y = explicit_euler(f, t0, y0, h, n)
print(t,y)

# 6b
t_span = (0, 3)
sol_numeric = solve_ivp(f, t_span, y0, method='RK45')
print(sol_numeric)