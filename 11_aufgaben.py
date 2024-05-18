import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
"""
# Aufgabe 2
f = lambda t, y : y/t-t**2/2
y0 = 4
t0 = 2
h = 1
y1 = y0 + h*f(t0,y0)
t1 = t0 + h          

y2 = y1 + h*f(t1,y1) 
t2 = t1 + h          

y3 = y2 + h*f(t2,y2) 
t3 = t2 + h         

print("t1=",t1, "y1=",y1)
print("t2=",t2, "y2=",y2)
print("t3=",t3, "y1=",y3)

# Fehler
y_ex = lambda t : 3*t-t**3/4
print("t0=",t0, "err=", abs(y0-y_ex(t0)))
print("t1=",t1, "err=", abs(y1-y_ex(t1)))
print("t2=",t2, "err=", abs(y2-y_ex(t2)))
print("t3=",t3, "err=", abs(y3-y_ex(t3)))

# Aufgabe 3a
t, n = sp.symbols('t n')
dn_dt = -0.5 * n**(3/2) + 400 * (1 - sp.exp(-2*t))
n0 = 100
h = 0.02
end = 3
anzahl_schritte = int(end/h)
n_werte = [n0]
for i in range(anzahl_schritte):
    n_neu = n_werte[-1] + h * dn_dt.subs({n: n_werte[-1], t: i * h})
    n_werte.append(n_neu)
print(f"Aufgabe 3a: {n_werte[-1]}")

# Aufgabe 3b

t = sp.symbols('t')
n = sp.Function('n')
dn_dt = -0.5 * n(t)**(3/2) + 400 * (1 - sp.exp(-2*t))
n0_3b = {n(0): 100}
solution_sympy = sp.dsolve(sp.Derivative(n(t), t) - dn_dt, n(t), ics=n0_3b)
def dn_dt_numeric(t, n):
    return -0.5 * n**(3/2) + 400 * (1 - np.exp(-2*t))
t_span = (0, 3)
sol_numeric = solve_ivp(dn_dt_numeric, t_span, [n0_3b[n(0)].evalf()], t_eval=[3])
print(f"Aufgabe 3b: {sol_numeric.y[0][-1]}")

sol_numeric = solve_ivp(dn_dt, t_span, n0, method='RK45')
"""
# Aufgabe 3c
# noch erledigen!!!!


# Aufgabe 4
t = sp.symbols('t')
x, y = sp.symbols('x y', cls=sp.Function)
dx_dt = x(t) - y(t)*t
dy_dt = t + y(t)
x0 = 1
y0 = 1
h = 0.4
endzeit = 1.2
anzahl_schritte = int(endzeit / h + 1)
print(anzahl_schritte)
x_werte = [x0]
y_werte = [y0]
for i in range(anzahl_schritte):
    x_neu = x_werte[-1] + h * dx_dt.subs({x(t): x_werte[-1], y(t): y_werte[-1]})
    y_neu = y_werte[-1] + h * dy_dt.subs({y(t): y_werte[-1], t: i * h})
    x_werte.append(x_neu)
    y_werte.append(y_neu)
    
print("Aufgabe 4:")
for t_wert, x_wert, y_wert in zip(np.arange(0, endzeit+h, h), x_werte, y_werte):
    print(f"t = {t_wert}: x = {x_wert}, y = {y_wert}")

# Aufgae 5
t = sp.symbols('t')
x, y, u, v = sp.symbols('x y u v', cls=sp.Function)

dx_dt = u(t)
du_dt = -((u(t)*sp.sqrt(u(t)**2 + v(t)**2))**2) + v(t)**2
dy_dt = v(t)
dv_dt = -((v(t)*sp.sqrt(u(t)**2 + v(t)**2))**2) + u(t)**2 - 10

ics = {x(0): 0, u(0): 10, y(0): 0, v(0): 0}

system_of_eqs = (sp.Eq(sp.diff(x(t), t), dx_dt),
                 sp.Eq(sp.diff(u(t), t), du_dt),
                 sp.Eq(sp.diff(y(t), t), dy_dt),
                 sp.Eq(sp.diff(v(t), t), dv_dt))

solution = sp.dsolve(system_of_eqs, ics=ics)
print(solution)



