import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 2
x = sp.Symbol('x')
fx = sp.tan(x) - sp.exp(x)
ffx = sp.diff(fx, x)
fffx = sp.diff(ffx,x)
x0 = 1.0

def newton_optimize(x0, ffx, fffx, tol=1e-10):
    it = 0
    while abs(ffx.subs(x, x0)) > tol:
        f_v = ffx.subs(x,x0)
        ff_v = fffx.subs(x,x0)
        x0 -= f_v / ff_v
        it += 1

    minimum_x = x0
    minimum_f = fx.subs(x, minimum_x)
    return minimum_x, minimum_f, it

print("Aufgabe 2:")
minimum_x, minimum_f, iterations = newton_optimize(x0, ffx, fffx)
print(f"Minimum bei x = {minimum_x}") #:.10f
print(f"f(x) am Minimum = {minimum_f}")
print(f"Anzahl der Iterationen: {iterations}")


def newton(equations, variables, initial_guess, tol, max_iterations):
    J = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])
    f = sp.Matrix(equations)
    x = sp.Matrix(initial_guess)
    it = 0

    for _ in range(max_iterations):
        f_val = f.subs(dict(zip(variables, x)))
        if f_val.norm() < tol:
            break
        J_val = J.subs(dict(zip(variables, x)))
        delta_x = J_val.LUsolve(-f_val)
        x += delta_x
        it += 1
    res_r = [float(val) for val in x]

    return res_r, it

x, y = sp.symbols('x y')
variables = [x, y]
max_iterations = 100

# Aufgabe 3
equations3 = [-2*x**3+3*y**2 + 42, 5*x**2 + 3*y**3 - 69]
initial_guess3 = [1.0, 1.0]
tol3 = 1e-8
result3, iterations3 = newton(equations3, variables, initial_guess3, tol3, max_iterations)
print(f"Aufgabe 3: Ergebnis: {result3}, Iterationsn: {iterations3}")

# Aufgabe 4
equations4 = [x*sp.exp(y)-1, y-1-x**2]
initial_guess4 = [-1.0, -1.0]
tol4 = 1e-6
result4, iterations4 = newton(equations4, variables, initial_guess4, tol4, max_iterations)
print(f"Aufgabe 4: Ergebnis: {result4}, Iterationen: {iterations4}")

# Aufgabe 5
equations5 = [20*x*y**2-3*x**3-50, 4*x**2-3*y**3-4]
initial_guess5 = [2.0, 1.0]
tol5 = 1e-6
result5, iterations5 = newton(equations5, variables, initial_guess5, tol5, max_iterations)
print(f"Aufgabe 5: Ergebnis: {result5}, Iterationen: {iterations5}")

# Aufgabe 6
equations6 = [6*x-sp.cos(x)-2*y,8*y-x*y**2-sp.sin(x)]
initial_guess6 = [0.0, 0.0]
tol6 = 1e-4
result6, iterations6 = newton(equations6, variables, initial_guess6, tol6, max_iterations)
print(f"Aufgabe 6: Ergebnis: {result6}, Iterationen: {iterations6}")

# Aufgabe 7a
"""
f1 = lambda x,y: x**2 - y - 1
f2 = lambda x,y: (x - 2)**2 + (y - 1/2)**2 - 1
x, y = np.linspace(-2, 4, 400),np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z1,Z2 = f1(X, Y), f2(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z1, levels=[0], colors='blue', label='$f_1(x, y) = 0$')
plt.contour(X, Y, Z2, levels=[0], colors='red', label='$f_2(x, y) = 0$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Höhenlinien von $f_1(x, y)$ und $f_2(x, y)$')
plt.legend()
plt.grid(True)
plt.show()
"""

# Aufgabe 7b
equations7 = [x**2-y-1,(x-2)**2+(y-1/2)**2-1]
initial_guess7a = [1.0, 0.1]
initial_guess7b = [1.5, 1.4]
tol7 = 1e-4
result7a, iterations7a = newton(equations7, variables, initial_guess7a, tol7, max_iterations)
result7b, iterations7b = newton(equations7, variables, initial_guess7b, tol7, max_iterations)
print(f"Aufgabe 7b x1: Ergebnis: {result7a}, Iterationen: {iterations7a}")
print(f"Aufgabe 7b x2: Ergebnis: {result7b}, Iterationen: {iterations7b}")
