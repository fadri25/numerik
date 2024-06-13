import sympy as sp
import numpy as np
import math
"""
#1a 
x = sp.Symbol('x')
fx = sp.exp(-x)*sp.sin(2*x) # Formel anpassen
ffx = sp.diff(fx, x)
fffx = sp.diff(ffx,x)
ffffx = sp.diff(fffx,x) # Das wäre die dritte usw....

print(f'Erste Ableitung: {ffx}')
print(f"Zweite Ableitung: {fffx}")
print(f"dritte: {ffffx}")

# 1b
x = sp.Symbol('x')
x0 = 1.0  # Stelle x0

f = sp.exp(-x)*sp.sin(2*x)  # Funktion
f0 = f
f1 = sp.diff(f, x)
f2 = sp.diff(f1, x)
f3 = sp.diff(f2,x)
fk_list = [f0.subs(x, x0), f1.subs(x, x0), f2.subs(x, x0),f3.subs(x,x0)]

def taylor_factory(x0, fk_list):
    n = len(fk_list)
    x = sp.Symbol('x')
    taylor_polynom = sum(fk_list[k] * (x - x0)**k / sp.factorial(k) for k in range(n))
    return taylor_polynom

# Taylor-Polynom zweiter Ordnung
t2 = taylor_factory(x0, fk_list)
print("Taylor-Polynommm:", t2)

# 1c
# Fehler an Stelle x
x1 = 0.0  # Fehlerstelle
f0_x1 = f.subs(x, x1)
t2_x1 = t2.subs(x, x1)
print("Originalfunktion an x1:", f0_x1)
print("Taylor-Polynom an x1:", t2_x1)
app = t2_x1 - f0_x1
print("Approximation:", app)

# 2a
def bisection(f, a, b, max=3):
    iterations = 0
    assert(f(a) * f(b) < 0.0)
    while iterations < max:
        iterations += 1
        m = (a + b) / 2
        fm = f(m)
        if fm == 0.0:
            return m, m
        elif fm * f(b) < 0.0:
            a = m
        else:
            b = m
    return [a, b], iterations

def find_intervals(f, x_range=(-1, 1), step=1):
    a, b = x_range
    intervals = []
    x = a
    while x < b:
        if f(x) * f(x + step) <= 0:
            intervals.append((x, x + step))
        x += step
    return intervals

f = lambda x: 0.5*x**3 -2 +8*x**2
a = 0.0
b = 1.0
root, iterations = bisection(f, a,b)
print(f"Aufgabe 2a {root, iterations}")

f = lambda x: x**3 - 1.9*x**2 - 5.1*x + 5.3
intervals = find_intervals(f)
for sublist in intervals:
    for index, item in enumerate(sublist):
        if index == 0:
            first_item = item
        elif index == 1:
            second_item = item
    print(bisection(f, first_item, second_item, 1.0e-8))

# 3a
def newton(f, df, x0, tol):
    x,a = sp.symbols('x a')
    iterations = 0
    x_current = x0
    while abs(f.evalf(subs={x: x_current})) > tol:
        x_current = x_current - f.evalf(subs={x: x_current}) / df.evalf(subs={x: x_current})
        iterations += 1
    return x_current, iterations

x,a = sp.symbols('x a')
f = a*x**3-x**2+2*x-a # Funktion
df = sp.diff(f, x)
x0 = 1.0 # Startwert
print(df)
root, iterations = newton(f, df, x0, 1.0e-1)
print(f"Wurzel = {root}, Iterationen = {iterations}")

# 3b
def newton(equations, variables, initial_guess, tol, max_iterations):
    J = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])
    f = sp.Matrix(equations)
    x = sp.Matrix(initial_guess)
    it = 0

    # Ausgabe der Jacobi-Matrix
    print("Jacobi-Matrix:")
    sp.pprint(J)

    # Ausgabe der Inversen Jacobi-Matrix
    J_inv = J.inv()
    print("Inverse der Jacobi-Matrix:")
    sp.pprint(J_inv)

    # Ausgabe der Formel für die Newton-Iteration
    print("Formel für die Newton-Iteration:")
    delta_x_formula = J.inv() * (-f)
    sp.pprint(x + delta_x_formula)

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
max_iterations = 10 # Max. iterationen

equations3 = [-x**3 + 3*y - 2, y*sp.exp(x)-2] # Matrix
initial_guess3 = [1.0, 1.0] # Anfangsversuche
tol3 = 1e-8
result3, iterations3 = newton(equations3, variables, initial_guess3, tol3, max_iterations)
print(f"Ergebnis: {result3}, Iterationen: {iterations3}")

# 4a
x = sp.Symbol('x')
x_values = [0.25, 0.5, 1.0,2.0]  # Punkte x
y_values = [0.3185,0.4564,0.4686,0.2471]  # Punkte y

# Lagrange Interpolationspolynome
def lagrange_basis_polynomials(x_values, x):
    n = len(x_values)
    basis_polynomials = []
    for i in range(n):
        term = 1
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        basis_polynomials.append(term)
    return basis_polynomials

lagrange_polynomials = lagrange_basis_polynomials(x_values, x)
for i, lp in enumerate(lagrange_polynomials):
    print(f"L_{i}(x) = {sp.simplify(lp)}")

# Interpolationspolynom
def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0
    basis_polynomials = lagrange_basis_polynomials(x_values, x)
    for i in range(n):
        result += y_values[i] * basis_polynomials[i]
    return result

# Berechnen des Interpolationspolynoms
p = lagrange_interpolation(x_values, y_values, x)
p_simplified = sp.simplify(p)
coeffs = sp.Poly(p_simplified, x).all_coeffs()

print(f"Interpolationspolynom: {p_simplified}")
print(f"Koeffizientenvektor a: {coeffs}")

# 4c
x = sp.symbols('x')
f1 = 0.308*x**3 - 1.24193333333333*x**2 + 1.3483*x + 0.0542333333333334
a, b = 0.4, 0.8

# Simpson's Regel
simpsons_f1 = (b - a) / 6 * (f1.subs(x, a) + 4 * f1.subs(x, (a + b) / 2) + f1.subs(x, b))
print(simpsons_f1)

# 6a
x, w0, w1, w2 = sp.symbols('x w0 w1 w2')

# Gleichungen aufstellen um Lineares Gleichungssystem zu erhalten
eq1 = sp.Eq(w0 + w1 + w2, 2)
eq2 = sp.Eq(w0 * (-1) + w1 * (-1/2) + w2 * 0, 0)
eq3 = sp.Eq(w0 * (-1) + w1 * (-1/2)**2 + w2 * 0**2, 2/3)
weights = sp.solve((eq1, eq2, eq3), (w0, w1, w2)) # Löst das Gleichungssystem und daraus ergeben sich die Gewichte w0,w1 und w2
print(f'Weights: {weights}')
# Aufgabe 5b
f = x**2 # Integral aus Aufgabe
Q = weights[w0] * f.subs(x, -1) + weights[w1] * f.subs(x, -1/2) + weights[w2] * f.subs(x, 0) # Quadraturformel mit berechneten w Werten
print(f'Q={Q}') # Formel von Q
erf_approx =  Q # Berechnen Wert mit Grenzen von Integral aus Aufgabe
erf_exakt = sp.erf(1)
print(f'Näherungswert: {erf_approx.evalf()}, Exakter Wert: {erf_exakt.evalf()}')
"""

def mittelpunktsregel( a, b, f ):
    return (b-a) * f((a+b)/2)
def trapezregel( a, b, f ):
    return (b-a)/2 * f(a) + (b-a)/2 * f(b)
def simpsonregel( a, b, f ):
    return (b-a)/6*f(a) + 4*(b-a)/6*f((a+b)/2) + (b-a)/6*f(b)
# Gewichte auf Standardintervall [0,1]
A = np.array([[1, 1, 1], [-1/2, -1/10, 1/2], [(-1/2)**2, (-1/10)**2, (1/2)**2]])
b = np.array([2, 0, 2/3])
w = np.linalg.solve(A, b)
print('w=',w)
# Gewichte auf [a,b] = (b-a)*w
print(mittelpunktsregel(A,b,lambda x:x**2))