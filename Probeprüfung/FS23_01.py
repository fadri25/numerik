import sympy as sp
import numpy as np
import math
from scipy.integrate import solve_ivp

# 1a
# Erste n Ableitungen berechnen:
x = sp.Symbol('x')
fx = sp.exp(-x/2) * sp.cos(sp.sqrt(3*x)/2) # Formel anpassen
ffx = sp.diff(fx, x)
fffx = sp.diff(ffx,x)
ffffx = sp.diff(fffx,x) # Das wäre die dritte usw....

print(f'1a Erste Ableitung: {ffx}')
print(f"1a Zweite Ableitung: {fffx}")

# 1b
def taylor_coefficients(x0, fk_list):
    derivatives = np.array([fk(x0) for fk in fk_list])
    n = len(fk_list) - 1
    factorials = np.array([math.factorial(k) for k in range(n + 1)])
    return derivatives / factorials

def taylor_factory(x0, fk_list):
    derivatives = [fk for fk in fk_list]
    factorials = np.array([math.factorial(n) for n in range(len(fk_list))])
    coefficients = derivatives / factorials
    return lambda x: np.sum([ck * (x - x0)**k for k, ck in enumerate(coefficients)], axis=0)

f0 = lambda x: np.exp(-x/2) * np.cos((np.sqrt(3)*x)/2)
f1 = lambda x: -(np.exp(-x/2)/2)*(np.sqrt(3)*np.sin((np.sqrt(3)*x)/2)+np.cos((np.sqrt(3)*x)/2))
f2 = lambda x: (np.exp(-x/2)/2)*(np.sqrt(3)*np.sin((np.sqrt(3)*x)/2)-np.cos((np.sqrt(3)*x)/2))

x0 = 0.0
fk_list = [f0, f1, f2]
c = taylor_coefficients(x0, fk_list)
print("Taylor-Koeffizienten: ", c)

# 1c
fk_list1 = [f0(x0), f1(x0), f2(x0)]
d = taylor_factory(x0, fk_list1)
x1 = 0.5
x2 = 1.6
print(d(x1)-f0(x1))
print(d(x2)-f0(x2))

# 2a
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
max_iterations = 10 # Max. iterationen

equations3 = [x**2 - (sp.sin(x)/4), -sp.sin(y)] # Matrix
initial_guess3 = [2.0, -2.0] # Anfangsversuche
tol3 = 10e-6
result3, iterations3 = newton(equations3, variables, initial_guess3, tol3, max_iterations)
print(f"Ergebnis: {result3}, Iterationen: {iterations3}")

# 2b
def Newt2D():
    tol = 0.0001
    x = np.array([2., -2.])
    n = 0
    while np.linalg.norm(fun(x)) > tol:
        d = np.linalg.solve(jac(x),fun(x))
        x -= d
        n += 1
        print(x, n)
        
def fun(x):
    y = np.array([x[0]**2-np.sin(x[0])/4. - x[1] - 1.,
                    -np.sin(x[1]) - x[0] - 1])
    return y

def jac(x):
    J = np.array([ [2*x[0]-np.cos(x[0])/4., -1],
                [-1., -np.cos(x[1])] ])
    return J


# 3a
A = sp.Matrix([[1, -1, 1],
               [3, -1, -2],
               [-1, 1, 3]])
P, L, U = A.LUdecomposition()
print(f"Aufgabe 3b: A = {P}, b = {L}")

# 3b
B = np.array([[1, 0, 0],
               [3, 1, 0],
               [-1, 0, 1]])
b = np.array([1,10,-2])
c = np.linalg.solve(B, b)
print(f"Aufgabe 3c, x = {c}")

# 3c
B = np.array([[1, -1, 1],
               [3, -1, -2],
               [-1, 1, 3]])
b = np.array([1,10,-2])
c = np.linalg.solve(B, b)
print(f"Aufgabe 3c, x = {c}")

# 3d
C = np.array([[10^-25, -1, 1],
               [3, -1, -10^4],
               [-1, 1, 3]])
condition_number = np.linalg.cond(C)
cond = np.linalg.cond(B)
print("Konditionszahl der Matrix A:", condition_number, cond)

# 4a
x = np.array([0, np.pi/6, np.pi/2, 5*np.pi/6, np.pi, 2*np.pi])
y = np.array([1, 2, 3, 2, 0, 0])
A = np.vstack([np.ones(len(x)), np.sin(x)]).T
b = y
ATA = A.T @ A
ATb = A.T @ b
print(A)
print("Matrix A^T * A:")
print(ATA)
print("\nRechte Seite A^T * b:")
print(ATb)

# 4b
theta = np.linalg.solve(ATA, ATb)
a, b = theta
print("\nLösungen für a und b:")
print(f"a = {a}, b = {b}")

# 4c
B = np.vstack([np.ones(len(x)), x**2]).T
c = y
ATB = B.T @ B
ATc = B.T @ c
print("Matrix A^T * A:")
print(ATB)
print("\nRechte Seite A^T * b:")
print(ATc)

# 4d
e,f = np.linalg.solve(ATB, ATc)
print(e,f)

# 4e
p = np.linalg.solve(ATA, ATb)
q = np.linalg.solve(ATB, ATc)
print("Fehlerquadratsumme=", np.linalg.norm(A@p-y))
print("Fehlerquadratsumme=", np.linalg.norm(B@q-y))

# 5a
x = 28
y_values = [11, 13, 15, 17]  # Punkte x
x_values = [17, 26, 31, 34]  # Punkte y

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
print(p)

# 6a
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
