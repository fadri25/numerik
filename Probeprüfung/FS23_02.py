import sympy as sp
import numpy as np
import math
from scipy.integrate import solve_ivp

# 1a
x = sp.Symbol('x')
fx = sp.exp(-x/2)*sp.cos((sp.sqrt(3)*x)/2) # Formel anpassen
ffx = sp.diff(fx, x)
fffx = sp.diff(ffx,x)
sp.pprint(ffx)
sp.pprint(fffx)

# 1b
x = sp.Symbol('x')
x0 = 0.0  # Stelle x0

f = sp.exp(-x/2)*sp.cos((sp.sqrt(3)*x)/2)  # Funktion
f0 = f
f1 = sp.diff(f, x)
f2 = sp.diff(f1, x)
fk_list = [f0.subs(x, x0), f1.subs(x, x0), f2.subs(x, x0)]

def taylor_factory(x0, fk_list):
    n = len(fk_list)
    x = sp.Symbol('x')
    taylor_polynom = sum(fk_list[k] * (x - x0)**k / sp.factorial(k) for k in range(n))
    return taylor_polynom

# Taylor-Polynom zweiter Ordnung
t2 = taylor_factory(x0, fk_list)
sp.pprint(t2)

# 1c
x1 = 0.5
x2 = 1.6
print(abs(t2.subs(x, x1) - f.subs(x, x1)).evalf())
print(abs(t2.subs(x,x2) - f.subs(x,x2)).evalf())

# 2a
def newton(equations, variables, initial_guess, tol, max_iterations):
    J = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])
    f = sp.Matrix(equations)
    x = sp.Matrix(initial_guess)
    it = 0

    # Ausgabe der Jacobi-Matrix
    #print("Jacobi-Matrix:")
    sp.pprint(J)

    # Ausgabe der Inversen Jacobi-Matrix
    J_inv = J.inv()
    #print("Inverse der Jacobi-Matrix:")
    #sp.pprint(J_inv)

    # Ausgabe der Formel für die Newton-Iteration
    #print("Formel für die Newton-Iteration:")
    delta_x_formula = J.inv() * (-f)
    #sp.pprint(x + delta_x_formula)

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

equations3 = [x**2-(sp.sin(x)/4), -sp.sin(y)] # Matrix
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
    y = np.array([x[0]**2-np.sin(x[0])/4. -x[1]-1.,-np.sin(x[1])-x[0]-1])
    return y

def jac(x):
    J = np.array([[2*x[0]-np.cos(x[0])/4,-1],[-1,-np.cos(x[1])]])
    return J

# 2c
print(Newt2D())

# 3a
A = sp.Matrix([[1, -1, 1],
               [3, -1, -2],
               [-1, 1, 3]])
U, L, P = A.LUdecomposition()
print(f"Aufgabe 3b: A = {U}, b = {L}")

# 3b
B = np.array([[1, 0, 0],
               [3, 1, 0],
               [-1, 0, 1]])
b = np.array([1,10,-2])
c = np.linalg.solve(B, b)
print(f"Aufgabe 3b, x = {c}")

# 3c
C = np.array([[1, -1, 1],
               [0, 2, -5],
               [0, 0, 4]])
d = np.array([1,7,-1])
c = np.linalg.solve(C, d)
print(f"Aufgabe 3c, x = {c}")


# 3d
B_cond = np.array([[1, -1, 1],
                        [3, -1, -2],
                        [-1, 1, 3]])
A_cond = np.array([[10^25, -1, 1],
                    [3, -1, -10^4],
                    [-1, 1, 3]])
print(np.linalg.cond(B_cond))
print(np.linalg.cond(A_cond))

# 4a
x = np.array([0, 1/6*np.pi, 1/2*np.pi,5/6*np.pi,np.pi,2*np.pi])
y = np.array([1, 2,3,2,0,0])
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
e, f = theta
print("\nLösungen für a und b:")
print(f"a = {e}, b = {f}")

# 4c
B = np.vstack([np.ones(len(x)), x**2]).T
BTB = B.T @ B
BTb = B.T @ b
print(B)
print("Matrix A^T * A:")
print(BTB)
print("\nRechte Seite A^T * b:")
print(BTb)

# 4d
theta = np.linalg.solve(BTB, BTb)
c, d = theta
print("\nLösungen für a und b:")
print(f"a = {c}, b = {d}")

# 4e
p = np.linalg.solve(A.T@A, A.T@y)
q = np.linalg.solve(B.T@B, B.T@y)
print("Fehlerquadratsumme=", np.linalg.norm(A@p-y))
print("Fehlerquadratsumme=", np.linalg.norm(B@q-y))

# 5a
x = 28
y_values = [11,13,15,17]  # Punkte x
x_values = [17, 26, 31,34]  # Punkte y

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
    
# 5b
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


# 6a/b
def explicit_euler(f, t0, y0, h, n):
    t = np.empty(n + 1)
    t[0] = t0
    y = np.empty((n + 1, len(y0)))  # geändert
    y[0] = y0
    for k in range(n):
        t[k + 1] = t[k] + h
        y[k + 1] = y[k] + h * f(t[k], y[k])
    return t, y

M = np.array([[0.0, 1.0], [-2.0, 1.0]]) # z1'(t) und z2'(t)
f = lambda t, y: np.dot(M, y)
t0 = 0.0
y0 = np.array([1.0, 0.0]) # z1(0) und z2(0)
h = 0.1 # Zeitschrit
n = 2 # Anzahl Schritte
t, y = explicit_euler(f, t0, y0, h, n)
print(t,y)

# 6c
t_span = (0, 3)
sol_numeric = solve_ivp(f, t_span, y0, method='RK45')
print(sol_numeric)
