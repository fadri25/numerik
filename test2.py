import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Aufgabe 1a
"""
x = sp.symbols('x')
i = [0, 1, 2]
xi = [-1, 1, 3]
yi = [-1, 0, 2]

l0 = ((x - xi[1]) * (x - xi[2])) / ((xi[0] - xi[1]) * (xi[0] - xi[2]))
l1 = ((x - xi[0]) * (x - xi[2])) / ((xi[1] - xi[0]) * (xi[1] - xi[2]))
l2 = ((x - xi[0]) * (x - xi[1])) / ((xi[2] - xi[0]) * (xi[2] - xi[1]))

# Aufgabe 1b
p = yi[0] * l0 + yi[1] * l1 + yi[2] * l2

# Aufgabe 1c
p_value_at_1_5 = p.subs(x, 1.5)

print(l0)
print(l1)
print(l2)
print(p)
print(p_value_at_1_5)
"""
# Aufgabe 2
"""
x = sp.symbols('x')

p1 = x + 1
p2 = 2 * x - 1
p3 = 3 * x**2
p4 = x**2 - x
p5 = x**3 + 1
p6 = x**4
p7 = x**4 - 2 * x
p8 = x**4 - 2 * x**2 + 4

points1 = [(-4, 48), (-2, 12), (2, 12), (4, 48)]
points2 = [(1, 1), (2, 3), (4, 7), (8, 15)]
points3 = [(-1, 0), (0, 1), (1, 2), (2, 9)]
points4 = [(1, 0), (2, 2), (4, 12), (8, 56)]

def matches_points(poly, points):
    return all(sp.expand(poly.subs(x, px)) == py for px, py in points)

matching_polynomials = {}

for poly in [p1, p2, p3, p4, p5, p6, p7, p8]:
    if matches_points(poly, points1):
        matching_polynomials["points1"] = poly
    if matches_points(poly, points2):
        matching_polynomials["points2"] = poly
    if matches_points(poly, points3):
        matching_polynomials["points3"] = poly
    if matches_points(poly, points4):
        matching_polynomials["points4"] = poly

print(matching_polynomials)
"""

# Aufgabe 3a
"""
x_data = np.array([-1, 1, 3, 4])
y_data = np.array([-1, 0, 2, 5])
n = len(x_data)

A = np.vstack((np.ones(n), x_data)).T
B = y_data
A_T_A = A.T @ A
A_T_B = A.T @ B
A_T_A_sym = sp.Matrix(A_T_A)
A_T_B_sym = sp.Matrix(A_T_B)

print(A_T_A_sym)
print(A_T_B_sym)

# Aufgabe 3b
coeffs_linear = sp.Matrix(A_T_A).LUsolve(sp.Matrix(A_T_B))
print(coeffs_linear)

# Aufgabe 3c
A_exp = np.vstack((np.ones(n), np.exp(x_data))).T
A_T_A_exp = A_exp.T @ A_exp
A_T_B_exp = A_exp.T @ B
coeffs_exp = sp.Matrix(A_T_A_exp).LUsolve(sp.Matrix(A_T_B_exp))
print(coeffs_exp)
"""

# Aufgabe 4a
"""
w0, w1, w2 = sp.symbols('w0 w1 w2')

eq1 = sp.Eq(w0 + w1 + w2, 2)
eq2 = sp.Eq(w0 * (-3/4) + w1 * (1/4) + w2 * (3/4), 0) 
eq3 = sp.Eq(w0 * (3/4)**2 + w1 * (1/4)**2 + w2 * (3/4)**2, 2/3)

solution = sp.solve((eq1, eq2, eq3), (w0, w1, w2))
print(solution)
"""

# Aufgabe 5
"""
def odes(t, z):
    z1, z2, z3, z4 = z
    dz1_dt = z2
    dz2_dt = z1 + 2*z4 - z1 / (z1**2 + z3**2)**(3/2)
    dz3_dt = z4
    dz4_dt = z3 - 2*z2 - z3 / (z1**2 + z3**2)**(3/2)
    return [dz1_dt, dz2_dt, dz3_dt, dz4_dt]
z0 = [1, 3, 2, 4]
sol = solve_ivp(odes, [0, 10], z0, t_eval=[10])
u_10 = sol.y[0, -1]
v_10 = sol.y[2, -1]
print(u_10)
print(v_10)
"""

# Aufgabe 6a
"""
R = np.array([[0, 0, 0, 9], 
              [0, 0, 7, 8], 
              [0, 4, 5, 6], 
              [3, 2, 1, 0]])

det_R = np.linalg.det(R)
print(det_R)

# Aufgabe 6b
def einsetzen(R, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n, 0, -1):  # n -> 1 rückwärts
        sum = 0
        for j in range(i, n):  #i -> n-1 vorwärts
            sum += R[n-i, j] * x[j]
        x[i-1] = (b[n-i] - sum) / R[n-i, i-1]
    return x

# Aufgabe 6c
b = np.array([1.0, 2.0, 3.0, 4.0])
x = einsetzen(R, b)
x_rounded = np.round(x, 5)
print(x_rounded)
"""





