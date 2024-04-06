import sympy as sp
import numpy as np
from scipy.linalg import lu

# Aufgabe 2 (nicht ganz von Hand)
Z = sp.Matrix([[-1, 1, 1],
               [1, -3, -2],
               [5, 1, 4]])

P9 = Z.LUdecomposition()
L9 = Z.LUdecomposition()
print(f"Aufgabe 3b: P = {P9}, L = {L9}") # U ist noch leer

# Aufgabe 3a
x = sp.Symbol('x')

def gauss_algorithm(matrix):

    rref_matrix, _ = matrix.rref()
    solutions = rref_matrix[:, -1]
    
    return solutions

coeff_matrix = sp.Matrix([[1, 2, -1, 9],
                          [4, -2, 6, -4],
                          [3, 1, 0, 9]])

solutions = gauss_algorithm(coeff_matrix)

print("Aufgabe 3a: ",end="")
for i, sol in enumerate(solutions):
    print(f"x{i+1} = {sol} ", end="")
print("")

# Aufgabe 3b
A = sp.Matrix([[1, 2, -1, 9],
               [4, -2, 6, -4],
               [3, 1, 0, 9]])

P, L, U = A.LUdecomposition()
print(f"Aufgabe 3b: P = {P}, L = {L}, U = {U}") # U ist noch leer
 
# Aufgabe 3c
B = np.array([[1, 2, -1],
               [4, -2, 6],
               [3, 1, 0]])
b = np.array([9,-4,9])
c = np.linalg.solve(B, b)
print(f"Aufgabe 3c, x = {c}")

# Aufgabe 4a
D = sp.Matrix([[-2, 4, 1],
               [4, -11, 1],
               [-8, 19, 6]])

b1 = sp.Matrix([8, -4, 40])
b2 = sp.Matrix([-10, 14, -44])
x1 = D.solve_least_squares(b1)
x2 = D.solve_least_squares(b2)
print(f"Aufgabe 4a x1: {x1}, x2: {x2}")

# Aufgabe 4b
E = sp.Matrix([[-2, 4, 1],
               [4, -11, 1],
               [-8, 19, 6]])
e = sp.Matrix([8,-4,40])
f = sp.Matrix([-10,14,-44])
P2, L2, U2 = E.LUsolve(e)
P1, L1, U1 = E.LUsolve(f)

print(f"Aufgabe 4b x1: {P2}, {L2}, {U2}, x2: {P1}, {L1}, {U1}")

# Aufgabe 5a
n = 22
F = np.vander(np.linspace(0,1,n)) # Erstellt 22 matrizen (vander)
x_exakt = np.ones(n) # 22 [1,1,1,1,1,]
f = F@x_exakt
g = np.linalg.solve(F,f)
print(f"Aufgabe 5a: {np.linalg.norm(g-x_exakt)/np.linalg.norm(x_exakt)*100}")
# Code versucht, relative Genauigkeit der Lösung des linearen GLS zu berechnen.

# Aufgabe 5b
cond_A = format(np.linalg.cond(F))
print(f"Aufgabe 5b: Kondition = {cond_A}")


# Aufgabe 6
K = np.array([[1, 2, -1],
              [4, -2, 6],
              [3, 1, 0]])
i = np.array([9, -4, 9])

k1, k2, k3 = lu(K)
y1 = np.linalg.solve(k1, k2)
x6 = np.linalg.solve(k3, y1)

print(f"Aufgabe 6: x1 = {x6[0]}, x2 = {x6[1]}, x3 = {x6[2]}")
