import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 2
xdat = np.array([ -1, 0, 1, 2, 3 ])
ydat = np.array([ 1, 3.5, 6, 8.5, 10])

# y = m*x + q
A = np.array([xdat, np.ones(xdat.shape)]).T
p = np.linalg.solve(A.T@A, A.T@ydat)
print("m=",p[0], "q=", p[1])

x = np.linspace(xdat.min(), xdat.max())
plt.figure()
plt.plot(x, p[0]*x+p[1])
plt.plot(xdat, ydat, 'o')

print("Fehlerquadratsumme=", np.linalg.norm(A@p-ydat))

