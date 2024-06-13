import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Daten
xdat = np.array([-2, -1, 3, 4, 6])
ydat = np.array([0, 0.5, 2, 2, 5])

# y = m*x + q
A = np.vstack([xdat, np.ones(len(xdat))]).T
p, residuals, rank, s = np.linalg.lstsq(A, ydat, rcond=None)
m, q = p

print("m =", m, "q =", q)

# Fehlerquadratsumme berechnen
fehlerquadratsumme = np.sum((A @ p - ydat)**2)
print("Fehlerquadratsumme =", fehlerquadratsumme)

# Überprüfung mit f(2)
x_test = 2
y_test = q * x_test + m
print("f(2) =", y_test)

# Plotten der Regressionsgeraden und der Datenpunkte
x = np.linspace(xdat.min(), xdat.max(), 100)
plt.figure()
plt.plot(x, m*x + q, label='Regressionsgerade')
plt.plot(xdat, ydat, 'o', label='Datenpunkte')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
