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

"""
import numpy as np
import matplotlib.pyplot as plt

n = 30
x = np.linspace(-2.5, 2.5, n)
noise = np.random.rand(n) - 0.5
y = 2.0 * x + 3.0 + noise

A = np.column_stack((np.ones(n), x))
AT = np.transpose(A)
b, m = np.linalg.solve(AT @ A, np.dot(AT, y))
print(m, b)

p = np.array([b, m])
r = y - np.dot(A, p)  # Residuum
print('Fehlerquadratsumme:', np.linalg.norm(r)**2)

plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x, m * x + b, 'r-')
plt.show()
"""