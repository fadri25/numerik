import math
import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 2
#a) 1000
#b) 1010

# Aufgabe 3
result_a = 0.1 + 0.2 + 0.3 == 0.6
print("a):", result_a)
result_b = 0.1 + 0.2 + 0.3 == 0.3 + 0.2 + 0.1
print("b):", result_b)
result_c = 0.1 * (0.3 + 0.1) == 0.1 * 0.3 + 0.1 ** 2
print("c):", result_c)

# Aufgabe 4
# a)
f = lambda x: 1 - np.cos(x)

# b)
x_values = np.linspace(-0.5e-7, 0.5e-7, 400)
y_values = f(x_values)

plt.plot(x_values, y_values)
plt.title('Plot von f(x) = 1 - cos(x) um x = 0')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
#plt.show()

# c)
# Beobachtungen folgen

# Aufgabe 5
x = 2.346
y = 2.344

# a)
exact_res = round(x - y,3)

# b)
x_c = round(x, 2)
y_c = round(y,2)
f_res = round(x_c - y_c, 3)

# c) Berechnung des relativen Fehlers
relative_err = abs(exact_res - f_res) / exact_res
# Ausgabe der Ergebnisse
print("a) ", exact_res)
print("b) ", f_res)
print("c) ", relative_err)

# Aufgabe 6
p = lambda x: x**6 - 12*x**5 + 60*x**4 - 160*x**3 + 240*x**2 - 192*x + 64

x_values = np.linspace(1.99, 2.01, 400)
y_values = p(x_values)

plt.plot(x_values, y_values)
plt.title('Plot von p(x) = x^6 - 12x^5 + 60x^4 - 160x^3 + 240x^2 - 192x + 64')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.grid(True)
plt.show()









