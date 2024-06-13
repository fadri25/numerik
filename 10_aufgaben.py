import numpy as np
import sympy as sp
from scipy.integrate import quad

# Aufgabe 2
x = sp.symbols('x')
f1 = 4 * x**3
f2 = 5 * x**4
a, b = -1, 1

# Simpson's Regel
simpsons_f1 = (b - a) / 6 * (f1.subs(x, a) + 4 * f1.subs(x, (a + b) / 2) + f1.subs(x, b))
simpsons_f2 = (b - a) / 6 * (f2.subs(x, a) + 4 * f2.subs(x, (a + b) / 2) + f2.subs(x, b))

#  3/8 Regel
three_eighths_f1 = (b - a) / 8 * (f1.subs(x, a) + 3 * f1.subs(x, a + (b - a) / 3) + 3 * f1.subs(x, a + 2 * (b - a) / 3) + f1.subs(x, b))
three_eighths_f2 = (b - a) / 8 * (f2.subs(x, a) + 3 * f2.subs(x, a + (b - a) / 3) + 3 * f2.subs(x, a + 2 * (b - a) / 3) + f2.subs(x, b))

exact_f1 = sp.integrate(f1, (x, a, b))
exact_f2 = sp.integrate(f2, (x, a, b))

print(simpsons_f1,
      simpsons_f2,
      three_eighths_f1,
      three_eighths_f2,
      exact_f1,
      exact_f2)

# Aufgabe 3
f = x * sp.sin(x)
exact_integral = sp.integrate(f, (x, a, b))
def f_numeric(x):
    return x * np.sin(x)

# Mittelpunkt Regel
midpoint = (b - a) * f_numeric((a + b) / 2)

# Trapez Regel
trapezoidal = (b - a) / 2 * (f_numeric(a) + f_numeric(b))

# Simpson's Regel
simpson = (b - a) / 6 * (f_numeric(a) + 4 * f_numeric((a + b) / 2) + f_numeric(b))

# 3/8 Regel
three_eighths = (b - a) / 8 * (f_numeric(a) + 3 * f_numeric(a + (b - a) / 3) + 3 * f_numeric(a + 2 * (b - a) / 3) + f_numeric(b))

exact_integral_numeric, _ = quad(f_numeric, a, b)
error_midpoint = abs(exact_integral_numeric - midpoint)
error_trapezoidal = abs(exact_integral_numeric - trapezoidal)
error_simpson = abs(exact_integral_numeric - simpson)
error_three_eighths = abs(exact_integral_numeric - three_eighths)

print(midpoint,
      trapezoidal,
      simpson,
      three_eighths,
      exact_integral_numeric,
      error_midpoint,
      error_trapezoidal,
      error_simpson,
      error_three_eighths)

# Aufgabe 4
x_data = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
f_data = np.array([0.5, 0.6, 0.8, 1.3, 2, 3.2, 4.8])

# Aufgabe 4a
def trapez_regel(x, y):
    n = len(x)
    integral = 0
    for i in range(n - 1):
        integral += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
    return integral

integral_trapezoidal = trapez_regel(x_data, f_data)
print(integral_trapezoidal)

# Aufgabe 4b
def simpson_regel(x, y):
    n = len(x)
    integral = y[0] + y[-1]
    for i in range(1, n - 1, 2):
        integral += 4 * y[i]
    for i in range(2, n - 1, 2):
        integral += 2 * y[i]
    integral *= (x[2] - x[0]) / 3
    return integral * (x[1] - x[0]) / 3

integral_simpson = simpson_regel(x_data, f_data)
print(integral_simpson)

# Aufgabe 4c
def drei_acht_regel(x, y):
    n = len(x)
    integral = y[0] + y[-1]
    for i in range(1, n - 1):
        if i % 3 == 0:
            integral += 2 * y[i]
        else:
            integral += 3 * y[i]
    integral *= 3 * (x[1] - x[0]) / 8
    return integral

integral_three_eighths = drei_acht_regel(x_data, f_data)
print(integral_three_eighths)

# Aufgabe 5a
x, w0, w1, w2 = sp.symbols('x w0 w1 w2')

# Allgemeine Formel: Q = w0f(x0) + w1f(x1) + w2f(x2) -> w sind Gewichte und x Punkte, an denen die Funktion ausgewertet wird
"""
1. Gleichung: (Polynom 1) Q = w0f(x0) + w1f(x1) + w2f(x2) // Ergibt 2 (integral 2/0(1)dx = 2)
2. Gleichung: (Polynom x) Q = w0 * 0 + w1 * 3/4 + w2 * 2 = 2 // Ergibt 2 (integral 2/0(x)dx = 2)
3. Gleichung: (Polynom x^2) Q = w0 * 0 + w1 * (3/4)^2 + w2 * 2^2 = 2 // Ergibt 8/3 (integral 2/0(x^2)dx = 8/3)
"""
# Gleichungen aufstellen um Lineares Gleichungssystem zu erhalten
eq1 = sp.Eq(w0 + w1 + w2, 2)
eq2 = sp.Eq(w1 * (3/4) + w2 * 2, 2)
eq3 = sp.Eq(w1 * (3/4)**2 + w2 * 2**2, 8/3)
weights = sp.solve((eq1, eq2, eq3), (w0, w1, w2)) # Löst das Gleichungssystem und daraus ergeben sich die Gewichte w0,w1 und w2
print(f'Weights: {weights}')
# Aufgabe 5b
f = sp.exp(-x**2) # Integral aus Aufgabe
Q = weights[w0] * f.subs(x, 0) + weights[w1] * f.subs(x, 3/4) + weights[w2] * f.subs(x, 2) # Quadraturformel mit berechneten w Werten
print(f'Q={Q}') # Formel von Q
erf_approx = (2 / sp.sqrt(sp.pi)) * Q # Berechnen Wert mit Grenzen von Integral aus Aufgabe
erf_exakt = sp.erf(2)
print(f'Näherungswert: {erf_approx.evalf()}, Exakter Wert: {erf_exakt.evalf()}')


# Aufgabe 6
def f(x):
    return 4 / (1 + x**2)
a, b = 0, 1
n = 100
h = (b - a) / n

# Mittelpunktregel
midpoint_sum = sum(f(a + (i + 0.5) * h) for i in range(n))
midpoint_result = midpoint_sum * h

# Trapezregel
trapezoidal_sum = 0.5 * f(a) + sum(f(a + i * h) for i in range(1, n)) + 0.5 * f(b)
trapezoidal_result = trapezoidal_sum * h

# Simpsonregel
simpson_sum = f(a) + f(b)
simpson_sum += 4 * sum(f(a + (i + 0.5) * h) for i in range(n))
simpson_sum += 2 * sum(f(a + i * h) for i in range(1, n))
simpson_result = simpson_sum * h / 3

print(midpoint_result, 
      trapezoidal_result,
      simpson_result)


