import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

x = sp.Symbol('x')
# Aufgabe 2
x_val1 = [-1, 0, 1, 2]
y1 = [-11, -2, 3, 22]
x_val2 = [-1, 0, 2, 3]
y2 = [-1, -3, 1, -27]
# Lagrange Interpolation
def lagrange_interpolation(x_values, y_values,x):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

# Lösen eines linearen Gleichungssystems
def linear_system_interpolation(x_values, y_values):
    n = len(x_values)
    A = np.zeros((n, n))
    B = np.array(y_values)
    for i in range(n):
        for j in range(n):
            A[i][j] = x_values[i] ** (n - 1 - j)
    coefficients = np.linalg.solve(A, B)
    polynomial = sum(coefficients[i] * x ** (n - 1 - i) for i in range(n))
    return polynomial

# Grafische Darstellung
def plot_interpolation(x_values, y_values, polynomial, title):
    x_interp = np.linspace(min(x_values), max(x_values), 100)
    y_interp = [polynomial.subs('x', x_val) for x_val in x_interp]
    plt.plot(x_interp, y_interp, label=title)

print(f"Lagrange Interpolation", x_val1, y1, lagrange_interpolation(x_val1, y1,x))
print(f"Linear System Interpolation", x_val2, y2, linear_system_interpolation(x_val2,y2))
"""
for title, x_values, y_values, interpolation_method in [("Lagrange Interpolation", x_val1, y1, lagrange_interpolation), 
                                                        ("Linear System Interpolation", x_val2, y2, linear_system_interpolation)]:
    plt.figure()
    plt.scatter(x_values, y_values, label='Data', marker='o')
    polynomial = interpolation_method(x_values, y_values)
    plot_interpolation(x_values, y_values, polynomial, title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    #plt.show()
"""
    
# Aufgabe 3
x_val3 = [-2, 0, 1, 2, 4]
y3 = [-31, -5, None, 13, 119]

missing_index = y3.index(None)
interpolating_polynomial = lagrange_interpolation(x_val3[:missing_index] + x_val3[missing_index + 1:], 
                                                  y3[:missing_index] + y3[missing_index + 1:],x)
missing_y_value = interpolating_polynomial.subs(x, x_val3[missing_index])

print("Aufgabe 3:", missing_y_value)

# Aufgabe 4a
x_val4 = [-1, 0, 2, 3]
y4 = [1, -3, 1, -27]

polynomial_linear_system = linear_system_interpolation(x_val4, y4)
polynomial_lagrange = lagrange_interpolation(x_val4, y4,x)
print("Aufgabe 4a1:", polynomial_linear_system)
print("Aufgabe 4a2:", polynomial_lagrange)

# Aufgabe 4b
spline = CubicSpline(x_val4, y4, bc_type='natural')

# Aufgabe 5
windgeschwindigkeit = [23, 35, 48, 61]
leistung = [320, 490, 540, 500]
leistung_bei_40_km_h = lagrange_interpolation(windgeschwindigkeit, leistung, 40)
print("Die Leistung bei 40 km/h beträgt:", leistung_bei_40_km_h)

# Aufgabe 6
jahre = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010])
anteile = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7])

# a
jahre_verschoben = jahre - 1981
koeffizienten = np.polyfit(jahre_verschoben, anteile, deg=len(jahre)-1)
interpolationspolynom_a = np.poly1d(koeffizienten)
anteil_1983_a = interpolationspolynom_a(2) 

# b
spline_interpolation = interp1d(jahre, anteile, kind='cubic')
anteil_1983_b = spline_interpolation(1983)

# c
plt.figure(figsize=(10, 6))
plt.scatter(jahre, anteile, color='red', label='Daten')
x_values_a = np.linspace(min(jahre_verschoben), max(jahre_verschoben), 100)
plt.plot(x_values_a + 1981, interpolationspolynom_a(x_values_a), label='Interpolationspolynom (numpy.polyfit)')
x_values_b = np.linspace(min(jahre), max(jahre), 100)
plt.plot(x_values_b, spline_interpolation(x_values_b), label='Spline-Interpolation (scipy)')
plt.xlabel('Jahr')
plt.ylabel('Anteil der Haushalte (%)')
plt.title('Entwicklung des Anteils an Haushalten mit mindestens einem Computer')
plt.legend()
plt.grid(True)
plt.show()
print("Anteil der Haushalte im Jahr 1983 (numpy.polyfit):", anteil_1983_a)
print("Anteil der Haushalte im Jahr 1983 (Spline-Interpolation):", anteil_1983_b)


