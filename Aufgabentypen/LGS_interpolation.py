import sympy as sp
import numpy as np
from scipy.optimize import minimize
"""
# LGS dass Vektor durch die Punkte in der Tabelle geht
a0, a1, a2 = sp.symbols('a0 a1 a2')
x = sp.Symbol('x')

f = a0 + a1 * sp.cos(x) + a2 * sp.sin(x) # Funktion
x_values = [-sp.pi/2, 0, sp.pi/2] # X-Werte
y_values = [0, 2, 0] # Y-Werte

equations = [] # Gleichungssystem aufstellen
for xi, yi in zip(x_values, y_values):
    equations.append(f.subs(x, xi) - yi)

solutions = sp.solve(equations, (a0, a1, a2)) # Lösen des Gleichungssystems

print(f"Lösungen für a0, a1, a2: {solutions}")
"""
# Fehlerquadratsumme minimieren


