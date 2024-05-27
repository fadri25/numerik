import numpy as np
import sympy as sp
import math

# Erste n Ableitungen berechnen:
x = sp.Symbol('x')
fx = 1/3 * x**3 + 1/2*x**2 -x -1 # Formel anpassen
ffx = sp.diff(fx, x)
fffx = sp.diff(ffx,x)
ffffx = sp.diff(fffx,x) # Das wäre die dritte usw....

print(f'Erste Ableitung: {ffx}')
print(f"Zweite Ableitung: {fffx}")


# Taylor-Polynom berechnen
x = sp.Symbol('x')
x0 = 2.0  # Stelle x0

f = sp.cos((1/3 * x)**3 + (1/2 * x)**2 - x - 1)  # Funktion
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
print("Taylor-Polynom:", t2)

# Fehler an Stelle x
x1 = 8/5  # Fehlerstelle
f0_x1 = f.subs(x, x1)
t2_x1 = t2.subs(x, x1)
print("Originalfunktion an x1:", f0_x1)
print("Taylor-Polynom an x1:", t2_x1)
app = t2_x1 - f0_x1
print("Approximation:", app)



