import sympy as sp

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