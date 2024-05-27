import sympy as sp

x = sp.Symbol('x')
x_val1 = [-sp.pi/2, 0, sp.pi/2] # Punkte x
y1 = [0, 2, 0] # Punkte y

# Lagrange Interpolation
def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

poly = lagrange_interpolation(x_val1, y1, x)
poly = sp.simplify(poly)
coeffs = sp.Poly(poly, x).all_coeffs()

print(f"Interpolationspolynom: {poly}")
print(f"Koeffizientenvektor a: {coeffs}")
