import sympy as sp

x = sp.Symbol('x')
x_values = [-sp.pi/2, 0, sp.pi/2]  # Punkte x
y_values = [0, 2, 0]  # Punkte y

# Lagrange Interpolationspolynome
def lagrange_basis_polynomials(x_values, x):
    n = len(x_values)
    basis_polynomials = []
    for i in range(n):
        term = 1
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        basis_polynomials.append(term)
    return basis_polynomials

lagrange_polynomials = lagrange_basis_polynomials(x_values, x)
for i, lp in enumerate(lagrange_polynomials):
    print(f"L_{i}(x) = {sp.simplify(lp)}")

# Interpolationspolynom
def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0
    basis_polynomials = lagrange_basis_polynomials(x_values, x)
    for i in range(n):
        result += y_values[i] * basis_polynomials[i]
    return result

# Berechnen des Interpolationspolynoms
p = lagrange_interpolation(x_values, y_values, x)
p_simplified = sp.simplify(p)
coeffs = sp.Poly(p_simplified, x).all_coeffs()

print(f"Interpolationspolynom: {p_simplified}")
print(f"Koeffizientenvektor a: {coeffs}")

#Berechnen von p(1)
p_1 = p_simplified.subs(x, 1)
print(f"p(1) = {p_1}")
