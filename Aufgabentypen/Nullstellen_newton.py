import sympy as sp

#1D ######################################################
def newton(f, df, x0, tol):
    x = sp.Symbol('x')
    iterations = 0
    x_current = x0
    while abs(f.evalf(subs={x: x_current})) > tol:
        x_current = x_current - f.evalf(subs={x: x_current}) / df.evalf(subs={x: x_current})
        iterations += 1
    return x_current, iterations

x = sp.Symbol('x')
f = sp.exp(-x-1) - sp.log(x**2 + 1) # Funktion
df = sp.diff(f, x)
x0 = 0.0 # Startwert

root, iterations = newton(f, df, x0, 1.0e-10)
print(f"Wurzel = {root}, Iterationen = {iterations}")


#2D ######################################################

def newton(equations, variables, initial_guess, tol, max_iterations):
    J = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])
    f = sp.Matrix(equations)
    x = sp.Matrix(initial_guess)
    it = 0

    # Ausgabe der Jacobi-Matrix
    print("Jacobi-Matrix:")
    sp.pprint(J)

    # Ausgabe der Inversen Jacobi-Matrix
    J_inv = J.inv()
    print("Inverse der Jacobi-Matrix:")
    sp.pprint(J_inv)

    # Ausgabe der Formel für die Newton-Iteration
    print("Formel für die Newton-Iteration:")
    delta_x_formula = J.inv() * (-f)
    sp.pprint(x + delta_x_formula)

    for _ in range(max_iterations):
        f_val = f.subs(dict(zip(variables, x)))
        if f_val.norm() < tol:
            break
        J_val = J.subs(dict(zip(variables, x)))
        delta_x = J_val.LUsolve(-f_val)
        x += delta_x
        it += 1
    res_r = [float(val) for val in x]

    return res_r, it

x, y = sp.symbols('x y')
variables = [x, y]
max_iterations = 10 # Max. iterationen

equations3 = [x**2 + y**2 -1, x-y] # Matrix
initial_guess3 = [1.0, 1.0] # Anfangsversuche
tol3 = 1e-8
result3, iterations3 = newton(equations3, variables, initial_guess3, tol3, max_iterations)
print(f"Ergebnis: {result3}, Iterationen: {iterations3}")
