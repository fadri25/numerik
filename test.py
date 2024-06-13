import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
"""

# Differential
x = sp.Symbol('x')
fx = 1/3 * x**3 + 1/2*x**2 -x -1
ffx = sp.diff(fx, x)
fffx = sp.diff(ffx,x)

print(ffx)
print(fffx)

# Aufgabe 3a
x0 = 2.0
x1 = -0.6
def f0(x):
    return 1/3 * x**3 + 1/2*x**2 -x -1
def f1(x):
    return x**2+x-1
def f2(x):
    return 2*x+1

fk_list = [f0(x0), f1(x0), f2(x0)]

def taylor_factory(x0, fk_list):
    derivatives = [fk for fk in fk_list]
    factorials = np.array([math.factorial(n) for n in range(len(fk_list))])
    coefficients = derivatives / factorials
    return lambda x: np.sum([ck * (x - x0)**k for k, ck in enumerate(coefficients)], axis=0)

t3 = taylor_factory(x0, fk_list)
print(t3(x0))

# Jacobi Matrix
x, y = sp.symbols('x y')
f1= x**2+y**2-1
f2=x - y
F = sp.Matrix([f1,f2])
print(F.jacobian([x,y]))


# Newton Verfahren mehrdimensional
def newton(equations, variables, initial_guess, tol, max_iterations):
    J = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])
    f = sp.Matrix(equations)
    x = sp.Matrix(initial_guess)
    it = 0

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
max_iterations = 10

equations3 = [-2*x**3+3*y**2 + 42, 5*x**2 + 3*y**3 - 69]
initial_guess3 = [1.0, 1.0]
tol3 = 1e-8
result3, iterations3 = newton(equations3, variables, initial_guess3, tol3, max_iterations)
print(f"Aufgabe 3: Ergebnis: {result3}, Iterationsn: {iterations3}")

# Interpolation
x = sp.Symbol('x')
x_val1 = [-sp.pi/2, 0, sp.pi/2]
y1 = [0, 2, 0]
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

# Lineare Interpolation
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
print(f"Lagrange Interpolation", x_val1, y1, lagrange_interpolation(x_val1, y1,x))
print(f"Linear System Interpolation", x_val1, y1, linear_system_interpolation(x_val1,y1))

# Regression
xdat = [-sp.pi/2, 0, sp.pi/2, sp.pi]
ydat = [0, 2, 0, -1]
x = sp.Symbol('x')
m, q = sp.symbols('m q')

model = m * x + q

X = sp.Matrix([[xi, 1] for xi in xdat])
Y = sp.Matrix(ydat)
params = sp.Matrix([m, q])

normal_eq = X.T * X
right_hand_side = X.T * Y

solution = normal_eq.inv() * right_hand_side

m_val, q_val = solution[0], solution[1]

print("m =", m_val)
print("q =", q_val)

A = sp.Matrix([xdat, [1]*len(xdat)]).T
p = sp.Matrix([m_val, q_val])
error = sp.Matrix(ydat) - A * p
error_sum_of_squares = (error.T * error)[0]

print("Fehlerquadratsumme =", error_sum_of_squares)

# Aufgabe 2
xdat = np.array([-np.pi/2, 0, np.pi/2, np.pi])
ydat = np.array([ 0, 2, 0, -1 ])
A = np.array([xdat, np.ones(xdat.shape)]).T
p = np.linalg.solve(A.T@A, A.T@ydat)
print("m=",p[0], "q=", p[1])
print("Fehlerquadratsumme=", np.linalg.norm(A@p-ydat))

x = np.linspace(xdat.min(), xdat.max())
plt.figure()
plt.plot(x, p[0]*x+p[1])
plt.plot(xdat, ydat, 'o')
plt.show()

"""
 
x = np.array([0, 3/4, 2])

A = np.array([[1, 1, 1],
            [0, 3/4, 2],
            [0, (3/4)**2, 2**2]]) #mein versuch mit dem hoch 2
 
#Integrale ausrechnen für f(x) = 1, f(x) = x, f(x) = x**2 von a = 0 bis b = 2
#stimmt aber nicht weil es nicht Polynom ist
x = sp.Symbol('x')
f = x**2; a = 0; b = 2
definite_integral = sp.integrate(f, (x, a, b))
 
b = np.array([2, 2, 8/3])
w = np.linalg.solve(A, b)
print("Gewichte: ", w) #stimmen nicht
