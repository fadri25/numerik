import math
import numpy as np
import matplotlib.pyplot as plt
"""
# Aufgabe 2a
x0 = 1.0
x1 = 0.8
f = lambda x: x * np.sqrt(1+x)
df = lambda x: np.sqrt(1+x) + x/(2*np.sqrt(1 + x))
t1 = lambda x: f(x0) + (x - x0) * df(x0)
print(t1(x0))

# Aufgabe 2b
x = np.linspace(-np.pi, np.pi, 100)
plt.figure()
plt.plot(x, t1(x), label='$t_1(x)$')
plt.plot(x, f(x), '--', label="$f(x)$")
plt.legend()
plt.show()

# Aufgabe 2c
error = f(x1) - t1(x1)
print(error)
"""
# Aufgabe 3a
x0 = 2.0
x1 = 8/5
def f0(x):
    return 1/3*x**3 + 1/2*x**2 - x -1
def f1(x):
    return x**2+x-1
def f2(x):
    return 2*x+1
def f3(x):
    return (math.exp(3*x/2)*math.sin(math.exp(x/2) - 1) - math.exp(x/2)*math.sin(math.exp(x/2) - 1) - 3*math.exp(x)*math.cos(math.exp(x/2) - 1))/8
fk_list = [f0(x0), f1(x0), f2(x0)]

def taylor_factory(x0, fk_list):
    derivatives = [fk for fk in fk_list]
    factorials = np.array([math.factorial(n) for n in range(len(fk_list))])
    coefficients = derivatives / factorials
    return lambda x: np.sum([ck * (x - x0)**k for k, ck in enumerate(coefficients)], axis=0)

t3 = taylor_factory(x0, fk_list)
print(f"moi {t3(x0)}")

"""
# Aufgabe 3b
x_vals = np.linspace(-1, 1)
t3_vals = [t3(x_val) for x_val in x_vals]
f_vals = [f0(x_val) for x_val in x_vals]
plt.plot(x_vals, t3_vals, label='$t_3(x)$')
plt.plot(x_vals, f_vals, '--', label='$f(x)$')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Taylorpolynom 3. Grades und $f(x)$')
plt.grid(True)
plt.show()
"""
# Aufgabe c
app = t3(x1)-f0(x1)
print(app)


# Aufgabe 4a
x0 = 0.0
def f(x):
    return math.exp(x)

fk_list = [f(x0), f(x0), f(x0), f(x0)]
def taylor_factory(x0, fk_list):
    derivatives = np.array([fk for fk in fk_list])
    factorials = np.array([math.factorial(n) for n in range(len(fk_list))])
    coefficients = derivatives / factorials

    return lambda x: np.sum([ck * (x - x0)**k for k, ck in enumerate(coefficients)], axis=0)

t3 = taylor_factory(x0, fk_list)
print(t3(x0))

# Aufgabe 4b
def taylor_factory_n(x0, n):
    fk_list = [f(x0) for _ in range(n)]
    derivatives = np.array([fk for fk in fk_list])
    factorials = np.array([math.factorial(n) for n in range(len(fk_list))])
    coefficients = derivatives / factorials

    return lambda x: np.sum([ck * (x - x0)**k for k, ck in enumerate(coefficients)], axis=0)

tx = taylor_factory_n(x0, 5)
print(tx(x0))
"""
# Aufgabe 4c
x = 1.0
n_values = [5, 10, 20, 40]

for n in n_values:
    exact_value = np.exp(x)
    approx_value = taylor_factory_n(x, n)
    error = abs(exact_value - approx_value(x))
    print(f"For n = {n}:")
    print(f"Exact Value: {exact_value}")
    print(f"Approximated Value: {approx_value(x)}")
    print(f"Error: {error}\n")

"""
