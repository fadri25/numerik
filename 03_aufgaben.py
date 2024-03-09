import math
import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 3a
x0 = 0.0
x1 = -0.6
def f0(x):
    return math.cos(math.exp(x/2) - 1)
def f1(x):
    return -math.exp(x/2)*math.sin(math.exp(x/2) - 1)/2
def f2(x):
    return -(math.exp(x/2)*math.sin(math.exp(x/2) - 1) + math.exp(x)*math.cos(math.exp(x/2) - 1))/4
def f3(x):
    return (math.exp(3*x/2)*math.sin(math.exp(x/2) - 1) - math.exp(x/2)*math.sin(math.exp(x/2) - 1) - 3*math.exp(x)*math.cos(math.exp(x/2) - 1))/8
fk_list = [f0(x0), f1(x0), f2(x0), f3(x0)]

def taylor_factory(x0, fk_list):
    derivatives = [fk for fk in fk_list]
    factorials = np.array([math.factorial(n) for n in range(len(fk_list))])
    coefficients = derivatives / factorials
    return lambda x: np.sum([ck * (x - x0)**k for k, ck in enumerate(coefficients)], axis=0)

t3 = taylor_factory(x0, fk_list)
print(t3(x0))

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

# Aufgabe c
app = t3(x1)-f0(x1)
print(app)

