import math
import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 2a
def bisection3(f, a, b, max=3):
    iterations = 0
    assert(f(a) * f(b) < 0.0)
    while iterations < max:
        iterations += 1
        m = (a + b) / 2
        fm = f(m)
        if fm == 0.0:
            return m, m
        elif fm * f(b) < 0.0:
            a = m
        else:
            b = m
    return [a, b], iterations
f = lambda x: x - 2*np.exp(-x)
a = 0.0
b = 1.0
root, iterations = bisection3(f, a,b)
print(f"Aufgabe 2a {root, iterations}")

# Aufgabe 2b
def newton(f, df, x, max = 3):
    iterations = 0
    while iterations < max:
        x = x - f(x) / df(x)
        iterations += 1
    return x, iterations

f = lambda x: x - 2*np.exp(-x)
df = lambda x: 2*np.exp(1)
x = 1.0

root, iterations = newton(f, df, x)
print(f"Aufgabe 2b {root, iterations}")

# Aufgabe 3a
def newton(f, df, x, tol):
    iterations = 0
    while abs(f(x)) > tol:
        x = x - f(x) / df(x)
        iterations += 1
    return x, iterations

f = lambda x: np.exp(-x-1) - np.log(x**2+1)
df = lambda x: -np.exp(-x-1) - 1/(x**2+1)*2*x
x = 0.0

root, iterations = newton(f, df, x, 10.0e-10)
print(f"Aufgabe 3a {root, iterations}")

# Aufgabe 3b
def bisection(f, a, b, tol):
    iterations = 0
    assert(f(a) * f(b) < 0.0)
    while abs(b-a) > tol:
        iterations += 1
        m = (a + b) / 2
        fm = f(m)
        if fm == 0.0:
            return m, m
        elif fm * f(b) < 0.0:
            a = m
        else:
            b = m
    return [a, b], iterations

f = lambda x: np.exp(-x-1) - np.log(x**2+1)
a = 0.0
b = 1.0

root, iterations = bisection(f, a, b, 10.0e-10)
print(f"Aufgabe 3b {root, iterations}")

# Aufgabe 4a (Newton)
def find_intervals(f, x_range=(-10, 10), step=1):
    a, b = x_range
    intervals = []
    x = a
    while x < b:
        if f(x) * f(x + step) <= 0:
            intervals.append((x, x + step))
        x += step
    return intervals

def newton(f, df, x, tol):
    while abs(f(x)) > tol:
        x = x - f(x) / df(x)
    return x

f = lambda x: x**3 - 1.9*x**2 - 5.1*x + 5.3
df = lambda x: 3*x**2 - 3.8*x - 5.1
intervals = find_intervals(f)
def extract(intervals):
    return [item[0] for item in intervals]
lst = extract(intervals)
print("Aufgabe 4a")
for i in lst:
    print(newton(f,df,i,1.0e-8))


# Aufgabe 4b (Bisektion)
def bisection(f, a, b, tol):
    assert(f(a) * f(b) < 0.0)
    while abs(b-a) > tol:
        m = (a + b) / 2
        fm = f(m)
        if fm == 0.0:
            return m, m
        elif fm * f(b) < 0.0:
            a = m
        else:
            b = m
    return [a, b]

f = lambda x: x**3 - 1.9*x**2 - 5.1*x + 5.3
intervals = find_intervals(f)
print("Aufgabe 4b")
for sublist in intervals:
    for index, item in enumerate(sublist):
        if index == 0:
            first_item = item
        elif index == 1:
            second_item = item
    print(bisection(f, first_item, second_item, 1.0e-8))

# Überprüfung
def i(x):
    return x**3 - 1.9*x**2 - 5.1*x + 5.3
print(i(3.009463168680668))
print(i(0.883614182472229))
print(i(-1.9930773675441742))

# Aufgabe 5a
def bisection(f, a, b, tol):
    assert f(a) * f(b) < 0.0
    while abs(b-a) > tol:
        m = (a + b) / 2
        fm = f(m)
        if fm == 0.0:
            return [m, m]
        elif fm * f(b) < 0.0:
            a = m
        else:
            b = m
    return [a, b]

def A(p):
    return 0.01 * p
def N(p):
    if p == 0.0:
        return float('inf')
    else:
        return p**(-0.2) + p**(-0.4)
def F(p):
    return A(p) - N(p)

print(f"Aufgabe 5a {bisection(F,a= 0, b=1000, tol =10.0e-10)}")

# Aufgabe 5b
def find_intervals(f, x_range=(0.01, 100), step=0.1):
    a, b = x_range
    intervals = []
    x = a
    while x < b:
        if f(x) * f(x + step) <= 0:
            intervals.append((x, x + step))
        x += step
    return intervals

def newton(f, df, x, tol):
    while abs(f(x)) > tol:
        x = x - f(x) / df(x)
    return x

f = lambda x: x**(-0.2) + x**(-0.4) - 0.01 * x
df = lambda x: -0.2*x**1.2 - 0.4*x**1.4 - 0.01
intervals = find_intervals(f)
def extract(intervals):
    return [item[0] for item in intervals]
lst = extract(intervals)
print("Aufgabe 5b")
for i in lst:
    print(newton(f,df,i,10.0e-10))
