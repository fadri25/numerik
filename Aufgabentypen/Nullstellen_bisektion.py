import numpy as np
import sympy as sp

def bisection(f, a, b, max=3):
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

def find_intervals(f, x_range=(-10, 10), step=1):
    a, b = x_range
    intervals = []
    x = a
    while x < b:
        if f(x) * f(x + step) <= 0:
            intervals.append((x, x + step))
        x += step
    return intervals

f = lambda x: x - 2*np.exp(-x)
a = 0.0
b = 1.0
root, iterations = bisection(f, a,b)
print(f"Aufgabe 2a {root, iterations}")

f = lambda x: x**3 - 1.9*x**2 - 5.1*x + 5.3
intervals = find_intervals(f)
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