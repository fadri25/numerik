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
"""
A(p) = 0.01 · p und N(p) = p^-0.2 + p^-0.4
a) Bestimmen Sie mittels Bisektion den Produktpreis auf tol = 10^-10 genau.
b) Bestimmen Sie mit dem Newton-Verfahren den Produktpreis auf tol = 10^-10 genau
"""
# Aufgabe 5a
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

A = lambda x: 0.01 * x
N = lambda x: x**-0.2 + x^-0.4

print(bisection(f, a, b, 10.0e-10))


