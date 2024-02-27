import numpy as np
"""
# Aufgabe 2a
x = np.array([1,2,3,4])
y = np.array([1,1,1,1])
sum = 0
for i in range(0,4):
    sum += x[i] * y[i]
print(sum)

# Aufgabe 2b
print(np.dot(x,y))
    
# Aufgabe 3a
p = np.array([[1,0,0,0],
              [0,0,0,1],
              [0,0,1,0],
              [0,1,0,0]])
a = np.array([[1,1,1,1],
              [2,2,2,2],
              [3,3,3,3],
              [4,4,4,4]])
print(np.dot(p,a))
print(np.dot(a,p))

# Aufgabe 3b
print(np.linalg.inv(p))
"""
# Aufgabe 4
def f2c(temp):
    cel = (temp -32)*(5/9)
    return cel
temp = int(input("Gib die Temperatur ein: "))
print(f2c(temp))
"""
# Aufgabe 5
def pi_approximation(N):
    result = 0
    for k in range(N + 1):
        num = np.math.factorial(4*k) * (1103 + 26390*k)
        den = np.math.factorial(k)**4 * 396**(4*k)
        result += num / den
    result *= np.sqrt(8) / 9801
    return 1 / result
print(pi_approximation(1))
print(np.pi)

# Aufgabe 6c
plt.figure(1)
x = np.linspace(1e-4,10,100)
y = 4 / np.sqrt(2*x**3)
plt.xlabel( 'x' )
plt.ylabel( '4/sqrt(2*x^3)' )
plt.title( 'Plot von y = 4/sqrt(2*x^3)  in logarithmischer x- und y-Achse' )	
plt.loglog( x, y, color='red' )
plt.show()

# d)
plt.figure(2)
x = np.linspace(1e-4,10,100)
y = 5*(3**x)**2
plt.xlabel( 'x' )
plt.ylabel( '5*(3^x)^2' )
plt.title( 'Plot von y = 5*(3^x)^2 in logarithmischer y-Achse' )	
plt.semilogy( x, y, color='red' )
plt.show()
"""