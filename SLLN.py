
import numpy as np
spike = float(input('Enter spikiness value: '))
N = int(input('Enter number of iterations: '))

def q(m, n):
    return 1/(1 + (abs(m - 50))**spike + (abs(n - 50))**spike)

normalization = 0
for m in list(range(100)):
    for n in list(range(100)):
        normalization += q(m, n)
        time_avg = 0

for k in list(range(N)):
    m = np.random.randint(0,100)
    n = np.random.randint(0,100)
    time_avg = (k*time_avg + 10000*(m + n )*q(m , n)/normalization)/(k + 1)
    
print('Computed mean = ' + str(time_avg))