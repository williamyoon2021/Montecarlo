
import numpy as np
import statistics

# q: probability mass distribution
def q(m, n, spike):
    return 1/(1 + (abs(m - 50)) ** spike + (abs(n - 50)) ** spike)

# simulation: implements SLLN approach
def simulation(spike, N):
    normalization = 0
    for m in list(range(100)):
        for n in list(range(100)):
            normalization += q(m, n, spike)
            time_avg = 0

    for k in list(range(N)):
        m = np.random.randint(0,100)
        n = np.random.randint(0,100)
        time_avg = (k*time_avg + 10000*(m + n )*q(m , n, spike)/normalization)/(k + 1)

    return time_avg

# trials: run trials to compute average E_p(f) for various (spike, N) combinations
def trials():
    specific_inputs = [
        (0.2, 10000),
        (0.2, 50000),
        (0.2, 100000),

        (2, 10000),
        (2, 50000),
        (2, 100000),

        (11, 10000),
        (11, 50000),
        (11, 100000),
    ]

    for spike, N in specific_inputs:
        # run 10 independent chains and compute mean of f(m, n) for each
        histogram = [simulation(spike, N) for _ in range(10)]

        # compute the average across the 10 runs
        mean = statistics.mean(histogram)

        print(f'Mean = {mean}' + f', Spike = {spike}, N = {N}')

def user_input():
    spike = float(input('Enter spikiness value: '))
    N = int(input('Enter number of iterations: '))
    time_avg = simulation(spike, N)
    print('Computed mean = ' + str(time_avg) + f', spike = {spike}, N = {N}')

def main():
    # user_input()
    trials()
    
if __name__ == "__main__":
    main()