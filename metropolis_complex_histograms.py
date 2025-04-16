
import numpy as np
import random
import statistics
import matplotlib.pyplot as plt


# propose: uniformly choose over nearest {steps} neighbors
def propose(pair, steps):
    
    # initialize
    m = pair[0]
    n = pair[1]

    # store neighbor possibilities
    neighbors = []

    # loop over step x step box
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            neighbors.append(((m + i) % 100, (n + j) % 100))


    # random probability of choosing each neighbor
    return random.choice(neighbors)


# q: probability mass distribution
def q(m, n, spike):
    return 1/(1 + (abs(m - 50)) ** spike + (abs(n - 50)) ** spike)

# f: given function
def f(m, n):
    return m + n

# accept_reject: accept/reject functionality
def accept_reject(pair, neighbor, spike):

    # Xk
    m_pair = pair[0]
    n_pair = pair[1]
    
    # Yk+1
    m_neighbor = neighbor[0]
    n_neighbor = neighbor[1]

    # distinct distributions
    pair_q = q(m_pair, n_pair, spike)
    neighbor_q = q(m_neighbor, n_neighbor, spike)

    # Yk+1 >= Xk
    if neighbor_q >= pair_q:
        return neighbor

    # Yk+1 < Xk
    # <= probability Yk+1/Xk sets Xk+1 = Xk
    if np.random.random() <= neighbor_q / pair_q:
        return neighbor
    
    # same as 1-Yk+1/Xk
    return pair

# chain: generate chain of points
def chain(spike, N, steps):

    # generate a random m,n
    pairs = [(np.random.randint(0, 99), np.random.randint(0, 99))]
    for _ in range(N):
        # use the most recent pair as the Xk
        pair = pairs[-1]

        # select a neighbor
        neighbor = propose(pair, steps)

        # append the result of the accept/reject process with the neighbor
        pairs.append(accept_reject(pair, neighbor, spike))

    # returns the chain of m,n pairs
    return pairs


# trials: run trials to compute average E_p(f) for various (spike, N) combinations
def trials():
    specific_inputs = [
        (0.2, 50000),
        (2, 50000),
        (11, 50000)
    ]

    steps = int(input("Enter step size: "))

    for spike, N in specific_inputs:
        # run 10 independent chains and compute mean of f(m, n) for each
        results = [statistics.mean(f(m, n) for m, n in chain(spike, N, steps)) for _ in range(10)]
        
        # compute the average across the 10 runs
        mean = statistics.mean(results)

        print(f'Computed mean = {mean}' + f', spike = {spike}, N = {N}, step_size = {steps}')


# plot_comparison: show empirical histogram and theoretical distribution side by side
def plot_comparison(pairs, spike, N, steps):
    # extract (m, n) samples
    m_vals = [m for m, _ in pairs]
    n_vals = [n for _, n in pairs]

    # compute empirical 2D histogram
    hist, _, _ = np.histogram2d(m_vals, n_vals, bins=100, range=[[0, 99], [0, 99]])
    m_grid, n_grid = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")

    # compute theoretical distribution
    q_grid = np.array([[q(m, n, spike) for n in range(100)] for m in range(100)])
    norm_q = q_grid / np.sum(q_grid)

    # create side-by-side 3D plots
    fig = plt.figure(figsize=(14, 6))

    # empirical histogram
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.bar3d(m_grid.ravel(), n_grid.ravel(), np.zeros_like(hist.ravel()),
              1, 1, hist.ravel(), zsort='average')
    ax1.set_title(f"Actual Histogram, N={N}, Spike={spike}, Step Size={steps}")
    ax1.set_xlabel("m")
    ax1.set_ylabel("n")
    ax1.set_zlabel("Frequency")

    # theoretical distribution
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(m_grid, n_grid, norm_q, cmap='viridis')
    ax2.set_title(f"Theoretical Histogram, N={N}, Spike={spike}, Step Size={steps}")
    ax2.set_xlabel("m")
    ax2.set_ylabel("n")
    ax2.set_zlabel("p(m,n)")

    plt.tight_layout()
    plt.show()


# histogram_generation: prompt user for spike and N and step size, run simulation, 
# and plot empirical vs. theoretical histograms
def histogram_generation():
    spike = float(input('Enter spikiness value: '))
    N = int(input('Enter number of iterations: '))
    steps = int(input('Enter step size: '))

    # run chain to collect samples
    samples = chain(spike, N, steps)
    mean = statistics.mean(f(m, n) for m, n in samples)
    print(f'Mean = {mean}' + f', Spike = {spike}, N = {N}')

    # visualize the empirical and theoretical distributions
    plot_comparison(samples, spike, N, steps)


def main():
    # histogram_generation()
    trials()
    
if __name__ == "__main__":
    main()
