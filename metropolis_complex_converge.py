
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


# convergence_chain: return running averages of f(m, n) over chain
def convergence_chain(spike, N, steps):
    # generate a random m,n
    pairs = [(np.random.randint(0, 99), np.random.randint(0, 99))]

    # generate list of running averages
    running_averages = [f(*pairs[0])]

    for i in range(1, N):
        # use the most recent pair as the Xk
        pair = pairs[-1]

        # select a neighbor
        neighbor = propose(pair, steps)

        # append the result of the accept/reject process with the neighbor
        next_state = accept_reject(pair, neighbor, spike)
        pairs.append(next_state)

        # compute new running average
        new_avg = (running_averages[-1] * i + f(*next_state)) / (i + 1)
        running_averages.append(new_avg)

    return running_averages

# plot_convergence: compare convergence across spike & step_size grouped by N
def plot_convergence(specific_inputs, step_sizes, num_trials):
    iterations = [100000]

    for N in iterations:
        plt.figure(figsize=(10, 6))
        for spike in {spike for (spike, n) in specific_inputs if n == N}:
            for step in step_sizes:
                # collect trial averages
                trial_means = [convergence_chain(spike, N, step) for _ in range(num_trials)]
                avg_means = np.mean(trial_means, axis=0)

                # plot average running mean
                plt.plot(avg_means, label=f"Spike={spike}, Step={step}")

        # plot true expected value line
        plt.axhline(y=100, color='gray', linestyle='--', label='Actual Mean, 100')
        plt.xlabel("Iterations")
        plt.ylabel("Running Mean")
        plt.title(f"Convergence, N = {N}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def convergence_test():
    # step sizes 
    step_sizes = list(map(int, input("Enter step sizes (space-separated): ").split()))

    specific_inputs = [
        (11, 100000)
    ]

    # plot convergence using 10 trials
    plot_convergence(specific_inputs, step_sizes, 10)

def main():
    convergence_test()
    
if __name__ == "__main__":
    main()
