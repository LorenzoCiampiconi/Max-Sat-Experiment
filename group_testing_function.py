import numpy as np
import numpy.random as nrndm



def generate_input(n, p):
    # input vector
    x = [0] * n

    # generate a sparse input to recover
    for i in range(n):
        x[i] = nrndm.choice(range(0, 2), p=[0.99, 0.01])

    #counting number of faulty items
    k = x.count(1)

    return x, k

def generate_pool_matrix(n, k, t):
    a = np.array(np.zeros((t, n), dtype= int))
    for i in range(t):
        for j in range(n):
            a[i][j] = nrndm.choice(range(0, 2), p=[(1 - (np.log(2))/k),  (np.log(2)/k)])
    #print("matrix generated, getting result")
    return a

def get_results(t, a, x, noiseless, noise_probability):
    #vector of the tests
    y = [0] * t
    # get result of test
    if noiseless:
        for i in range(t):
            if 1 in (a[i] * x):
                y[i] = 1
    else:
        for i in range(t):
            if 1 in (a[i] * x):
                y[i] = 1
            y[i] = abs(y[i] - nrndm.choice(range(0, 2), p = [1 - noise_probability, noise_probability]))

    return y




