import numpy as np
import numpy.random as nrndm
import random as rndm


def generate_input(n, p_i):

    # generate a sparse input to recover
    x = rndm.sample(range(1, n + 1), nrndm.binomial(n,  p_i))

    # counting number of faulty items
    k = len(x)

    return x, k


def generate_pool_matrix(n, k, t):

    a= []
    for i in range(t):
            a.append(rndm.sample(range(1, n + 1), nrndm.binomial(n, (np.log(2)/k))))
    # print("matrix generated, getting result")
    return a


def get_results(t, a, x, noiseless, noise_probability):
    # vector of the tests
    y = [0] * t
    # get result of test
    if noiseless:
        for i in range(t):
            if any(x_i in a[i] for x_i in x):
                y[i] = 1
    else:
        for i in range(t):
            if any(x_i in a[i] for x_i in x):
                y[i] = 1
            y[i] = abs(y[i] - nrndm.choice(range(0, 2), p = [1 - noise_probability, noise_probability]))

    return y

