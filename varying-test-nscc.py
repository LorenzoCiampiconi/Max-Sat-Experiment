#!/usr/bin/env python3
import Max_Sat_interface as mxs
import group_testing_function as gtf
import numpy as np

OUT_NAME = "output_file"

# pre-init parameter
N = 100
N_trials = 1
Sus = 4
P = 0.01
Lambda = 0.9




class Trial:
    def __init__(self, var):
        self.E = []  # on number of tests
        self.P = []
        self.t_e = []  # temporary error
        self.t_h = []  # temporary hamming distance
        self.var = var


def main(n, p, noiseless, u):

    n_trials = N_trials


    lambda_w = Lambda

    k = round(n * p)

    x, k_g = gtf.generate_input(n, p)

    # avoid k = 0, trivial example
    while (abs(k_g - k) / n) >= 0.01 or k_g == 0:
        print("choosing")
        x, k_g = gtf.generate_input(n, p)

    x_s = [1 if i in x else 0 for i in range(1, n + 1)]

    noise_probability = 0.05

    T = [1, int(n / (5 * np.log2(n)) + 1)]

    # generate T
    i = 1
    # until we reach the condition of success have more dense trials
    while T[i] < k * np.log2(n / k):
        if T[i] - T[i - 1] > round(n / 100):
            T.append(int(2 * T[i] - T[i - 1] - round(n / 100)))
        else:
            T.append(int(T[i] + round(n / 100)))
        i += 1

    # dense trials around k* log2(n/k)
    while T[i] < n:
        T.append(int(2 * T[i] - T[i - 1] + round(n / 100)))

        i += 1

    '''

    T = [1, int(n / (5 * np.log2(n)) + 1)]

    # generate T
    i = 1

    # until we reach the condition of success have more dense trials
    while T[i] < k * np.log2(n / k):
        if T[i] - T[i - 1] > 1:
            T.append(int(2 * T[i] - T[i - 1] - 1))
        else:
            T.append(int(T[i] + 1))
        i += 1

    # dense trials around k*log2(n/k)
    while T[i] < n:
        T.append(int(2 * T[i] - T[i - 1] + 1))

        i += 1

    '''

    # T = [1, 10, 30]

    nw_trials = []

    noise_weight = [0.0]

    if not noiseless:
        noise_weight = [lambda_w]

    for i in noise_weight:
        nw_trials.append(Trial(i))

    time_MAXHS = []

    # for every t number of tests
    for t in T:
        print(t)

        for tr in nw_trials:
            tr.t_e = []  # blank temporary
            tr.t_h = []  # blank temporary

        for i in range(n_trials):
            a = gtf.generate_pool_matrix(n, k, t)

            for tr in nw_trials:
                y = gtf.get_results(t, a, x, noiseless, noise_probability)
                mxs.output(n, t, x, y, a, noiseless, tr.var)

                tests = (k*t*np.log10(n))/(n)

                r, noise, tm = mxs.call_Max_Sat(n)

                # add execution time
                time_MAXHS.append(tm)

                # calculating hamming distance between model result and input x

                vs = [[r[j], x_s[j]] for j in range(len(r))]
                hamming_distance = sum([1 if vs_i[0] != vs_i[1] else 0 for vs_i in vs])

                tr.t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.t_e.append(1)
                else:
                    tr.t_e.append(0)

        for tr in nw_trials:
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))

    X = []

    t_maxhs = np.mean(time_MAXHS)

    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    with open(OUT_NAME + "-" + str(u), "w") as output_file:

        output_string = ''

        output_string += str(n) + "\n" + str(k) + "\n" + str(lambda_w) + "\n"

        output_string += str(T) + "\n"

        for tr in nw_trials:
            output_string += str(tr.E) + "\n" + str(tr.P) + "\n"

        output_string += str(t_maxhs) + "\n"

        output_file.write(output_string)


main(750, 0.1, True, 1)
main(750, 0.3, True, 2)
main(750, 0.5, True, 3)
main(750, 0.7, True, 4)

main(8000, 0.01, True, 5)
main(8000, 0.02, True, 6)
main(8000, 0.03, True, 7)
main(8000, 0.04, True, 8)
