#!/usr/bin/env python3
import Max_Sat_interface as mxs
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import general_lp_interface as lp

OUT_NAME = "output_file"

# pre-init parameter
N = 100
number_of_trials = 1
Sus = 4
P = 0.01
Lambda = 0.9


class Trial:
    def __init__(self, var):
        self.E = []  # on number of tests
        self.P = []
        self.E_W = []
        self.P_W = []
        self.t_e = []  # temporary error
        self.t_h = []  # temporary hamming distance
        self.t_e_w = []  # temporary error
        self.t_h_w = []  # temporary hamming distance
        self.lp_E = []  # on number of tests
        self.lp_P = []
        self.lp_E_C = []  # on number of tests
        self.lp_P_C = []
        self.lp_t_e = []  # temporary error
        self.lp_t_h = []  # temporary hamming distance
        self.lp_t_e_c = []  # temporary error
        self.lp_t_h_c = []  # temporary hamming distance
        self.var = var


def main_comparison_maxhs_lp(n, p, noiseless, u):
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

    # T = [1, 10, 30]

    nw_trials = []

    noise_weight = [0.0]

    lambda_w = round((np.log((1-noise_probability)/noise_probability)) / (np.log((1-(k/n))/(k/n))), 2)

    if not noiseless:
        noise_weight = [lambda_w]

    for i in noise_weight:
        nw_trials.append(Trial(i))

    mean_time_lp = []
    mean_time_maxhs = []

    success_lp = []
    success_maxhs = []

    tr = nw_trials[0]

    # for every t number of tests
    for t in T:
        print(t)

        time_lp = []
        time_maxhs = []

        tr.t_h = []
        tr.lp_t_h = []
        tr.lp_t_e = []
        tr.t_e = []

        for i in range(number_of_trials):

            a = gtf.generate_pool_matrix(n, k, t)

            y = gtf.get_results(t, a, x, noiseless, noise_probability)

            # *************MAX_HS*************

            mxs.output(n, t, x, y, a, noiseless, tr.var)

            r, noise, tm = mxs.call_Max_Sat(n)

            # add execution time
            time_maxhs.append(tm)
            # calculating hamming distance between model result and input x
            hamming_distance = sd.hamming(x_s, r)
            tr.t_h.append(hamming_distance)

            # there's an error?
            if hamming_distance > 0:
                tr.t_e.append(1)
            else:
                tr.t_e.append(0)

            # *************LP_RELAX*************
            r_lp_i, tm = lp.solve(y, a, n, noiseless)

            # add execution time
            time_lp.append(tm)

            # ****CAST****

            r_lp = [int(i) for i in r_lp_i[:n]]

            # calculating hamming distance between model result and input x
            hamming_distance = sd.hamming(x_s, r_lp)
            tr.lp_t_h.append(hamming_distance)

            # there's an error?
            if hamming_distance > 0:
                tr.lp_t_e.append(1)
            else:
                tr.lp_t_e.append(0)

        mean_time_lp.append(np.mean(time_lp))
        mean_time_maxhs.append(np.mean(time_maxhs))
        for tr in nw_trials:
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))
            tr.lp_E.append(np.mean(tr.lp_t_h))
            tr.lp_P.append(1 - np.mean(tr.lp_t_e))

    X = []

    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    with open(OUT_NAME + "-" + str(u), "w") as output_file:

        output_string = ''

        output_string += str(n) + "\n" + str(k) + "\n" + str(lambda_w) + "\n"

        output_string += str(T) + "\n"

        for tr in nw_trials:
            output_string += str(tr.E) + "\n" + str(tr.P) + "\n"

        output_string += str(mean_time_maxhs) + "\n"

        for tr in nw_trials:
            output_string += str(tr.lp_E) + "\n" + str(tr.lp_P) + "\n"

        output_string += str(mean_time_lp) + "\n"

        output_file.write(output_string)


def main_varying_maxhs(n, p, noiseless, u):

    n_trials = number_of_trials

    k = round(n * p)

    x, k_g = gtf.generate_input(n, p)

    # avoid k = 0, trivial example
    while (abs(k_g - k) / n) >= 0.01 or k_g == 0:
        print("choosing")
        x, k_g = gtf.generate_input(n, p)

    x_s = [1 if i in x else 0 for i in range(1, n + 1)]

    noise_probability = 0.05

    lambda_w = round((np.log((1 - noise_probability) / noise_probability)) / (np.log((1 - (k / n)) / (k / n))), 2)

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


main_comparison_maxhs_lp(100, 0.01, True, 100)

'''
main_varying_maxhs(1000, 0.03, False, 22)
main_varying_maxhs(1000, 0.05, False, 23) 
main_varying_maxhs(1000, 0.07, False, 24)
main_varying_maxhs(2000, 0.01, False, 25)
main_varying_maxhs(2000, 0.01, False, 26)
main_varying_maxhs(2000, 0.03, False, 27)
main_varying_maxhs(2000, 0.05, False, 28)
main_varying_maxhs(2000, 0.07, False, 29)
main_varying_maxhs(4000, 0.01, False, 30)
main_varying_maxhs(4000, 0.03, False, 31)
main_varying_maxhs(4000, 0.05, False, 32)
main_varying_maxhs(4000, 0.07, False, 33)
main_varying_maxhs(8000, 0.01, False, 34)
main_varying_maxhs(8000, 0.03, False, 35)
main_varying_maxhs(8000, 0.05, False, 36)
main_varying_maxhs(8000, 0.07, False, 37)
main_varying_maxhs(10000, 0.01, False, 38)
main_varying_maxhs(10000, 0.03, False, 39)
main_varying_maxhs(10000, 0.05, False, 40)
main_varying_maxhs(10000, 0.07, False, 41)
'''



