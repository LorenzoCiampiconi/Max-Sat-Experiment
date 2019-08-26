import Max_Sat_interface as mxs
import general_lp_interface as lp
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd

weighted = False


class trial:
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


def main(n, p, noiseless):
    k = round(n * p)

    x, k_g = gtf.generate_input(n, p)

    # avoid k = 0, trivial example
    # avoid k = 0, trivial example
    while (abs(k_g - k) / n) >= 0.01 or k_g == 0:
        print("choosing")
        x, k_g = gtf.generate_input(n, p)

    x_s = [1 if i in x else 0 for i in range(1, n + 1)]

    noise_probability = 0.05

    T = [1, int(n / (5* np.log2(n)) +1)]

    # generate T
    i = 1
    # until we reach the condition of success have more dense trials
    while T[i] < k * np.log2(n / k):
        if T[i] - T[i - 1] > round(n/100):
            T.append(int(2 * T[i] - T[i - 1] - round(n/100)))
        else:
            T.append(int(T[i] + round(n/100)))
        i += 1

    # dense trials around k* log2(n/k)
    while T[i] < n:
        T.append(int(2 * T[i] - T[i - 1] + round(n/100)))

        i += 1

    # T = [1, 10, 30]

    nw_trials = []

    noise_weight = [0.0]

    if not noiseless:
        noise_weight = [round((np.log((1-noise_probability)/noise_probability)) / (np.log((1-(k/n))/(k/n))), 2)]
        #noise_weight = [0.9]
        print(k)
        print(n)
        print(noise_weight)

    for i in noise_weight:
        nw_trials.append(trial(i))

    mean_time_lp = []
    mean_time_maxhs = []

    # for every t number of tests
    for t in T:
        print(t)

        time_lp = []
        time_maxhs = []

        for tr in nw_trials:
            # Max_HS
            tr.t_e = []  # blank temporary
            tr.t_h = []  # blank temporary
            tr.t_e_w = [] # blank temporary
            tr.t_h_w = []  # blank temporary
            # LP-Relax
            tr.lp_t_e = []  # blank temporary
            tr.lp_t_h = []  # blank temporary
            tr.lp_t_e_c = []  # blank temporary
            tr.lp_t_h_c = []  # blank temporary

        for i in range(100):
            a = gtf.generate_pool_matrix(n, k, t)

            for tr in nw_trials:
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

                # *************NON-COMPACT*************
                mxs.non_compact_output(n, t, x, y, a, noiseless, tr.var)

                r, noise, tm = mxs.call_Max_Sat(n)


                # add execution time
                time_lp.append(tm)
                # calculating hamming distance between model result and input x
                hamming_distance = sd.hamming(x_s, r)
                tr.lp_t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.lp_t_e.append(1)
                else:
                    tr.lp_t_e.append(0)

        for tr in nw_trials:
            # MAX_HS
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))
            print(1 - np.mean(tr.t_e))
            # LP_RELAX
            tr.lp_E.append(np.mean(tr.lp_t_h))
            tr.lp_P.append(1 - np.mean(tr.lp_t_e))
            print(1 - np.mean(tr.lp_t_e))

        mean_time_lp.append(np.mean(time_lp))
        mean_time_maxhs.append(np.mean(time_maxhs))

    X = []

    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    for tr in nw_trials:
        values = [[tr.lp_P[i], tr.P[i]] for i in range(len(tr.P))]
        data = pd.DataFrame(values, T, columns=["non-compact", "compact"])

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        plt.plot(T, tr.P, "yo")
        plt.plot(T, tr.lp_P, "bx")
        plt.plot(X, tr.P, "r", label = "Recovery Bound", linewidth=2.5)
        plt.title("Probability of success with  n = " + str(n) + " k = " + str(k), fontsize=13)
        plt.xlabel("Number of tests m", fontsize=13)
        plt.ylabel("Probability of success", fontsize=13)
        plt.legend(loc="lower right")
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        plt.savefig("PS, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()

        values = [[tr.lp_E[i], tr.E[i]] for i in range(len(tr.P))]
        data = pd.DataFrame(values, T, columns=["non-compact", "compact"])

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        plt.plot(T, tr.E, "yo")
        plt.plot(T, tr.lp_E, "bx")
        plt.plot(X, tr.E, "r", label = "Recovery Bound", linewidth=2.5)
        plt.title("Hamming distance trend with  n = " + str(n) + " k = " + str(k), fontsize=13)
        plt.xlabel("Number of tests m", fontsize=13)
        plt.ylabel("Error (Hamming Distance)", fontsize=13)
        plt.legend(loc = "upper right")
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        plt.savefig("HD, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()

    values = [[mean_time_lp[i], mean_time_maxhs[i]] for i in range(len(mean_time_maxhs))]
    data = pd.DataFrame(values, T, columns=["XOR-MaxSAT", "MaxSAT"])

    a = [0] + mean_time_maxhs[1:]
    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.plot(T, mean_time_lp, "bx")
    plt.plot(T, mean_time_maxhs, "yo")
    plt.plot(X, a, "r", label="Recovery Bound", linewidth=2.5)
    plt.title("n = " + str(n) + " k = " + str(k), fontsize=13)
    plt.xlabel("Number of tests m", fontsize=13)
    plt.ylabel("Time (s)", fontsize=13)
    plt.legend(loc="upper right")
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.savefig(" MAXTime PS, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + "noiseless= " + str(
        noiseless) + ".png")
    plt.show()


main(50, 0.02, False)
