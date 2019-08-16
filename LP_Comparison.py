import Max_Sat_interface as mxs
import general_lp_interface as lp
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd


n = 250

noiseless = False
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


def main():
    x, k = gtf.generate_input(n, 5/n)

    # avoid k = 0, trivial example
    while k != 4:
        x, k = gtf.generate_input(n, 5 / n)

    x_s = [1 if i in x else 0 for i in range(1, n + 1)]

    noise_probability = 0.05

    T = [1, int(n / (5* np.log2(n)) +1)]

    # generate T
    i = 1
    # until we reach the condition of success have more dense trials
    while T[i] < k * np.log2(n / k):
        if T[i] - T[i - 1] > 1:
            T.append(int(2 * T[i] - T[i - 1] - 1))
        else:
            T.append(int(T[i] + 1))
        i += 1

    # dense trials around k* log2(n/k)
    while T[i] < n:
        T.append(int(2 * T[i] - T[i - 1] + 1))

        i += 1

    # T = [1, 10, 30]

    nw_trials = []

    noise_weight = [0.0]

    if not noiseless:
        noise_weight = [0.9]

    for i in noise_weight:
        nw_trials.append(trial(i))

    time_LP = []
    time_MAXHS = []
    time_MAXHS_W = []

    # for every t number of tests
    for t in T:
        print(t)

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

        for i in range(75):
            a = gtf.generate_pool_matrix(n, k, t)

            for tr in nw_trials:
                y = gtf.get_results(t, a, x, noiseless, noise_probability)

                # *************MAX_HS*************

                mxs.output(n, t, x, y, a, noiseless, tr.var)

                r, noise, tm = mxs.call_Max_Sat(n)

                # add execution time
                time_MAXHS.append(tm)
                # calculating hamming distance between model result and input x
                hamming_distance = (n)*sd.hamming(x_s, r)
                tr.t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.t_e.append(1)
                else:
                    tr.t_e.append(0)

                # ********* WEIGHTED *********
                if weighted:

                    # trying dynamic lambda
                    w = t/n * k

                    mxs.output(n, t, x, y, a, noiseless, w)
                    r, noise, tm = mxs.call_Max_Sat(n)
                    # add execution time
                    time_MAXHS_W.append(tm)
                    # calculating hamming distance between model result and input x
                    hamming_distance = (n) * sd.hamming(x_s, r)
                    tr.t_h_w.append(hamming_distance)

                    # there's an error?
                    if hamming_distance > 0:
                        tr.t_e_w.append(1)
                    else:
                        tr.t_e_w.append(0)

                # *************LP_RELAX*************
                r_lp_i, tm = lp.solve(y, a, n, noiseless)

                # add execution time
                time_LP.append(tm)

                # ****CAST****

                r_lp = [int(i) for i in r_lp_i[:n]]

                # calculating hamming distance between model result and input x
                hamming_distance = (n)*sd.hamming(x_s, r_lp)
                tr.lp_t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.lp_t_e.append(1)
                else:
                    tr.lp_t_e.append(0)

                # ****CEIL****

                r_lp = [round(i) for i in r_lp_i[:n]]

                # calculating hamming distance between model result and input x
                hamming_distance = (n) * sd.hamming(x_s, r_lp)
                tr.lp_t_h_c.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.lp_t_e_c.append(1)
                else:
                    tr.lp_t_e_c.append(0)

        for tr in nw_trials:
            # MAX_HS
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))
            if(weighted):
                tr.E_W.append(np.mean(tr.t_h_w))
                tr.P_W.append(1 - np.mean(tr.t_e_w))
            # LP_RELAX
            tr.lp_E.append(np.mean(tr.lp_t_h))
            tr.lp_P.append(1 - np.mean(tr.lp_t_e))
            # LP_RELAX
            tr.lp_E_C.append(np.mean(tr.lp_t_h_c))
            tr.lp_P_C.append(1 - np.mean(tr.lp_t_e_c))

    X = []

    t_lp = np.mean(time_LP)
    t_maxhs = np.mean(time_MAXHS)

    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    for tr in nw_trials:
        values = [[tr.P[i], tr.lp_P_C[i]] for i in range(len(tr.P))]
        data = pd.DataFrame(values, T, columns=["MAX-SAT", "LP"])

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        plt.plot(T, tr.P, "bo")
        plt.plot(T, tr.lp_P_C, "yx")
        plt.plot(X, tr.P, "r", label = "k * log2(n / k)", linewidth=2.5)
        plt.title("Error trend of Max_Sat applied to Group Testing with  e = " + str(n) + " k = " + str(k))
        plt.xlabel("Number of tests m")
        plt.ylabel("Probability of success")
        plt.legend(loc="lower right")
        plt.savefig("PS, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()

        values = [[tr.E[i], tr.lp_E_C[i]] for i in range(len(tr.P))]
        data = pd.DataFrame(values, T, columns=["MAX-SAT", "LP"])

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        plt.plot(T, tr.E, "bo")
        plt.plot(T, tr.lp_E_C, "yx")
        plt.plot(X, tr.E, "r", label = "k * log2(n / k)", linewidth=2.5)
        plt.title("Error trend of Max_Sat applied to Group Testing with  e = " + str(n) + " k = " + str(k))
        plt.xlabel("Number of tests m")
        plt.ylabel("Error (Hamming Distance)")
        plt.legend(loc = "upper right")
        plt.savefig("HD, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()




main()