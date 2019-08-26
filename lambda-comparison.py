import Max_Sat_interface as mxs
import general_lp_interface as lp
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
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

    weight_1 = noise_probability + (k/n)*(1-(1-(k/n))**(n*(np.log(2)/k)))
    weight_2 = (np.log((1-noise_probability)/noise_probability)) / (np.log((1-(k/n))/(k/n)))

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

        for i in range(100):
            a = gtf.generate_pool_matrix(n, k, t)

            for tr in nw_trials:
                y = gtf.get_results(t, a, x, noiseless, noise_probability)

                # *************MAX_HS*************

                print("l-w")

                mxs.output(n, t, x, y, a, noiseless, 1 / weight_1)

                r, noise, tm = mxs.call_Max_Sat(n)

                # add execution time
                time_MAXHS.append(tm)
                # calculating hamming distance between model result and input x
                hamming_distance = sd.hamming(x_s, r)
                tr.t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.t_e.append(1)
                else:
                    tr.t_e.append(0)

                # *************MAX_SAT_W2*************

                print("s-w")
                mxs.output(n, t, x, y, a, noiseless, weight_2)
                r_lp, noise, tm = mxs.call_Max_Sat(n)

                # add execution time
                time_LP.append(tm)

                # ****CAST****

                #r_lp = [int(i) for i in r_lp_i[:n]]

                # calculating hamming distance between model result and input x
                hamming_distance = sd.hamming(x_s, r_lp)
                tr.lp_t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.lp_t_e.append(1)
                else:
                    tr.lp_t_e.append(0)

                # ****CEIL****

                # r_lp = [round(i) for i in r_lp_i[:n]]

                # calculating hamming distance between model result and input x
                hamming_distance = sd.hamming(x_s, r_lp)
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
        values = [[tr.lp_P_C[i], tr.P[i]] for i in range(len(tr.P))]
        data = pd.DataFrame(values, T, columns=["LP", "MAX-SAT"])

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        plt.plot(T, tr.P, "yo")
        plt.plot(T, tr.lp_P_C, "bx")
        plt.plot(X, tr.P, "r", label = "Recovery Bound", linewidth=2.5)
        plt.title("Probability of success with  e = " + str(n) + " k = " + str(k))
        plt.xlabel("Number of tests m")
        plt.ylabel("Probability of success")
        plt.legend(loc="lower right")
        plt.savefig("PS, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()

        values = [[tr.lp_E_C[i], tr.E[i]] for i in range(len(tr.P))]
        data = pd.DataFrame(values, T, columns=["LP", "MAX-SAT"])

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        plt.plot(T, tr.E, "yo")
        plt.plot(T, tr.lp_E_C, "bx")
        plt.plot(X, tr.E, "r", label = "Recovery Bound", linewidth=2.5)
        plt.title("Hamming distance trend with  e = " + str(n) + " k = " + str(k))
        plt.xlabel("Number of tests m")
        plt.ylabel("Error (Hamming Distance)")
        plt.legend(loc = "upper right")
        plt.savefig("HD, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()


main(100, 0.03, False)

