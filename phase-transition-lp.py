import Max_Sat_interface as mxs
import general_lp_interface as lp
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

"n = 100"
weighted = True
number_of_trial = 100
dir = "results-to-plot/"
OUT_NAME = "output_file"


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


def main(noiseless, n, p):
    k = round(n * p)

    x, k_g = gtf.generate_input(n, p)

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
        noise_weight = [0.9]

    for i in noise_weight:
        nw_trials.append(trial(i))

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

        tr.lp_t_e = []
        tr.t_e = []

        for i in range(number_of_trial):

            a = gtf.generate_pool_matrix(n, k, t)

            y = gtf.get_results(t, a, x, noiseless, noise_probability)

            # *************MAX_HS*************

            mxs.output(n, t, x, y, a, noiseless, tr.var)

            r, noise, tm = mxs.call_Max_Sat(n)

            # add execution time
            time_maxhs.append(tm)
            # calculating hamming distance between model result and input x
            hamming_distance = (n)*sd.hamming(x_s, r)
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
            hamming_distance = n*sd.hamming(x_s, r_lp)
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

        mean_time_lp.append(np.mean(time_lp))
        mean_time_maxhs.append(np.mean(time_maxhs))
        success_lp.append(1 - np.mean(tr.lp_t_e))
        success_maxhs.append(1 - np.mean(tr.t_e))

    X = []

    X_i = []

    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    values = mean_time_lp
    data = pd.DataFrame(values, T, columns=["LP"])

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.plot(T, values, "bx")
    plt.plot(X, mean_time_lp, "r", label="Recovery Bound", linewidth=2.5)
    plt.title("n = " + str(n) + " k = " + str(k), fontsize=13)
    plt.xlabel("Number of tests m", fontsize=13)
    plt.ylabel("Time (s)", fontsize=13)
    plt.legend(loc="upper right")
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.savefig("n = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + "noiseless= " + str(noiseless) + ".png")
    plt.show()

    values = [[mean_time_lp[i], mean_time_maxhs[i]] for i in range(len(mean_time_maxhs))]
    data = pd.DataFrame(values, T, columns=["LP", "MaxSAT"])

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
    plt.savefig(" MAXTime PS, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + "noiseless= " + str(noiseless) + ".png")
    plt.show()

    '''
    values = [[success_lp[i], success_maxhs[i]] for i in range(len(mean_time_maxhs))]
    data = pd.DataFrame(values, T, columns=["LP", "MAX-sat"])
    data = data.rolling(3).mean()

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.plot(X, success_lp, "r", label="Recovery Bound")
    plt.title("Accuracy trend of Max_Sat and LP e = " + str(n) + " k = " + str(k))
    plt.xlabel("Number of tests t")
    plt.ylabel("Probability of success")
    plt.legend(loc="lower right")
    plt.savefig("AccuracyPS, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
    plt.show()
    '''

    '''
    plt.plot(T, mean_time_lp, "bx", label = "Time of LP")
    plt.plot(T, mean_time_lp, "b")
    plt.plot(T, mean_time_maxhs, "r+", label = "Time of MAX-HS")
    plt.plot(T, mean_time_maxhs, "r")
    plt.title("Time trend of Max_Sat and LP n = " + str(n) + " k = " + str(k))
    plt.xlabel("Number of tests t")
    plt.ylabel("Time in seconds")
    plt.legend(loc="lower right")
    plt.savefig("Time PS, n = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
    plt.show()

    plt.plot(T, success_lp, "bx", label="LP")
    plt.plot(T, success_lp, "b")
    plt.plot(T, success_maxhs, "r+", label="MAX-HS")
    plt.plot(T, success_maxhs, "r")
    plt.title("Accuracy trend of Max_Sat and LP n = " + str(n) + " k = " + str(k))
    plt.xlabel("Number of tests t")
    plt.ylabel("Probability of success")
    plt.legend(loc="lower right")
    plt.savefig("AccuracyPS, n = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
    plt.show()
    '''

    with open(dir + OUT_NAME + "-" + str(noiseless) + "-n = " + str(n) + "-k = " + str(k), "w+") as output_file:

        output_string = ''

        output_string += str(n) + "\n" + str(k) + "\n"

        output_string += str(T) + "\n"

        output_string += str(mean_time_lp) + "\n" + str(mean_time_maxhs) + "\n"

        output_file.write(output_string)


main(True, 250, 10/250)
main(True, 500, 10/500)
main(True, 750, 10/750)
main(True, 1000, 10/1000)

main(True, 250, 0.01)
main(True, 500, 0.01)
main(True, 750, 0.01)
main(True, 1000, 0.01)

main(True, 250, 0.03)
main(True, 500, 0.03)
main(True, 750, 0.03)
main(True, 1000, 0.03)
