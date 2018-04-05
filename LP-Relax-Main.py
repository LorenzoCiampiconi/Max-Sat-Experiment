import general_lp_interface as lp
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt


n = 100


noiseless = False
class trial:
    def __init__(self, var):
        self.E = [] #on number of tests
        self.P = []
        self.t_e = [] #temporary error
        self.t_h = [] #temporary hamming distance
        self.var = var

def main():
    x, k = gtf.generate_input(n, 5/n)


    #avoid k = 0, trivial example
    while k != 4:
        x, k = gtf.generate_input(n, 5 / n)

    noise_probability = 0.0

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


    #dense trials around k* log2(n/k)
    while T[i] < n:
        T.append(int(2 * T[i] - T[i - 1] + 1))

        i += 1;

    #T = [1, 10, 30]

    nw_trials = []

    noise_weight = [0.0]

    for i in noise_weight:
        nw_trials.append(trial(i))

    #for every t number of tests
    for t in T:
        print(t)

        for tr in nw_trials:
            tr.t_e = [] #blank temporary
            tr.t_h = [] #blank temporary

        for i in range(5):
            a = gtf.generate_pool_matrix(n, k, t)

            for tr in nw_trials:
                y = gtf.get_results(t, a, x, noiseless, noise_probability)

                r = lp.solve(y, a, n, noiseless)[:n]

                #calculating hamming distance between model result and input x

                hamming_distance = (n)*sd.hamming(x,r)
                tr.t_h.append(hamming_distance)

                #there's an error?
                if hamming_distance > 0:
                    tr.t_e.append(1)
                else:
                    tr.t_e.append(0)

        for tr in nw_trials:
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))

    X = []
    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    for tr in nw_trials:

        plt.plot(T, tr.P, "bo")
        plt.plot(X, tr.P, "r")
        plt.plot(T, tr.P, "g")
        plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
        plt.xlabel("Number of tests t")
        plt.ylabel("Probability of success")
        plt.savefig("PS, n = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()

        plt.plot(T, tr.E, "bo")
        plt.plot(X, tr.E, "r")
        plt.plot(T, tr.E, "g")
        plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
        plt.xlabel("Number of tests t")
        plt.ylabel("Error (Hamming Distance)")
        plt.savefig("HD, n = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()



main()