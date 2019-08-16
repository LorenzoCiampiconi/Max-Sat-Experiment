import Max_Sat_interface as mxs
import general_lp_interface as lp
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt

n = 100


noiseless = False

class trial:
    def __init__(self, var):
        self.E = []  # on number of tests
        self.P = []
        self.t_e = []  # temporary error
        self.t_h = []  # temporary hamming distance
        self.lp_E = []  # on number of tests
        self.lp_P = []
        self.lp_t_e = []  # temporary error
        self.lp_t_h = []  # temporary hamming distance
        self.var = var

def main():
    x, k = gtf.generate_input(n, 5/n)


    # avoid k = 0, trivial example
    while k != 4:
        x, k = gtf.generate_input(n, 5/n)



    T = [65]#number of tests

    W = 0.9

    trials = []

    E_P = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]

    for i in T:
        trials.append(trial(i))

    # for every t number of tests
    for e_p in E_P:
        print(e_p)

        for tr in trials:
            tr.t_e = [] # blank temporary
            tr.t_h = [] # blank temporary
            tr.lp_t_e = [] # blank temporary
            tr.lp_t_h = [] # blank temporary

        for i in range(100):

            for tr in trials:
                a = gtf.generate_pool_matrix(n, k, tr.var)

                y = gtf.get_results(tr.var, a, x, noiseless, e_p)

                #*************MAX_HS*************
                mxs.output(n, tr.var, x, y, a, noiseless, tr.var)

                r, noise = mxs.call_Max_Sat(n)

                #calculating hamming distance between model result and input x
                hamming_distance = (n)*sd.hamming(x, r)
                tr.t_h.append(hamming_distance)

                #there's an error?
                if hamming_distance > 0:
                    tr.t_e.append(1)
                else:
                    tr.t_e.append(0)

                # *************LP_RELAX*************
                r_lp = lp.solve(y, a, n, noiseless)[:n]
                r_lp = [int(i) for i in r_lp]

                # calculating hamming distance between model result and input x
                hamming_distance = (n)*sd.hamming(x, r_lp)
                tr.lp_t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.lp_t_e.append(1)
                else:
                    tr.lp_t_e.append(0)

        for tr in trials:
            #MAX_HS
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))
            #LP_RELAX
            tr.lp_E.append(np.mean(tr.lp_t_h))
            tr.lp_P.append(1 - np.mean(tr.lp_t_e))

    X = []

    for tr in trials:
        plt.plot(E_P, tr.P, "bo", label="Max_sat")
        plt.plot(E_P, tr.lp_P, "yo", label="LP_Relax")
        plt.plot(E_P, tr.P, "g")
        plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
        plt.xlabel("Probability of a test to be inverted")
        plt.ylabel("Probability of success")
        plt.savefig("PS, n = " + str(n) + " k = " + str(k) + "e_p = " + str(tr.var) + ".png")
        plt.show()

        plt.plot(E_P, tr.E, "bo", label="Max_sat")
        plt.plot(E_P, tr.lp_E, "yo", label="LP_Relax")
        plt.plot(E_P, tr.E, "g")
        plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
        plt.xlabel("Probability of a test to be inverted")
        plt.ylabel("Error (Hamming Distance)")
        plt.savefig("HD, n = " + str(n) + " k = " + str(k) + "e_p = " + str(tr.var) + ".png")
        plt.show()



main()