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


    noise_probability = 0.05

    T = [75]#number of tests


    trials = []

    noise_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1, 1.05, 1.1]

    for i in T:
        trials.append(trial(i))

    #for every t number of tests
    for w in noise_weights:
        print(w)

        for tr in trials:
            tr.t_e = [] #blank temporary
            tr.t_h = [] #blank temporary

        for i in range(100):
            print(i)

            for tr in trials:
                a = gtf.generate_pool_matrix(n, k, tr.var)
                y = gtf.get_results(tr.var, a, x, noiseless, noise_probability)
                mxs.output(n, tr.var , x, y, a, noiseless, w)

                r, noise = mxs.call_Max_Sat(n)

                #calculating hamming distance between model result and input x

                hamming_distance = (n)*sd.hamming(x,r)
                tr.t_h.append(hamming_distance)

                #there's an error?
                if hamming_distance > 0:
                    tr.t_e.append(1)
                else:
                    tr.t_e.append(0)

        for tr in trials:
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))

    lp_t_h = []
    lp_t_e=[]

    #********* LP - RELAXATION *********
    for i in range(100):
        print(i)
        a = gtf.generate_pool_matrix(n, k, tr.var)

        y = gtf.get_results(tr.var, a, x, noiseless, noise_probability)

        r = lp.solve(y, a, n, noiseless)[:n]

        hamming_distance = (n) * sd.hamming(x, r)
        lp_t_h.append(hamming_distance)

        # there's an error?
        if hamming_distance > 0:
            lp_t_e.append(1)
        else:
            lp_t_e.append(0)

    E = np.mean(lp_t_h)
    P = 1 - np.mean(lp_t_e)

    LP_E = [E for i in noise_weights]
    LP_P = [P for i in noise_weights]

    X = []

    for tr in trials:

        fig = plt.figure()
        plt.plot(noise_weights, tr.P, "bo", label = "max-sat")
        plt.plot(noise_weights, tr.P, "g")
        plt.plot(noise_weights, LP_P, "y", label = "LP")
        plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
        plt.xlabel("weight to the noise")
        plt.ylabel("Probability of success")
        plt.savefig("PS, n = " + str(n) + " k = " + str(k) + "nT = " + str(tr.var) + ".png")
        plt.show()

        plt.plot(noise_weights, tr.E, "bo", label = "max-sat")
        plt.plot(noise_weights, tr.E, "g")
        plt.plot(noise_weights, LP_E, "y", label="LP")
        plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
        plt.xlabel("Number of tests t")
        plt.ylabel("Error (Hamming Distance)")
        plt.savefig("HD, n = " + str(n) + " k = " + str(k) + "nT = " + str(tr.var) + ".png")
        plt.show()



main()