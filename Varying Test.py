import Max_Sat_interface as mxs
import group_testing_function as gtf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


# number of element
n = 150

# probability of an element to be faulty
p = 0.03


noiseless = False

class trial:
    def __init__(self, var):
        self.E = []  # on number of tests
        self.P = []
        self.t_e = []  # temporary error
        self.t_h = []  # temporary hamming distance
        self.var = var


def main():
    x, k = gtf.generate_input(n, p)

    # avoid k = 0, trivial example
    while k < 4:
        x, k = gtf.generate_input(n, 5 / n)

    print ("k chosen")
    x_s = [1 if i in x else 0 for i in range(1, n + 1)]

    noise_probability = 0.05

    '''
    T = [1]

    cost = 10

    for i in range(1, 11):
        T.append(int((n / 100) * (cost * i)))

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
        T.append(int(2 * T[i] - T[i - 1] + int(np.log10(n))))

        i += 1

    # T = [1, 10, 30]

    nw_trials = []

    noise_weight = [0.0]

    if not noiseless:
        noise_weight = [0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1, 1.05, 1.1]

    for i in noise_weight:
        nw_trials.append(trial(i))

    TI = []

    # for every t number of tests
    for t in T:
        print(t)

        ti = []

        for tr in nw_trials:
            tr.t_e = [] # blank temporary
            tr.t_h = [] # blank temporary

        for i in range(10):

            a = gtf.generate_pool_matrix(n, k, t)

            for tr in nw_trials:
                y = gtf.get_results(t, a, x, noiseless, noise_probability)

                # *************MAX_HS*************
                mxs.output(n, t, x, y, a, noiseless, tr.var)
                r, noise, ti_i = mxs.call_Max_Sat(n)

                ti.append(ti_i)


                # calculating hamming distance between model result and input
                vs = [[r[j], x_s[j]] for j in range(len(r))]
                hamming_distance = sum([1 if vs_i[0] != vs_i[1] else 0 for vs_i in vs])
                tr.t_h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    tr.t_e.append(1)
                else:
                    tr.t_e.append(0)

        for tr in nw_trials:
            # MAX_HS
            tr.E.append(np.mean(tr.t_h))
            tr.P.append(1 - np.mean(tr.t_e))

        ti = np.mean(ti)

        TI.append(ti)

        print("medium time is for t: " + str(t) + " " + str(ti))

    X = []

    test_axis = []
    lim_axis = []
    p_axis = []
    e_axis = []
    w_axis = []

    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    # printing 3D

    for tr in nw_trials:
        w_axis.append(np.asarray([tr.var] * len(T)))
        test_axis.append(np.asarray(T))
        lim_axis.append(np.asarray(X))
        p_axis.append(np.asarray(tr.P))
        e_axis.append(np.asarray(tr.E))

    w_axis = np.asarray(w_axis)
    test_axis = np.asarray(test_axis)
    p_axis = np.asarray(p_axis)
    w_axis = np.asarray(w_axis)
    e_axis = np.asarray(e_axis)

    font = {'family': 'helvet',
            'weight': 'bold',
            'size': 75}

    mpl.rc('font', **font)

    fig = plt.figure(i, figsize=(70, 90))
    ax = Axes3D(fig)
    plt.ylabel("Number of Test", fontsize=120.0, labelpad=150.0)
    plt.xlabel("Weights", fontsize=120.0, labelpad=150.0)
    ax.set_zlabel('Error (Hamming distance)', fontsize=120.0, labelpad=150.0)

    plt.title('Probability of success\n in function of number of test and lambda\n n =' + str(n) + ", k= " + str(k),
              fontsize=150.0, weight="bold")

    # Make data.
    Y = test_axis
    X = w_axis
    Z = e_axis

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    angle = 45
    ax.view_init(14, angle)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    print("HD, n = " + str(n) + " k = " + str(k) + "noisy_weight = " + ".png")
    fig.savefig("HD, n = " + str(n) + " k = " + str(k) + "noisy_weight = " + ".png")

    '''
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

        noisy_string = ("noiseless" if noiseless else "noisy")

        plt.plot(T, TI, "ro", label="medium time")
        plt.plot(X, TI, "r",  label="k * log2(n / k)")
        plt.plot(T, TI, "g")
        plt.title("Time   n = " + str(n) + " k = " + str(k) + " " + noisy_string)
        plt.xlabel("Number of tests t")
        plt.ylabel("Time")
        plt.legend(loc="upper right")
        plt.savefig("HD, n = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(tr.var) + ".png")
        plt.show()
    '''


main()