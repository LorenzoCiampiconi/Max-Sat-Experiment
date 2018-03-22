import Max_Sat_interface as mxs
import group_testing_function as gtf
import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt


n = 150

noiseless = False

def main():
    x, k = gtf.generate_input(n, 5/n)


    #avoid k = 0, trivial example
    while k != 4:
        x, k = gtf.generate_input(n, 5 / n)



    T = [1, int(n / (5* np.log2(n)) +1)]

    # generate T
    i = 1
    # until we reach the condition of success have more dense trials
    while T[i] < k * np.log2(n / k):
        if T[i] - T[i - 1] > 1:
            T.append(int(2 * T[i] - T[i - 1] - 1))
        else:
            T.append(int(T[i] + 1))
        i += 1;


    #dense trials around k* log2(n/k)
    while T[i] < n:
        T.append(int(2 * T[i] - T[i - 1] + 1))

        i += 1;

    #T = [1, 10, 30]

    E = [] #errors measured in hamming distance

    P = [] #probability of error
    #for every t number of tests
    for t in T:
        print(t)
        d_es = []
        error = []
        precisions = []
        for i in range(100):
            a = gtf.generate_pool_matrix(n, k, t)


            y = gtf.get_results(t, a, x, noiseless)
            mxs.output(n, t, x, y, a, noiseless)

            r, noise = mxs.call_Max_Sat(n)

            if t > 60:
                a = 1
            #calculating hamming distance between model result and input x

            hamming_distance = (n)*sd.hamming(x,r)
            d_es.append(hamming_distance)

            #there's an error?
            if hamming_distance > 0:
                error.append(1)
            else:
                error.append(0)



        E.append(np.mean(d_es))
        P.append(1 - np.mean(error))

    X = []
    for i in range(len(T)):
        X.append(k * np.log2(n / k))

    plt.plot(T,P, "bo")
    plt.plot(X, P, "r")
    plt.plot(T,P, "g")
    plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
    plt.xlabel("Number of tests t")
    plt.ylabel("Probability of success")
    plt.show()

    plt.plot(T, E, "bo")
    plt.plot(X, P, "r")
    plt.plot(T, E, "g")
    plt.title("Error trend of Max_Sat applied to Group Testing with  n = " + str(n) + " k = " + str(k))
    plt.xlabel("Number of tests t")
    plt.ylabel("Error (Hamming Distance)")
    plt.show()




main()