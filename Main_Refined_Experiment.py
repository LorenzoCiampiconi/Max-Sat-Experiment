import Max_Sat_interface as mxs
import group_testing_function as gtf
import numpy as np
import operator
import matplotlib.pyplot as plt


n = 100

def main():
    x, k = gtf.generate_input(n, 5/n)


    #avoid k = 0, trivial example
    while k == 0:
        x, k = gtf.generate_input(n, 5 / n)

    T = [1, int(n / (k * np.log(n)))]

    # generate T
    i = 1
    # until we reach the condition of success
    while T[i] < k * np.log2(n / k):
        if T[i] - T[i - 1] > 1:
            T.append(int(2 * T[i] - T[i - 1] - 1))
        else:
            T.append(int(T[i] + 1))
        i += 1;


    # linear progress until n
    while T[i] < n:
        T.append(int(T[i] + n / 10))
        i += 1

    p = 0

    #T = [1, 10, 30]

    E = []
    for t in T:
        print(t)
        d_es = []
        for i in range(200):
            a = gtf.generate_pool_matrix(n, k, t)
            y = gtf.get_results(t, a, x)
            mxs.output(n, k, x, y, a)
            r = mxs.call_Max_Sat()

            #calculating hamming distance between model result and input x

            distance_error = sum(np.abs(list(map(operator.sub, x, y))))

            d_es.append(distance_error)
        E.append(np.mean(d_es))

    plt.plot(T)
    plt.plot(E)
    plt.show()
    p = 0




main()