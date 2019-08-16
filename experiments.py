import Max_Sat_interface as mxs
import group_testing_function as gtf
import numpy as np
import math as m

OUT_NAME = "output_file"

K = 100

# number of element
N = [100, 150, 200, 250, 400, 500, 700, 800, 1000]

# probability of an element to be faulty
p = 0.03

# noise probability
d = 0.05

# if experiments are noiseless or noisy
noiseless = True

# optimized noise
fixed_noise = True

# function that generate constant density of tests
def generate_linear_test(n):

    tests = [1]

    i = 0

    while tests[i] < n:
        tests.append(int(2 * tests[i] - tests[i - 1] + int(np.log10(n))))

        i += 1

# function that generate tests more dense near the bound
def generate_dense_test_near_bound(n, k):

    tests = [1, int(n / (5 * np.log2(n)) + np.log(n))]

    # generate T
    i = 1

    # until we reach the condition of success have more dense trials
    while tests[i] < k * np.log2(n / k):
        if tests[i] - tests[i - 1] > np.ceil(m.log(n, K)):
            tests.append(int(2 * tests[i] - tests[i - 1] - np.ceil(m.log(n, K))))
        else:
            tests.append(int(tests[i] + np.ceil(m.log(n, K))))
        i += 1

    # dense trials around k*log2(n/k)
    while tests[i] < n:
        tests.append(int(2 * tests[i] - tests[i - 1] + int(np.log10(n))))

        i += 1




def main():

    t_prime = [100, 99, 95, 92, 90, 87, 80, 75, 70, 60, 40, 30, 25, 20, 19, 17, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.8, 1.5, 1.3, 1.1, 1]
    t_prime = [1/t for t in t_prime]

    T_CPU = []

    ME = []
    MP = []

    for n in N:

        print(n)

        # generate input
        x, k = gtf.generate_input(n, p)

        # avoid k = 0, trivial example
        while k == 0:
            x, k = gtf.generate_input(n, 5 / n)

        x_s = [1 if i in x else 0 for i in range(1, n + 1)]

        # pooled measurement bernoulli  belongings
        q = (np.log(2) / k)

        # opt_noise from theoretical bound
        opt_noise = d/(1 - d) + (1 - p)*(1 - p**((n - 1)*q))

        T = [int(n*t) for t in t_prime]

        T_cpu = []

        E = []
        P = []

        for t in T:

            t_cpu = []
            h = []
            e = []

            for i in range(100):

                a = gtf.generate_pool_matrix(n, k, t)

                y = gtf.get_results(t, a, x, noiseless, d)
                # *************MAX_HS*************
                mxs.output(n, t, x, y, a, noiseless, t)
                r, noise, ti_i = mxs.call_Max_Sat(n)

                t_cpu.append(ti_i)

                # calculating hamming distance between model result and input
                vs = [[r[j], x_s[j]] for j in range(len(r))]
                hamming_distance = sum([1 if vs_i[0] != vs_i[1] else 0 for vs_i in vs])
                h.append(hamming_distance)

                # there's an error?
                if hamming_distance > 0:
                    e.append(1)
                else:
                    e.append(0)

            E.append(np.mean(h))
            P.append(1 - np.mean(e))
            T_cpu.append(np.mean(t_cpu))

        ME.append(E)
        MP.append(P)
        T_CPU.append(T_cpu)

    with open(OUT_NAME, "w") as output_file:

        output_string = ''

        output_string += str(N) + "\n"

        output_string += str(t_prime) + "\n"

        output_string += str(ME) + "\n" + str(MP) + "\n"

        output_string += str(T_CPU) + "\n"

        output_file.write(output_string)


main()
