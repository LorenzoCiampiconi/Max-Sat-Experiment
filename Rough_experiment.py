import numpy as np
import numpy.random as nrndm

output_path = "C:/Users/Lorenzo/Documents/Polimi/Singapore/Research/"
output_for_max_HS = "max_sat problem"
output_data_set = "data set"
hard_weight = 1000
soft_weight = 1
fixed_header = "c\nc comments Weighted Max-SAT\nc\np wcnf "


def main():
    # number of item
    n = 250
    # number of test how to choose it?
    k = int(n ** (1/2.0))
    # input vector
    x = [0] * n

    # generate a sparse input to recover
    for i in range(n):
        x[i] = nrndm.choice(range(0, 2), p=[0.99, 0.01])

    print("sparse input generated")

    # define matrix of test and generate it
    # test results
    y = [0] * k
    a = generate_pool_matrix(n, k)
    # get result of test
    for i in range(k):
        if 1 in a[i] * x:
            y[i] = 1

    print("result has been got")

    output(n, k, x, y, a)


# function to generate file to be read by Max_HS and file of data_set
def output(n, k, x, y, a):
    data_set = "x = "
    count = x.count(1)
    positive_tests = y.count(1)

    for i in range(n):
        data_set += str(x[i]) + " (" + str(i + 1) + "), "
    data_set += "\n"
    data_set += "faulty occurency generated = " + str(count) + "\n"
    data_set += "y = "
    for i in range(k):
        data_set += str(y[i]) + " (" + str(i + 1) + "), "
    data_set += "\n"
    data_set += "positive tests = " + str(positive_tests) + "\n"
    data_set += "A =\n"
    for i in range(k):
        for j in range(n):
            data_set += str(a[i][j]) + " "
        data_set += "\n"

    header_to_maxHS = fixed_header + " " + str(n)
    hard_clauses_string = ''
    soft_clauses_string = ''
    neg = []
    nc = 0
    for i in range(k):
        if y[i] == 1:
            if 1 in a[i]:
                local_hard = str(hard_weight) + " "
                nc = nc + 1
                for j in range(n):
                    if a[i][j] == 1:
                        local_hard += str(j + 1) + " "
                local_hard += " 0\n"
                hard_clauses_string += local_hard
        else:
            for j in range(n):
                if 1 in a[i]:
                    if a[i][j] == 1 and j not in neg:
                        nc = nc + 1
                        neg.append((j))
                        local_hard = str(hard_weight) + " -" + str(j + 1) +" 0\n"
                        hard_clauses_string += local_hard

    # add soft_weight to ensure minimum number of item faulty
    for j in range(n):
        if j not in neg:
            nc += 1
            soft_clauses_string += str(soft_weight) + " -" + str(j + 1) + " 0\n"

    header_to_maxHS += " " + str(nc) + " " + str(hard_weight) + "\n"

    header_to_maxHS += hard_clauses_string + soft_clauses_string

    print("writing output")
    write_output(header_to_maxHS, data_set)

    return

def write_output(max_hs, data_set):

    #open the file to output the input for max_sat
    with open(output_path + output_for_max_HS, "w") as output_file:
        output_file.write(max_hs)

    with open(output_path + output_data_set, "w") as output_file:
        output_file.write(data_set)


# function to generate pool matrix
def generate_pool_matrix(n, k):
    a = np.array(np.zeros((k, n), dtype= int))
    for i in range(k):
        print(i)
        for j in range(n):
            a[i][j] = nrndm.choice(range(0, 2), p=[1 - 2/ k,  2/ k])
    print("matrix generated, getting result")
    return a


main()
