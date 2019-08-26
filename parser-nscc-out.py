import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os.path
import Print
import pandas as pd
import seaborn as sns

DIR = "../pull-nscc/Print-750/"
N = 101
FILE_GENERAL = "output_file-"
FILE_GENERAL_CMP = "output_file_comparison-"
EXTENSION = ".out"

FILE_2 = "output_file"


def parser_comparison():

    for i in range(N):

        print(i)

        if os.path.isfile(DIR + FILE_GENERAL_CMP + str(i) + EXTENSION):

            with open(DIR + FILE_GENERAL_CMP + str(i) + EXTENSION) as out:
                data = [ast.literal_eval(line) for line in out if line.strip()]

            n = data[0]
            k = data[1]
            lambdam = data[2]
            noiseless = data[3]
            T = data[4]
            E = data[5]
            P = data[6]
            T_CPU = data[7]
            lp_E = data[8]
            lp_P = data[9]
            lp_T_CPU = data[10]

            print_phase_transition(n, k, lambdam, T, noiseless, T_CPU, lp_T_CPU)


def parser():

    for i in range(N):

        print(i)

        if os.path.isfile(DIR + FILE_GENERAL + str(i) + EXTENSION):

            with open(DIR + FILE_GENERAL + str(i) + EXTENSION) as out:
                data = [ast.literal_eval(line) for line in out if line.strip()]

            n = data[0]
            k = data[1]
            lambdam = data [2]
            T = data[3]
            E = data[4]
            P = data[5]

            E = [e/(n) for e in E]

            X = []

            for i in range(len(T)):
                X.append(k * np.log2(n / k))

            values = P
            data = pd.DataFrame(values, T, columns=["MaxSAT"])

            sns.lineplot(data=data, palette="tab10", linewidth=2.5)
            plt.plot(T, P, "bo")
            plt.plot(X, P, "r", label="Recovery Bound")
            plt.plot(T, P, "g")
            plt.title("n = " + str(n) + ", k = " + str(k), fontsize=13)
            plt.xlabel("Number of tests m", fontsize=13)
            plt.ylabel("Probability of success", fontsize=13)
            plt.legend(loc="lower right")
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            plt.savefig("PS, e = " + str(n) + " k = " + str(k) + "l = " + str(lambdam) +  ".png")
            plt.show()
            plt.close()

            values = E
            data = pd.DataFrame(values, T, columns=["MaxSAT"])

            sns.lineplot(data=data, palette="tab10", linewidth=2.5)
            plt.plot(T, E, "bo")
            plt.plot(X, E, "r", label="Recovery Bound")
            plt.plot(T, E, "g")
            plt.title("n = " + str(n) + ", k = " + str(k), fontsize=13)
            plt.xlabel("Number of tests m", fontsize=13)
            plt.ylabel("Hamming Distance", fontsize=13)
            plt.legend(loc="upper right")
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            plt.savefig("HD, n = " + str(n) + " k = " + str(k) + "l = " + str(lambdam) + ".png")
            plt.show()
            plt.close()


def print_phase_transition(n, k, lambda_w, tests, noiseless, t_maxsat, t_lp):

    bound = []

    for i in range(len(tests)):
        bound.append(k * np.log2(n / k))

    values = t_lp
    data = pd.DataFrame(values, tests, columns=["LP"])

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.plot(tests, values, "bx")
    plt.plot(bound, t_lp, "r", label="Recovery Bound", linewidth=2.5)
    plt.title("n = " + str(n) + " k = " + str(k), fontsize=13)
    plt.xlabel("Number of tests m", fontsize=13)
    plt.ylabel("Time (s)", fontsize=13)
    plt.legend(loc="upper right")
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.savefig(
        "n = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(lambda_w) + "noiseless= " + str(noiseless) + ".png")
    plt.show()

    values = [[t_lp[i], t_maxsat[i]] for i in range(len(t_maxsat))]
    data = pd.DataFrame(values, tests, columns=["LP", "MaxSAT"])

    a = [0] + t_maxsat[1:]
    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.plot(tests, t_lp, "bx")
    plt.plot(tests, t_maxsat, "yo")
    plt.plot(bound, a, "r", label="Recovery Bound", linewidth=2.5)
    plt.title("n = " + str(n) + " k = " + str(k), fontsize=13)
    plt.xlabel("Number of tests m", fontsize=13)
    plt.ylabel("Time (s)", fontsize=13)
    plt.legend(loc="upper right")
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.savefig(" MAXTime PS, e = " + str(n) + " k = " + str(k) + "noisy_weight = " + str(lambda_w) + "noiseless= " + str(noiseless) + ".png")
    plt.show()


parser_comparison()
