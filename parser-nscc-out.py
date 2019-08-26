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
N = 10
FILE_GENERAL = "output_file-"
EXTENSION = ".out"

FILE_2 = "output_file"


def parser_2():

    for i in range(N):

        print(i)

        if os.path.isfile(FILE_2):

            with open(FILE_2) as out:
                data = [ast.literal_eval(line) for line in out if line.strip()]

            n = data[0]
            t = data[1]
            H = data[2]
            P = data[3]
            T_CPU = data[4]

            x = [[item]*len(t) for item in n]
            y = [t]*len(n)

            Print.print(x, y, T_CPU)


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


parser()
