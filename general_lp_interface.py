import cplex
import numpy as np

alpha = 1


def solve(y, A, nvar, noiseless):
    # get float for CPLEX
    A = [[float(a_ij) for a_ij in a_i] for a_i in A]
    y = [float(y_i) for y_i in y]

    a_i_n = []
    a_j_n = []

    Y_i = [item for item in y if item == 1]

    ones = [1.0 for i in range(len(y))]
    N = np.diag(ones)
    minus_N = np.diag(np.multiply(-1, ones))

    for i in range(len(y)):
        if y[i] == 1.0:
            a_i_n.append(A[i] + list(N[i]))
        else:
            a_j_n.append(A[i] +list(minus_N[i]))


    # variables name
    r_x = ["x" + str(i) for i in range(1, nvar + 1)] + ([] if noiseless else (["n" + str(i) for i in range(1, len(y) + 1)]))

    # objective function coefficient
    objective = [1 for i in range(nvar)] + ([] if noiseless else [alpha for i in range(len(y))])

    # upper bounds x<=1
    upper_bounds = [1 for i in range(nvar)] + ([] if noiseless else [1 for i in range(len(y))])

    # lower bounds x>= 0
    lower_bounds = [0 for i in range(nvar)] + ([] if noiseless else [0 for i in range(len(y))])

    # constraints
    m_constraints = [([r_x, a[:nvar]] if noiseless else [r_x, a]) for a in a_i_n] + \
                    [([r_x, a[:nvar]] if noiseless else  [r_x, a]) for a in a_j_n]

    #sense of the constraints
    m_senses = ["G" for a in a_i_n] + ["E" for a in a_j_n]

    #results
    m_rhs = [y_i for y_i in Y_i] + [0 for a in a_j_n]

    #name of the constraints
    m_names = ["c_" + str(i) for i in range(len(m_constraints))]

    # Create instance of the problem
    problem = cplex.Cplex()

    problem.objective.set_sense(problem.objective.sense.minimize)

    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=r_x)

    problem.linear_constraints.add(lin_expr=m_constraints,
                                   senses=m_senses,
                                   rhs=m_rhs,
                                   names=m_names)

    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)

    # Call solve
    problem.solve()

    # get the solution
    recovered_x = problem.solution.get_values()

    return recovered_x
