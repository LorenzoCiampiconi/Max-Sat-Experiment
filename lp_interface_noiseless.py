import cplex

alpha = 1

def solve(y, A, nvar, noiseless):

    #get float for CPLEX
    A = [[float(a_ij) for a_ij in a_i] for a_i in A]
    y = [float(y_i) for y_i in y]

    A_i = []
    A_j = []

    Y_i = [item for item in y if item == 1]

    for i in range(len(y)):
        if y[i] == 1.0:
            A_i.append((A[i]))
        else:
            A_j.append((A[i]))


    #variables name
    r_x = ["x" + str(i) for i in range(1, nvar +1)]

    #objective function coefficient
    objective = [1 for i in range(nvar)]

    #upper bounds x<=1
    upper_bounds = [1 for i in range(nvar)]

    #lower bounds x>= 0
    lower_bounds = [0 for i in range(nvar)]


    #constraints
    m_constraints = [[r_x, a] for a in A_i] + [[r_x[0:nvar], a] for a in A_j]
    m_senses= ["G" for a in A_i] + ["E" for a in A_j]
    m_rhs = [y_i for y_i in Y_i] + [0 for a in A_j]
    m_names = ["c_" + str(i) for i in range(len(m_constraints))]



    #Create instance of the problem
    problem = cplex.Cplex()

    problem.objective.set_sense(problem.objective.sense.minimize)

    problem.variables.add(obj=objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = r_x)

    problem.linear_constraints.add(lin_expr = m_constraints,
                                   senses = m_senses ,
                                   rhs = m_rhs,
                                   names = m_names)

    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)

    #Call solve
    problem.solve()

    #get the solution
    recovered_x = problem.solution.get_values()

    return recovered_x

