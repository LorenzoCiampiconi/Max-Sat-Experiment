import subprocess as sbp

#need to modify MAX_HS_DIR with your maxhs dir to use this
MAX_HS_DIR = r"../MaxHS-3.0"
COMMAND = "maxhs"
fixed_header = "c\nc comments Weighted Max-SAT\nc\np wcnf "
hard_weight = 10000
soft_weight = 1
input_for_max_HS = "/max_sat_input"
output_data_set = "data set"
SEP1 = "nv"
SEP2 = "nc"


#function that calls max_HS by an output file and parse the output
def call_Max_Sat(n):

    #get the output
    output_string = str(sbp.run([COMMAND , MAX_HS_DIR + input_for_max_HS], stdout=sbp.PIPE).stdout)

    #get the interesting output,
    output_string = (output_string.split(SEP1)[1]).split(SEP2)[0]
    model_string = (output_string[0:len(output_string) - 1]).split(" ")[1:]

    #parse the model into integer
    model = list(map(int, model_string))
    noise = model[n:]
    model = model[:n]
    model = list(map(lambda x : 1 if x>0 else 0,model))


    return model, noise

# function to generate file to be read by Max_HS and file of data_set
def output(n, t, x, y, a, noiseless, noise_weight):
    if noiseless:
        m = n
    else:
        m = n + t
    max_HS_input = fixed_header + " " + str(m)
    hard_clauses_string = ''
    soft_clauses_string = ''
    neg = []
    nc = 0

    #building hard constraint input to max_HS input
    if noiseless:
        for i in range(t):
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
                if 1 in a[i]:
                    for j in range(n):
                        if a[i][j] == 1 and j not in neg:
                            nc = nc + 1
                            neg.append((j))
                            local_hard = str(hard_weight) + " -" + str(j + 1) +" 0\n"
                            hard_clauses_string += local_hard
    #noisy settings
    else:
        for i in range(t):
            if y[i] == 1:
                if 1 in a[i]:
                    local_hard = str(hard_weight) + " "
                    nc = nc + 1
                    for j in range(n):
                        if a[i][j] == 1:
                            local_hard += str(j + 1) + " "
                    local_hard += (str(n + i +1))
                    local_hard += " 0\n"
                    soft_clauses_string += str(noise_weight) + " -" + str(n + i + 1) + " 0\n"
                    hard_clauses_string += local_hard
            else:
                if 1 in a[i]:
                    for j in range(n):
                        if a[i][j] == 1:
                            nc = nc + 1
                            local_hard = str(hard_weight) + " -" + str(j + 1) + " " + str(n + i + 1) +" 0\n"
                            hard_clauses_string += local_hard
                    soft_clauses_string += str(noise_weight)  + " -" + str(n + i + 1) + " 0\n"


    # add soft constraint to ensure minimum number of item faulty to max_HS input
    for j in range(n):
        if j not in neg:
            nc += 1
            soft_clauses_string += str(soft_weight) + " -" + str(j + 1) + " 0\n"

    max_HS_input += " " + str(nc) + " " + str(hard_weight) + "\n"

    max_HS_input += hard_clauses_string + soft_clauses_string


    with open(MAX_HS_DIR + input_for_max_HS, "w") as output_file:
        output_file.write(max_HS_input)

    return


