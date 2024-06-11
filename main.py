def newton_raphson(cdf_file_path):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import math


    txt_path = cdf_file_path
    # Replace 'your_file.txt' with the actual path to your text file
    file_path = txt_path

    # Read the text file into a pandas DataFrame
    raw_data = pd.read_table(file_path, sep='\t')

    # This for loop finds the start and end indexes of the bus data.
    # Moreover, it adds blank spaces before the '-' because of there are some contiguous data such as 1000.00-1000.00
    for i in range(len(raw_data)):
        if raw_data.iloc[i, 0].startswith("BUS DATA FOLLOWS"):
            bus_data_start_index = i + 1
        elif raw_data.iloc[i, 0].startswith("-999"):
            bus_data_end_index = i - 1
            break
        raw_data.iloc[i, 0] = raw_data.iloc[i, 0][:128].replace("-", " -")

    # This code block constructs the first row of the bus data matrix
    # There are different type of bus names such as 'Kanawha   V1' and 'Beaver Ck V1'
    # Directly splitting the rows creates arrays with different lengths.
    # Therefore, bus name is extracted from the table
    i = bus_data_start_index
    bus_number = raw_data.iloc[i, 0][0:4].split()
    numerical_data = raw_data.iloc[i, 0][18:].split()
    bus_data_raw = np.array(bus_number + numerical_data)
    i = i + 1

    # This while loop constructs the rest of the bus data matrix
    while i <= bus_data_end_index:
        bus_number = raw_data.iloc[i, 0][0:4].split()
        numerical_data = raw_data.iloc[i, 0][18:].split()

        new_row = np.array(bus_number + numerical_data)
        bus_data_raw = np.vstack((bus_data_raw, new_row))
        i = i + 1

    # This for loop finds the start and end indexes of the branch data.
    # The loop starts with bus_data_end_index+2 since bus_data_end_index+1 equals to "-999" and it conflicts with elif block
    for i in range(bus_data_end_index + 2, len(raw_data)):
        if raw_data.iloc[i, 0].startswith("BRANCH DATA FOLLOWS"):
            branch_data_start_index = i + 1
        elif raw_data.iloc[i, 0].startswith("-999"):
            branch_data_end_index = i - 1
            break

    # This code block constructs the first row of the branch data matrix.
    # There are missing values in the branch data matrix between 72 and 75 indexes which are unnecessary.
    # Therefore, data between 72 and 75 are extracted
    # There are some contiguous data such as 0.90431.10435 at 97 index.
    # Therefore, the data is split at 97th index manually
    i = branch_data_start_index
    branch_data_first_part = raw_data.iloc[i, 0][:72].split()
    branch_data_second_part = raw_data.iloc[i, 0][75:97].split()
    branch_data_third_part = raw_data.iloc[i, 0][97:].split()
    branch_data_raw = np.array(branch_data_first_part + branch_data_second_part + branch_data_third_part)
    i = i + 1

    while i <= branch_data_end_index:
        branch_data_first_part = raw_data.iloc[i, 0][:72].split()
        branch_data_second_part = raw_data.iloc[i, 0][75:97].split()
        branch_data_third_part = raw_data.iloc[i, 0][97:].split()

        new_row = np.array(branch_data_first_part + branch_data_second_part + branch_data_third_part)
        branch_data_raw = np.vstack((branch_data_raw, new_row))
        i = i + 1

    branch_data = np.zeros((len(branch_data_raw), 8))

    for i in range(len(branch_data)):
        branch_data[i, 0] = branch_data_raw[i, 0]  # Sending End
        branch_data[i, 1] = branch_data_raw[i, 1]  # Receiving End
        branch_data[i, 2] = branch_data_raw[i, 5]  # Branch Type
        branch_data[i, 3] = branch_data_raw[i, 6]  # R
        branch_data[i, 4] = branch_data_raw[i, 7]  # X
        branch_data[i, 5] = branch_data_raw[i, 8]  # Susceptance of the Branch
        branch_data[i, 6] = branch_data_raw[i, 13]  # Tap Magnitude
        branch_data[i, 7] = branch_data_raw[i, 14]  # Tap Angle

    bus_data = np.zeros((len(bus_data_raw), 15))

    for i in range(len(bus_data)):
        bus_data[i, 0] = bus_data_raw[i, 0]  # bus_number (0)  - since bus name is extracted
        bus_data[i, 1] = bus_data_raw[i, 3]  # type (4)        - indexes are decreased by 1
        bus_data[i, 2] = bus_data_raw[i, 6]  # P load (7)
        bus_data[i, 3] = bus_data_raw[i, 7]  # Q load (8)
        bus_data[i, 4] = bus_data_raw[i, 8]  # P Gen (9)
        bus_data[i, 5] = bus_data_raw[i, 9]  # Q Gen (10)
        bus_data[i, 6] = bus_data_raw[i, 12]  # max mvar (13)
        bus_data[i, 7] = bus_data_raw[i, 13]  # min mvar (14)
        bus_data[i, 8] = bus_data_raw[i, 14]  # shunt G (15)
        bus_data[i, 9] = bus_data_raw[i, 15]  # shunt B (16)
        bus_data[i, 10] = bus_data_raw[i, 4]  # bus voltage (5)
        bus_data[i, 11] = bus_data_raw[i, 5]  # bus angle (6)
        bus_data[i, 12] = 0 # Calculated P
        bus_data[i, 13] = 0 # Calculated Q
        bus_data[i,14] = 0 # converted ?



    y_bus = np.zeros((len(bus_data), len(bus_data)), dtype=complex)

    for i in range(len(bus_data)):
        y_bus[i, i] = complex(bus_data[i, 8], bus_data[i, 9])

    bus_order = bus_data[:, 0]

    for i in range(len(branch_data)):
        b_half = complex(0, branch_data[i, 5]) / 2
        Y = 1 / complex(branch_data[i, 3], branch_data[i, 4])
        from_bus_order = np.where(bus_order == branch_data[i, 0])[0][0]
        to_bus_order = np.where(bus_order == branch_data[i, 1])[0][0]
        y_bus[from_bus_order, from_bus_order] += b_half
        y_bus[to_bus_order, to_bus_order] += b_half

        if branch_data[i, 2] == 0:
            # If there is no transformer
            y_bus[from_bus_order, to_bus_order] -= Y
            y_bus[to_bus_order, from_bus_order] -= Y

            y_bus[from_bus_order, from_bus_order] += Y
            y_bus[to_bus_order, to_bus_order] += Y

        else:
            # If there is a transformer
            angle_in_rad = math.radians(branch_data[i,7])
            real = branch_data[i,6] * math.cos(angle_in_rad)
            imaginary = branch_data[i,6] * math.sin(angle_in_rad)
            a = complex(real, imaginary)
            a_conj = a.conjugate()

            y_bus[from_bus_order, to_bus_order] -= (Y / a_conj)
            y_bus[to_bus_order, from_bus_order] -= (Y / a)

            y_bus[from_bus_order, from_bus_order] += (Y / (a * a_conj))
            y_bus[to_bus_order, to_bus_order] += Y

    # print(y_bus)


    ybus_abs = np.abs(y_bus)

    # Plot the sparsity pattern for the absolute value of ybus
    plt.figure(figsize=(8, 8))
    plt.spy(ybus_abs, markersize=5)
    plt.title('Combined Sparsity Pattern of y_bus (Absolute Value)')
    plt.xlabel('Columns (Bus Index)')
    plt.ylabel('Rows (Bus Index)')
    plt.show()

    # ############################################################
    # # POWER FLOW STARTS
    # ############################################################
    #
    g_bus = np.real(y_bus)
    b_bus = np.imag(y_bus)
    #
    slack_bus_index = np.where(bus_data[:, 1] == 3)[0][0]
    slack_bus_number = int(bus_data[slack_bus_index, 0])
    slack_bus_voltage = bus_data_raw[slack_bus_index, 4]
    slack_bus_angle = bus_data_raw[slack_bus_index, 5]
    # print(slack_bus_index, slack_bus_number, slack_bus_voltage, slack_bus_angle)

    # # Flat Start (All unknown bus voltages and angles are equal to slack bus)
    for i in range(len(bus_data)):
        bus_data[i, 11] = slack_bus_angle
        if bus_data[i, 1] != 2:
            bus_data[i, 10] = slack_bus_voltage

    # print(g_bus)


    def p_calculator():
        # Calculates P for all buses except slack bus
        calculated_P = np.zeros((len(bus_data), 2))
        for i in range(len(bus_data)):
            calculated_P[i, 0] = bus_data[i, 0]
        for i in range(len(bus_data)):
            if bus_data[i, 1] != 3:
                for k in range(len(bus_data)):
                    v_i = bus_data[i, 10]
                    v_k = bus_data[k, 10]
                    theta = math.radians(bus_data[i, 11] - bus_data[k, 11])
                    P = round(v_i * v_k * (g_bus[i,k] * math.cos(theta) + b_bus[i,k] * math.sin(theta)), 4)
                    calculated_P[i, 1] += P
                    # print(f"busnumber = {bus_data[i, 0]}, v_i = {v_i}, busnumber = {bus_data[k, 0]} v_k = {v_k}, theta = {theta} P = {P}")
                    # print(f"{v_i} * {v_k} * ({g_bus[i, k]} * {math.cos(theta)} + {b_bus[i, k]} * {math.sin(theta)})")
        for i in range(len(calculated_P)):
            if abs(calculated_P[i,1]) <= 1e-3:
                calculated_P[i, 1] = 0
        for i in range(len(bus_data)):
            for j in range(len(calculated_P)):
                if bus_data[i,0] == calculated_P[j,0]:
                    bus_data[i,12] = calculated_P[j,1]
        return calculated_P

    # print(p_calculator())


    def q_calculator():
        # Calculates Q for all buses except slack bus
        # Checks the Q limit for PV buses and convert them to PQ bus if calculated Q is not in the limit
        calculated_Q = np.zeros((len(bus_data), 2))
        for i in range(len(bus_data)):
            calculated_Q[i, 0] = bus_data[i, 0]
        for i in range(len(bus_data)):
            if bus_data[i, 1] != 3:
                for k in range(len(bus_data)):
                    v_i = bus_data[i, 10]
                    v_k = bus_data[k, 10]
                    theta = math.radians(bus_data[i, 11] - bus_data[k, 11])
                    Q = round(v_i * v_k * (g_bus[i,k] * math.sin(theta) - b_bus[i,k] * math.cos(theta)),4)
                    calculated_Q[i, 1] += Q
                    # print(f"busnumber = {bus_data[i, 0]}, v_i = {v_i}, busnumber = {bus_data[k, 0]} v_k = {v_k}, theta = {theta} Q = {Q}")
                    # print(f"{v_i} * {v_k} * ({g_bus[i, k]} * {math.sin(theta)} - {b_bus[i, k]} * {math.cos(theta)})")
            # QLimits
            if bus_data[i, 1] == 2:
                if bus_data[i,6] != 0 and calculated_Q[i, 1] >= bus_data[i,6]:
                    calculated_Q[i, 1] = bus_data[i,6]
                    bus_data[i, 1] = 0
                    bus_data[i,14] = 1
                elif bus_data[i,7] != 0 and calculated_Q[i, 1] <= bus_data[i,7]:
                    calculated_Q[i, 1] = bus_data[i,7]
                    bus_data[i, 1] = 0
        for i in range(len(calculated_Q)):
            if abs(calculated_Q[i,1]) <= 1e-3:
                calculated_Q[i, 1] = 0
        for i in range(len(bus_data)):
            for j in range(len(calculated_Q)):
                if bus_data[i,0] == calculated_Q[j,0]:
                    bus_data[i,12] = calculated_Q[j,1]
        return calculated_Q


    # print(q_calculator())


    def mismatch_calculator():
        number_of_pv_buses = np.sum(bus_data[:, 1] == 2)
        number_of_pq_buses = np.sum(bus_data[:, 1] == 0) + np.sum(bus_data[:, 1] == 1)

        mismatch_p = np.zeros((number_of_pq_buses + number_of_pv_buses, 2))
        o = 0
        for i in range(len(bus_data)):
            if bus_data[i, 1] != 3:
                mismatch_p[i - o, 0] = bus_data[i, 0]
            else:
                o += 1

        mismatch_q = np.zeros((number_of_pq_buses, 2))
        o = 0
        for i in range(len(bus_data)):
            if bus_data[i, 1] == 0 or bus_data[i, 1] == 1:
                mismatch_q[i - o, 0] = bus_data[i, 0]
            else:
                o += 1
        # P
        p_calculated = p_calculator()
        for i in range(len(bus_data)):
            if bus_data[i, 1] != 3:
                p_total = bus_data[i, 4] - bus_data[i, 2]
                delta_p = p_total - p_calculated[i,1]
                bus_data[i,12] = p_calculated[i,1]
                for j in range(len(mismatch_p)):
                    if mismatch_p[j,0] == bus_data[i, 0]:
                        mismatch_p[j,1] = delta_p
                        break

        # Q
        q_calculated = q_calculator()
        for i in range(len(bus_data)):
            if bus_data[i, 1] == 0 or bus_data[i, 1] == 1:
                q_total = bus_data[i, 5] - bus_data[i, 3]
                delta_q = q_total - q_calculated[i,1]
                bus_data[i, 13] = q_calculated[i, 1]
                for j in range(len(mismatch_q)):
                    if mismatch_q[j,0] == bus_data[i, 0]:
                        mismatch_q[j,1] = delta_q
                        break

        mismatch_vector = np.vstack((mismatch_p, mismatch_q))
        max_mismatch = np.max(np.abs(mismatch_vector[:, 1]))
        return mismatch_vector, mismatch_p, mismatch_q, q_calculated, p_calculated, max_mismatch


    def jacobian(mismatch_p, mismatch_q, q_calculated, p_calculated):
        # mismatch_vector, mismatch_p, mismatch_q, q_calculated, p_calculated = mismatch_calculator()

        # H Matrix
        matrix_H = np.zeros((len(mismatch_p), len(mismatch_p)))
        for i in range(len(matrix_H)):
            bus_number_i = mismatch_p[i, 0]
            for k in range(len(matrix_H)):
                bus_number_k = mismatch_p[k, 0]
                if bus_number_i == bus_number_k:
                    for j in range(len(q_calculated)):
                        if q_calculated[j,0] == bus_number_i:
                            q = q_calculated[j,1]
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j,10]
                            b = b_bus[j,j]
                            break
                    matrix_H[i, i] = -q - b*v_i*v_i
                else:
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j,10]
                            theta_i = bus_data[j,11]
                            index_i = j
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_k:
                            v_k = bus_data[j,10]
                            theta_k = bus_data[j,11]
                            index_k = j
                            break
                    b = b_bus[index_i,index_k]
                    g = g_bus[index_i,index_k]
                    theta_ik = math.radians(theta_i-theta_k)
                    matrix_H[i,k] = v_i * v_k * (g*math.sin(theta_ik)-b*math.cos(theta_ik))
        # print(matrix_H)
        # L Matrix
        matrix_L = np.zeros((len(mismatch_q), len(mismatch_q)))
        for i in range(len(matrix_L)):
            bus_number_i = mismatch_q[i, 0]
            for k in range(len(matrix_L)):
                bus_number_k = mismatch_q[k, 0]
                if bus_number_i == bus_number_k:
                    for j in range(len(q_calculated)):
                        if q_calculated[j,0] == bus_number_i:
                            q = q_calculated[j,1]
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j,10]
                            b = b_bus[j,j]
                            break
                    matrix_L[i,i] = q - b*v_i*v_i

                else:
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j, 10]
                            theta_i = bus_data[j, 11]
                            index_i = j
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_k:
                            v_k = bus_data[j, 10]
                            theta_k = bus_data[j, 11]
                            index_k = j
                            break
                    b = b_bus[index_i, index_k]
                    g = g_bus[index_i, index_k]
                    theta_ik = math.radians(theta_i - theta_k)
                    matrix_L[i, k] = v_i * v_k * (g * math.sin(theta_ik) - b * math.cos(theta_ik))
        # print(matrix_L)
        # N Matrix
        matrix_N = np.zeros((len(mismatch_p), len(mismatch_q)))
        for i in range(len(mismatch_p)):
            bus_number_i = mismatch_p[i, 0]
            for k in range(len(mismatch_q)):
                bus_number_k = mismatch_q[k, 0]
                if bus_number_i == bus_number_k:
                    for j in range(len(p_calculated)):
                        if p_calculated[j, 0] == bus_number_i:
                            p = p_calculated[j, 1]
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j, 10]
                            g = g_bus[j, j]
                            break
                    matrix_N[i,k] = p + g * v_i * v_i
                else:
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j, 10]
                            theta_i = bus_data[j, 11]
                            index_i = j
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_k:
                            v_k = bus_data[j, 10]
                            theta_k = bus_data[j, 11]
                            index_k = j
                            break
                    b = b_bus[index_i, index_k]
                    g = g_bus[index_i, index_k]
                    theta_ik = math.radians(theta_i - theta_k)
                    matrix_N[i, k] = v_i * v_k * (g*math.cos(theta_ik) + b*math.sin(theta_ik))
        # print(matrix_N)
        # M Matrix
        matrix_M = np.zeros((len(mismatch_q), len(mismatch_p)))
        for i in range(len(mismatch_q)):
            bus_number_i = mismatch_q[i, 0]
            for k in range(len(mismatch_p)):
                bus_number_k = mismatch_p[k, 0]
                if bus_number_i == bus_number_k:
                    for j in range(len(p_calculated)):
                        if p_calculated[j, 0] == bus_number_i:
                            p = p_calculated[j, 1]
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j, 10]
                            g = g_bus[j, j]
                            break
                    matrix_M[i,k] = p - g * v_i * v_i
                else:
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_i:
                            v_i = bus_data[j, 10]
                            theta_i = bus_data[j, 11]
                            index_i = j
                            break
                    for j in range(len(bus_data)):
                        if bus_data[j, 0] == bus_number_k:
                            v_k = bus_data[j, 10]
                            theta_k = bus_data[j, 11]
                            index_k = j
                            break
                    b = b_bus[index_i, index_k]
                    g = g_bus[index_i, index_k]
                    theta_ik = math.radians(theta_i - theta_k)
                    matrix_M[i, k] = (-1) * v_i * v_k * (g*math.cos(theta_ik) + b*math.sin(theta_ik))
        # print(matrix_M)

        upper_part = np.hstack((matrix_H, matrix_N))
        lower_part = np.hstack((matrix_M, matrix_L))
        jacobian_matrix = np.vstack((upper_part, lower_part))
        inverse_jacobian = np.linalg.inv(jacobian_matrix)
        return jacobian_matrix, inverse_jacobian


    def updater(inverse_jacobian, mismatch_vector):

        mismatch = mismatch_vector[:, 1]
        correction = np.matmul(inverse_jacobian, mismatch)

        ########################################################################
        # for i in range(len(mismatch_p)):
        #     correction[i] = np.degrees(correction[i])
        # for i in range(len(mismatch_q)):
        #     for j in range(len(bus_data)):
        #         if mismatch_q[i,0] == bus_data[j,0]:
        #             correction[i+len(mismatch_p)] *= bus_data[j,10]
        #             break
        ########################################################################

        for i in range(len(mismatch_p)):
            for j in range(len(bus_data)):
                if mismatch_p[i, 0] == bus_data[j, 0]:
                    bus_data[j,11] += np.degrees(correction[i])
                    # bus_data[j, 11] += correction[i]
                    break

        for i in range(len(mismatch_q)):
            for j in range(len(bus_data)):
                if mismatch_q[i,0] == bus_data[j,0]:
                    bus_data[j,10] += (bus_data[j,10]*correction[i+len(mismatch_p)])
                    # bus_data[j, 10] += correction[i + len(mismatch_p)]
                    break
        # print(correction)

    iteration_counter = 0
    max_mismatch = 10

    start_time = time.time()
    while max_mismatch > 0.001 and iteration_counter < 100:
        # print("----------------------------------------------------------------------------------------------------------------")
        mismatch_vector, mismatch_p, mismatch_q, q_calculated, p_calculated, max_mismatch = mismatch_calculator()
        jacob, inv_jacob = jacobian(mismatch_p, mismatch_q, q_calculated, p_calculated)
        updater(inv_jacob, mismatch_vector)
        # print("----------------------------------------------------------------------------------------------------------------")
        # print(bus_data[:, [0, 10, 11]])
        iteration_counter += 1
        # print(f"Iteration: {iteration_counter}")

    if iteration_counter == 100:
        print("No convergence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Max iteration 300")

    converted_buses = []
    for i in range(len(bus_data)):
        if bus_data[i,14] == 1:
            converted_buses.append(bus_data[i,0])
    p_loss = np.sum(p_calculated[:,1])
    q_loss = np.sum(q_calculated[:,1])

    end_time = time.time()
    solution_time = end_time - start_time
    voltage_magnitude = bus_data[:,10]
    angle = bus_data[:,11]

    print("Voltage Magnitudes: ")
    print(voltage_magnitude)
    print("Angle:")
    print(angle)
    print("Solution Time:")
    print(solution_time)
    print("Number of iteartions:")
    print(iteration_counter)
    print("P Loss:")
    print(p_loss)
    print("Q loss:")
    print(q_loss)
    print("Converted buses:")
    print(converted_buses)

    return voltage_magnitude, angle, solution_time, iteration_counter, p_loss, q_loss, converted_buses


file_path = "C:\\Users\\Emir\\PycharmProjects\\newton-raphson\\3bus.txt"

voltage_magnitude, angle, solution_time, iteration_counter, p_loss, q_loss, converted_buses = newton_raphson(file_path)
