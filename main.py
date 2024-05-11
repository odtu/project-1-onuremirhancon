import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

start_time = time.time()
txt_path = "C:\\Users\\Emir\\PycharmProjects\\newton-raphson\\ieee300cdf.txt"
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

bus_data = np.zeros((len(bus_data_raw), 8))

for i in range(len(bus_data)):
    bus_data[i, 0] = bus_data_raw[i, 0]  # bus_number (0)  - since bus name is extracted
    bus_data[i, 1] = bus_data_raw[i, 3]  # type (4)        - indexes are decreased by 1
    bus_data[i, 2] = bus_data_raw[i, 6]  # P load (7)
    bus_data[i, 3] = bus_data_raw[i, 7]  # Q load (8)
    bus_data[i, 4] = bus_data_raw[i, 12]  # max mvar (13)
    bus_data[i, 5] = bus_data_raw[i, 13]  # min mvar (14)
    bus_data[i, 6] = bus_data_raw[i, 14]  # shunt G (15)
    bus_data[i, 7] = bus_data_raw[i, 15]  # shunt B (16)

ybus = np.zeros((len(bus_data), len(bus_data)), dtype=complex)

for i in range(len(bus_data)):
    ybus[i, i] = complex(bus_data[i, 6], bus_data[i, 7])

bus_order = bus_data[:, 0]

for i in range(len(branch_data)):
    b_half = complex(0, branch_data[i, 5]) / 2
    Y = 1 / complex(branch_data[i, 3], branch_data[i, 4])
    from_bus_order = np.where(bus_order == branch_data[i, 0])[0][0]
    to_bus_order = np.where(bus_order == branch_data[i, 1])[0][0]
    ybus[from_bus_order, from_bus_order] += b_half
    ybus[to_bus_order, to_bus_order] += b_half

    if branch_data[i, 2] == 0:
        # If there is no transformer
        ybus[from_bus_order, to_bus_order] = ybus[from_bus_order, to_bus_order] - Y
        ybus[to_bus_order, from_bus_order] = ybus[from_bus_order, to_bus_order]

        ybus[from_bus_order, from_bus_order] = ybus[from_bus_order, from_bus_order] + Y
        ybus[to_bus_order, to_bus_order] = ybus[to_bus_order, to_bus_order] + Y

    else:
        # If there is a transformer
        a = complex(branch_data[i, 6], branch_data[i, 7])
        a_conj = a.conjugate()

        ybus[from_bus_order, to_bus_order] -= (Y / a_conj)
        ybus[to_bus_order, from_bus_order] -= (Y / a)

        ybus[from_bus_order, from_bus_order] += (Y / (a * a_conj))
        ybus[to_bus_order, to_bus_order] += Y

# ybus_abs = np.abs(ybus)
#
# # Plot the sparsity pattern for the absolute value of ybus
# plt.figure(figsize=(8, 8))
# plt.spy(ybus_abs, markersize=1)
# plt.title('Combined Sparsity Pattern of ybus (Absolute Value)')
# plt.xlabel('Columns (Bus Index)')
# plt.ylabel('Rows (Bus Index)')
# plt.show()
# #

g_bus = np.real(ybus)
b_bus = np.imag(ybus)

slack_bus_index = np.where(bus_data[:, 1] == 3)[0][0]
slack_bus_number = int(bus_data[slack_bus_index, 0])
print(slack_bus_index)
print(slack_bus_number)
