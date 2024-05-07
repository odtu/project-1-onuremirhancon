import pandas as pd
import numpy as np

# Replace 'your_file.txt' with the actual path to your text file
file_path = 'ieee300cdf.txt'

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
    raw_data.iloc[i,0] = raw_data.iloc[i,0][:128].replace("-", " -")

# This code block constructs the first row of the bus data matrix
# There are different type of bus names such as 'Kanawha   V1' and 'Beaver Ck V1'
# Directly splitting the rows creates arrays with different lengths.
# Therefore, bus name is extracted from the table
i = bus_data_start_index
bus_number = raw_data.iloc[i, 0][0:4].split()
numerical_data = raw_data.iloc[i, 0][18:].split()
bus_data = np.array(bus_number + numerical_data)
i = i+1

# This while loop constructs the rest of the bus data matrix
while i <= bus_data_end_index:
    bus_number = raw_data.iloc[i, 0][0:4].split()
    numerical_data = raw_data.iloc[i, 0][18:].split()

    new_row = np.array(bus_number + numerical_data)
    bus_data = np.vstack((bus_data, new_row))
    i = i+1

# This for loop finds the start and end indexes of the branch data.
# The loop starts with bus_data_end_index+2 since bus_data_end_index+1 equals to "-999" and it conflicts with elif block
for i in range(bus_data_end_index+2,len(raw_data)):
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
branch_data = np.array(branch_data_first_part + branch_data_second_part + branch_data_third_part)
i = i+1

while i <= branch_data_end_index:
    branch_data_first_part = raw_data.iloc[i, 0][:72].split()
    branch_data_second_part = raw_data.iloc[i, 0][75:97].split()
    branch_data_third_part = raw_data.iloc[i, 0][97:].split()

    new_row = np.array(branch_data_first_part + branch_data_second_part + branch_data_third_part)
    branch_data = np.vstack((branch_data, new_row))
    i = i+1


branch_data_for_ybus = np.zeros((len(branch_data), 7), dtype=complex)

for i in range(len(branch_data)):
    #branch_data_for_ybus[i,0] = branch_data[i,0]  # Sending End
    #branch_data_for_ybus[i,1] = branch_data[i,1]  # Receiving End
    #branch_data_for_ybus[i,2] = branch_data[i,5]  # Branch Type
    branch_data_for_ybus[i,3] = complex(float(branch_data[1,6]),float(branch_data[1,7]))  # Series Admittance of the Branch
    print(branch_data_for_ybus[i,3])
    #branch_data_for_ybus[i,4] = branch_data[i,8]  # Susceptance of the Branch
    #branch_data_for_ybus[i,5] = branch_data[i,13]  # Tap Magnitude
    #branch_data_for_ybus[i,6] = branch_data[i,14]  # Tap Angle