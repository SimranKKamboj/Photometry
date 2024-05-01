# Open the text file for reading
#with open('tno_a', 'r') as file:
 
#   lines = file.readlines()

# Initialize a list to store the odd-numbered lines
#odd_lines = []

# Iterate through the lines in the file
#for i in range(len(lines)):
    # Check if the line number is odd (1-based index)
#    if (i + 1) % 2 == 1:
 #       odd_lines.append(lines[i].strip())

# Print the odd-numbered lines
#for line in odd_lines:
 #   print(line)
# Open the text file for reading
with open('tno_coord', 'r') as file:
    lines = file.readlines()

# Initialize a list to store the first two columns from each line
columns = []

# Iterate through the lines in the file
for line in lines:
    # Split the line into columns based on whitespace
    columns_data = line.strip().split()
    
    # Take only the first two columns if available
    if len(columns_data) >= 2:
        first_two_columns = columns_data[:2]
        columns.append(first_two_columns)

# Print the first two columns from each line
for column_pair in columns:
    print("\t".join(column_pair))  # Separate the columns with a tab or another delimiter if needed

