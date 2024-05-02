# Open the text file for reading
with open('tno_a', 'r') as file:
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

# Write the first two columns from each line to tno.txt
with open('tno_coord.txt', 'w') as output_file:
    for column_pair in columns:
        output_file.write("\t".join(column_pair) + "\n")  # Separate the columns with a tab and add a new line
