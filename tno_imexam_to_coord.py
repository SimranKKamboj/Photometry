# Open the text file for reading
with open('tno.txt', 'r') as file:
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
