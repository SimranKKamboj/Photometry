# Open the text file for reading
with open('tno_coord.txt', 'r') as file:
    lines = file.readlines()

# Define the prefix for the output files
output_prefix = 'a'

# Iterate through the lines and save them into separate files
for i, line in enumerate(lines):
    # Construct the output file name with a three-digit number
    output_filename = f'{output_prefix}{i+101:03}.tno.coord'
    
    # Open the output file for writing
    with open(output_filename, 'w') as output_file:
        output_file.write(line)
    
    # Print the current line and the corresponding output file name
    print(f'Saved line {i+1} to {output_filename}')
