# Run this after working with iraf
import os
import re

# Step 5: Consolidate output into gwyn_rp2.mag without deleting extra files
def combine_check_files(output_file_name='gwyn_rp_tno.mag'):
    # Specify the directory where your check* files are located
    directory = '.'  # Current directory
    output_file_path = os.path.join(directory, output_file_name)

    # Get a list of filenames starting with "check" and sort them
    check_files = [filename for filename in os.listdir(directory) if filename.startswith("check")]
    check_files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))

    # Save the first 75 lines from the first file
    header_lines = []
    first_file = check_files[0]
    with open(os.path.join(directory, first_file), 'r') as first_input_file:
        for _ in range(75):
            line = first_input_file.readline()
            if line.startswith("#"):
                header_lines.append(line)

    # Combine the contents of check* files into the output file
    with open(output_file_path, 'w') as output_file:
        # Write the header lines to the output file
        output_file.writelines(header_lines)

        # Append the content of all files, filtering out lines starting with "#"
        for filename in check_files:
            with open(os.path.join(directory, filename), 'r') as input_file:
                file_content = [line for line in input_file if not line.startswith("#")]
                output_file.writelines(file_content)

    print(f"Combined content into {output_file_name}.")

combine_check_files()
