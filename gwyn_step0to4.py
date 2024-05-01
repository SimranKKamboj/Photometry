import os
# step 0: initialization

a_list_file = "a.list"
with open(a_list_file, 'r') as f:
    input_files = f.readlines()
input_files = [x.strip() for x in input_files]

# Assuming nframes is defined somewhere in your code
nframes = len(input_files)

zmag_list_file = "gwyn_rp5.list"

# Step 1: Split zmag_rp2.list into separate files
def split_zmag_file(zmag_list_file):
    with open(zmag_list_file, 'r') as f:
        zmag_lines = f.readlines()

    for i, line in enumerate(zmag_lines, start=1):
        zmag_filename = "zmag{}".format(i)
        with open(zmag_filename, 'w') as zmag_file:
            zmag_file.write(line.strip())
print("zmag files created")  # Debug print to check if file creation is happening

# Call the function to split the zmag_rp2.list file
split_zmag_file(zmag_list_file)


# Step 2: Process a.list to get input filenames
def get_input_filenames(a_list_file):
    with open(a_list_file, 'r') as f:
        input_files = f.readlines()
    return [x.strip() for x in input_files]

output_file_cl = "testing_for_zmag.cl"
        
# Step 3: Generate phot commands and save to separate files
def generate_phot_commands(input_files, nframes, output_file_base):
    for i in range(nframes):
        output_file = "{}_{}.cl".format(output_file_base, i+1)
        with open(output_file, 'w') as f:
            frame = input_files[i]
            zmag_file = "zmag{}".format(i+1)
            with open(zmag_file, 'r') as zmag_f:
                zmag = float(zmag_f.read().strip())
            check_value = "check{}".format(i+1)
            epar_command = "photpars.zmag = {}\n".format(zmag)
            phot_command = "phot {} relative.star {}\n".format(frame, check_value)
            f.write(epar_command)
            f.write(phot_command)
    print("testing_for_zmag*.cl files created") 

output_file_base = "testing_for_zmag"
generate_phot_commands(input_files, nframes, output_file_base)


# # Step 3: Generate phot commands and save to file
# def generate_phot_commands(input_files, nframes, output_file_cl):
#     with open(output_file_cl, 'w') as f:
#         for i in range(nframes):
#             frame = input_files[i]
#             zmag_file = "zmag{}".format(i+1)
#             with open(zmag_file, 'r') as zmag_f:
#                 zmag = float(zmag_f.read().strip())
#             check_value = "check{}".format(i+1)
#             epar_command = "photpars.zmag = {}\n".format(zmag)
#             phot_command = "phot {} relative.star {}\n".format(frame, check_value)
#             f.write(epar_command)
#             f.write(phot_command)


# generate_phot_commands(input_files, nframes, output_file_cl)

#Step 4: combining to testing_for_zmag.cl
import re
import glob 

# Get a list of all filenames starting with 'testing_for_zmag'
file_list = glob.glob('testing_for_zmag*')

# Sort the filenames based on the numerical value of the * part
file_list.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))

# Write the filenames to the output file
output_file = 'testing_for_zmag.cl'
with open(output_file, 'w') as f:
    for filename in file_list:
        f.write("cl < " + filename + '\n')

print("testing_for_zmag*.cl files combined") 
print("Go to iraf and run cl < testing_for_zmag.cl")