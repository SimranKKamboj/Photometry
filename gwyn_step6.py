import os
# Step 6: deleting files

# Function to delete unnecessary files
def delete_check_files():
    # Specify the directory where your files are located
    directory = '.'  # Current directory

    # Get a list of filenames to delete
    files_to_delete = [filename for filename in os.listdir(directory) if filename.startswith("check") or filename.startswith("zmag") or filename.startswith("testing_for_zmag_")]

    # Delete each file
    for filename in files_to_delete:
        os.remove(os.path.join(directory, filename))
    print("check*, zmag* and cl files deleted")
    print(" Do a pdump on gwyn_rp*.mag")

delete_check_files()

