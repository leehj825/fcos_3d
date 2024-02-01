import os
import glob

# Path to the folder containing the text files
folder_path = '/Users/hyejunlee/fcos_3d/data/waymo_single/training/label_0/'

# Function to remove duplicate lines from a file
def remove_duplicates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    unique_lines = list(set(lines))
    with open(file_path, 'w') as file:
        file.writelines(unique_lines)

# Iterating over each text file in the folder
for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
    remove_duplicates(file_path)
