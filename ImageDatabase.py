"""
This script will contain all the necissary
"""



import os
import json

#iterativley renames files of a given extension within a directory with an ID
#example usage:
#   rename_files(Mar14Tests, ".jpg")
def rename_files(directory, file_extension):
    # Get a list of all files, sorted for consistency
    jpg_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(file_extension)])

    #ensure the dot is there if it is ommited
    if not(file_extension[0] == '.'):
        file_extension = '.' + file_extension

    for index, filename in enumerate(jpg_files, start=1):
        # Create new filename with zero-padded index
        new_name = f"{index:04}{file_extension}"
        # Get full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        # Rename the file
        print(f"Renaming '{filename}' to '{new_name}'")
        os.rename(old_path, new_path)



#read in the

def read_json(file_path):
    #read the contents of the file, itemize them
    file = open(file_path, 'r')
    data = json.load(file)
    file.close()

    for key in data:
        print(key)


read_json("dataset.json")

def verify_dataset(config_path):
    #ensure config file exists
    file = open(config_path, 'r')
    data = json.load(file)
    file.close()


    #at this point, it assumes it is formatted correctly







