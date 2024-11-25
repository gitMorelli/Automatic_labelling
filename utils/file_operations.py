import os

def delete_all_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    # Iterate over all items in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (and not a subfolder)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file
            print(f"Deleted: {file_path}")
        else:
            print(f"Skipping folder: {file_path}")
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"The file {file_path} does not exist.")