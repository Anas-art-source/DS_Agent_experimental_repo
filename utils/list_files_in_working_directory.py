
def list_files_in_working_directory(directory_path: str) -> str:
    """
    Lists all files in the specified directory and returns the list as a string.

    Parameters:
        directory_path (str): The path of the directory to list files from.
        
    Example Parameters: 
        - code
        - data

    Returns:
        str: A string containing the list of files in the directory.
    """
    import os

    # directory_path = '/home/khudi/Desktop/autogen_ds/coding/' + directory_name
    # Check if the directory path exists
    if not os.path.exists(directory_path):
        return "Directory does not exist."

    # List all files in the directory
    files = os.listdir(directory_path)

    # Convert the list of files to a string
    files_str = '\n'.join(files)
    return files_str

print(list_files_in_working_directory('/home/khudi/Desktop/autogen_ds/code'))