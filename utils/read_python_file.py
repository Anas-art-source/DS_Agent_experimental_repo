def read_python_file(file_path: str) -> str:
    """
    Reads the content of a file given its path and returns the content as a string.

    Parameters:
        file_name (str): The name of the file to read.

    Returns:
        str: The content of the file as a string.
    """
    # file_path = '/home/khudi/Desktop/autogen_ds/coding/code/' + file_name

    try:
        # Open the file and read its content
        with open(file_path, 'r') as file:
            file_content = file.read()
        return file_content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    


# # Example usage:
# file_path = "demo.py"  # Replace with the path of your file
# file_content = read_file(file_path)
# print("Content of the file:")
# print(file_content)
