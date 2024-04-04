

def write_code_to_file(file_path: str, code: str) -> None:
    """
    Creates a Python file with the specified name and writes the given code into it.

    Parameters:
        file_name (str): The name of the Python file to create. Save python code it this directory: /home/khudi/Desktop/autogen_ds/code
        code (str): The Python code to write into the file.
    """
    # file_name = '/home/khudi/Desktop/autogen_ds/coding/code/' + file_name
    try:
        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write the code into the file
            file.write(code)
        print(f"Python file '{file_path}' created successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# # Example usage:
# file_name = "example.py"  # Name of the Python file to create
# python_code = """
# def main():
#     print("Hello, world!")

# if __name__ == "__main__":
#     main()
# """  # Python code to write into the file
# write_code_to_file(file_name, python_code)