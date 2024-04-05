import subprocess

def execute_python_file(file_path: str, data_path:str, other_argument: str = None) -> str:
    """
    Executes a Python file and captures its output, including print statements and return values.

    Parameters:
        file_path (str): The path of the Python file to execute.
        args (str): Additional arguments to pass to the Python file.

    Returns:
        str: The output of the Python file (print statements and return values) as a string.
    """
    try:
        # Execute the Python file with additional arguments
        if len(other_argument):
            command = ["python3", file_path, data_path] + list(other_argument.split(' '))
            result = subprocess.run(command, capture_output=True, text=True)
            if result.stderr:
                return result.stderr.strip()
            return result.stdout.strip()
        else:
            command = ["python3", file_path, data_path]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.stderr:
                return result.stderr.strip()
            return result.stdout.strip() 

    except Exception as e:
        return f"An error occurred: {str(e)}"

# # Example usage:
# file_path = "4_april_xgboost_forecast.py"  # Replace with the path of your Python file
# data_file_path = "/home/khudi/Desktop/autogen_ds/datasets/final_pants_dataset_complete.csv"  # Replace with the path of your data file
# file_output = execute_python_file(file_path, data_file_path)
# print("Output of the Python file:")
# print(file_output)