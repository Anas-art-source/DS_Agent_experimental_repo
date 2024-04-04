import subprocess

def execute_python_file(file_path: str, data_file_path: str) -> str:
    """
    Executes a Python file and captures its output, including print statements and return values.

    Parameters:
        file_path (str): The path of the Python file to execute.
        data_file_path (str): The path of the data file to pass as an argument to the Python file.

    Returns:
        str: The output of the Python file (print statements and return values) as a string.
    """
    file_path = '/home/khudi/Desktop/autogen_ds/coding/code/' + file_path
    
    print(data_file_path)

    try:
        # Execute the Python file with the data file path as an argument
        result = subprocess.run(["python3", file_path, data_file_path], capture_output=True, text=True)
        if result.stderr:
            return result.stderr.strip()
        return result.stdout.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage:
file_path = "4_april_xgboost_forecast.py"  # Replace with the path of your Python file
data_file_path = "/home/khudi/Desktop/autogen_ds/datasets/final_pants_dataset_complete.csv"  # Replace with the path of your data file
file_output = execute_python_file(file_path, data_file_path)
print("Output of the Python file:")
print(file_output)