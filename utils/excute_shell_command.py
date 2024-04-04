import subprocess

def execute_shell_command(command: str) -> str:
    """
    Executes a shell command specified as a string and captures its output.

    Parameters:
        command (str): The shell command to execute.

    Returns:
        str: The output of the shell command as a string.
    """
    try:
        print("HEREE")
        # Execute the shell command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        print(output)
        return output
    except Exception as e:
        return f"An error occurred: {str(e)}"

# # Example usage:
# command = "pip install xgboost"  # Replace with the shell command you want to execute
# command_output = execute_shell_command(command)
# print("Output of the shell command:")
# print(command_output)