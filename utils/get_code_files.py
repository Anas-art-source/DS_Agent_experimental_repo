import csv

def get_code_files():
    """
    Read from a CSV file containing model information and generate output for each row.

    Args:
    csv_file (str): Path to the CSV file containing model information.

    Returns:
    str: Output string with information for each row in the CSV.
    """
    output = ""
    with open('/home/khudi/Desktop/autogen_ds/datasets/model_info.csv', 'r') as file:
        reader = csv.DictReader(file)
        row_count = 1
        for row in reader:
            output += f"**{row_count} Files**\n"
            output += f"File Path: {row['file_path']}\n"
            output += f"Date: {row['date']}\n"
            output += f"Code: {row['code']}\n\n"
            row_count += 1
    return output

# Example usage:
output_string = get_code_files()
print(output_string)