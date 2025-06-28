import pandas as pd
import os
import json

def get_dataset(datafile_path):
    """
    Parse the JSON file and create a dataset where each buggy line is a separate row.
    
    Args:
        datafile_path (str): Path to the JSON file containing bug data
        
    Returns:
        pd.DataFrame: Dataset with one row per buggy line
    """
    data = pd.read_json(datafile_path)
    rows = []

    base_columns = ['issue_id', 'repo_name', 'repo_url', 'title', 'body', 'before_fix_sha', 'after_fix_sha']

    for index, row in data.iterrows():
        print(f"Processing data point {index}")
        base_info = {col: row[col] for col in base_columns if col in row}
        base_info['stack_trace'] = f"{row.get('title', '')} {row.get('body', '')}"

        if 'localization_data' in row and row['localization_data'] is not None:
            new_rows = parse_localization_data(row['localization_data'], base_info)
            rows.extend(new_rows)

    return pd.DataFrame(rows)

def parse_localization_data(localization_data, base_info):
    """
    Process the localization data for a record and extract file information.
    
    Args:
        localization_data (dict): The localization data containing file information
        base_info (dict): Base information about the bug from the record
        
    Returns:
        list: List of dictionaries, each representing a row in the final dataset
    """
    result_rows = []

    if 'files' not in localization_data:
        return result_rows

    for file_path, file_data in localization_data['files'].items():
        directory_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)

        file_rows = extract_file_info(file_path, file_name, directory_path, file_data, base_info)
        result_rows.extend(file_rows)

    return result_rows

def extract_file_info(file_path, file_name, directory_path, file_data, base_info):
    """
    Extract information from a file's data, including buggy and fixed lines.
    
    Args:
        file_path (str): Full path to the file
        file_name (str): Name of the file
        directory_path (str): Directory containing the file
        file_data (dict): Data about the file, including diff information
        base_info (dict): Base information about the bug
        
    Returns:
        list: List of dictionaries, each representing a buggy line
    """
    result_rows = []
    diff_data = file_data.get('diff_data', {})
    buggy_lines = diff_data.get('buggy_lines', [])
    fixed_lines = diff_data.get('fixed_lines', [])
    line_mappings = match_buggy_to_fixed_lines(buggy_lines, fixed_lines)
    buggy_functions = file_data.get('buggy_functions', [])

    for buggy_line in buggy_lines:
        fixed_line_group = line_mappings.get(buggy_line['line_num'], [])
        function_info = identify_buggy_function(buggy_line['line_num'], buggy_functions)
        row = create_dataset_row(
            buggy_line,
            fixed_line_group,
            function_info,
            file_path,
            file_name,
            directory_path,
            base_info
        )
        result_rows.append(row)

    return result_rows

def match_buggy_to_fixed_lines(buggy_lines, fixed_lines):
    """
    Implement a heuristic to match buggy lines to their corresponding fixed lines without overlap.
    """
    line_mappings = {}

    if not buggy_lines or not fixed_lines:
        return line_mappings

    for buggy_line in buggy_lines:
        line_mappings[buggy_line['line_num']] = []

    fixed_line_map = {fl['line_num']: fl for fl in fixed_lines}
    all_assignments = []

    for buggy_line in buggy_lines:
        buggy_line_num = buggy_line['line_num']
        for fixed_line in fixed_lines:
            fixed_line_num = fixed_line['line_num']
            distance = abs(fixed_line_num - buggy_line_num)
            all_assignments.append((distance, buggy_line_num, fixed_line_num))

    all_assignments.sort(key=lambda x: x[0])
    assigned_fixed_line_nums = set()

    for distance, buggy_line_num, fixed_line_num in all_assignments:
        if fixed_line_num in assigned_fixed_line_nums:
            continue
        line_mappings[buggy_line_num].append(fixed_line_map[fixed_line_num])
        assigned_fixed_line_nums.add(fixed_line_num)

    return line_mappings

def identify_buggy_function(line_num, buggy_functions):
    """
    Identify the buggy function that contains a given line number.
    
    Args:
        line_num (int): Line number to check
        buggy_functions (list): List of buggy functions with their spans
        
    Returns:
        dict: Information about the function containing the line, or None if not found
    """
    for function in buggy_functions:
        start_line = function.get('start_line', 0)
        end_line = function.get('end_line', 0)
        if start_line <= line_num <= end_line:
            return function
    return None

def create_dataset_row(buggy_line, fixed_line_group, function_info, file_path, file_name, directory_path, base_info):
    """
    Create a row for the final dataset based on a buggy line and its corresponding information.
    """
    row = base_info.copy()
    row['path_to_buggy_file'] = directory_path
    row['buggy_file_name'] = file_name

    if function_info:
        row['buggy_function_name'] = function_info.get('name', '')
        row['buggy_function_span'] = [
            function_info.get('start_line', 0),
            function_info.get('end_line', 0)
        ]
    else:
        row['buggy_function_name'] = ''
        row['buggy_function_span'] = [0, 0]

    row['buggy_line_number'] = buggy_line['line_num']
    row['buggy_line_content'] = buggy_line['content']
    fixed_line_numbers = [fl['line_num'] for fl in fixed_line_group]
    fixed_line_contents = [fl['content'] for fl in fixed_line_group]
    row['fixed_lines_number'] = fixed_line_numbers
    row['fixed_lines_content'] = fixed_line_contents

    if fixed_line_numbers:
        row['fixed_lines_span'] = [min(fixed_line_numbers), max(fixed_line_numbers)]
    else:
        row['fixed_lines_span'] = [0, 0]

    return row

if __name__ == "__main__":
    datafile_path = "../data/processed_samples_train_all.json"
    dataset = get_dataset(datafile_path)
    dataset.to_csv("../data/bug_localization_dataset.csv", index=False)
    print(f"Dataset created with {len(dataset)} rows.")
    dataset.to_json("../data/bug_localization_dataset.json", orient='records', indent=2)
