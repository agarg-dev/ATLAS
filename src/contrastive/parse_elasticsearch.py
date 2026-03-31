import pandas as pd
import os
import json
import argparse

def get_dataset(datafile_path):
    """Parse bug records into a DataFrame with one row per buggy line."""
    data = pd.read_json(datafile_path)
    rows = []

    base_columns = ['issue_id', 'repo_name', 'repo_url', 'title', 'body', 'before_fix_sha', 'after_fix_sha']

    for index, row in data.iterrows():
        base_info = {col: row[col] for col in base_columns if col in row}
        base_info['stack_trace'] = f"{row.get('title', '')} {row.get('body', '')}"

        if 'localization_data' in row and row['localization_data'] is not None:
            new_rows = parse_localization_data(row['localization_data'], base_info)
            rows.extend(new_rows)

    return pd.DataFrame(rows)

def parse_localization_data(localization_data, base_info):
    """Extract per-file rows from a single record's localization_data block."""
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
    """Return one dataset row per buggy line in a file's diff data."""
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
    """Match buggy lines to fixed lines by minimum distance, without overlap."""
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
    """Return the function entry whose span contains line_num, or None."""
    for function in buggy_functions:
        start_line = function.get('start_line', 0)
        end_line = function.get('end_line', 0)
        if start_line <= line_num <= end_line:
            return function
    return None

def create_dataset_row(buggy_line, fixed_line_group, function_info, file_path, file_name, directory_path, base_info):
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
    parser = argparse.ArgumentParser(description="Parse Elasticsearch/Beetlebox bug dataset")
    parser.add_argument("--input_path", default="../../data/processed_samples_train_all.json",
                        help="Path to raw processed_samples_train_all.json")
    parser.add_argument("--output_path", default="../../data/elasticsearch/bug_localization_dataset.json",
                        help="Path to output bug_localization_dataset.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    dataset = get_dataset(args.input_path)
    dataset.to_json(args.output_path, orient='records', indent=2)
    print(f"Parsed {len(dataset)} bug records → {args.output_path}")
