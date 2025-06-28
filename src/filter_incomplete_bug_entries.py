import json
import pandas as pd
from pathlib import Path

def filter_dataset(input_path, json_output_path, csv_output_path):
    """
    Filter the bug localization dataset to remove entries with missing information.
    
    Args:
        input_path: Path to the input JSON file
        json_output_path: Path to save the filtered JSON output
        csv_output_path: Path to save the filtered CSV output
    """
    print(f"Loading dataset from {input_path}")
    
    # Load the dataset
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Original dataset contains {len(dataset)} entries")
    
    # Track reasons for skipping
    skip_reasons = {
        "missing_path": 0,
        "missing_file": 0,
        "missing_function": 0,
        "invalid_function_span": 0,
        "missing_buggy_line": 0,
        "empty_fixed_lines": 0,
        "invalid_fixed_span": 0
    }
    
    # Filter the dataset
    filtered_dataset = []
    skipped_count = 0
    
    for entry in dataset:
        should_skip = False
        
        # Check for missing buggy code information
        if 'path_to_buggy_file' not in entry or not entry['path_to_buggy_file']:
            skip_reasons["missing_path"] += 1
            should_skip = True
            
        if 'buggy_file_name' not in entry or not entry['buggy_file_name']:
            skip_reasons["missing_file"] += 1
            should_skip = True
            
        if 'buggy_function_name' not in entry or not entry['buggy_function_name']:
            skip_reasons["missing_function"] += 1
            should_skip = True
            
        if 'buggy_function_span' not in entry or not entry['buggy_function_span'] or entry['buggy_function_span'] == [0, 0]:
            skip_reasons["invalid_function_span"] += 1
            should_skip = True
            
        if 'buggy_line_number' not in entry or not entry['buggy_line_number']:
            skip_reasons["missing_buggy_line"] += 1
            should_skip = True
        
        # Check for missing fix information
        if ('fixed_lines_number' not in entry or not entry['fixed_lines_number'] or 
            'fixed_lines_content' not in entry or not entry['fixed_lines_content']):
            skip_reasons["empty_fixed_lines"] += 1
            should_skip = True
            
        if 'fixed_lines_span' not in entry or not entry['fixed_lines_span'] or entry['fixed_lines_span'] == [0, 0]:
            skip_reasons["invalid_fixed_span"] += 1
            should_skip = True
        
        if should_skip:
            skipped_count += 1
            continue
        
        # Include this entry in the filtered dataset
        filtered_dataset.append(entry)
    
    print(f"Filtered dataset contains {len(filtered_dataset)} entries")
    print(f"Skipped {skipped_count} entries with missing information")
    
    # Save as JSON
    with open(json_output_path, 'w') as f:
        json.dump(filtered_dataset, f, indent=2)
    print(f"Saved filtered dataset to {json_output_path}")
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(filtered_dataset)
    
    # Handle list columns for CSV export
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    df.to_csv(csv_output_path, index=False)
    print(f"Saved filtered dataset to {csv_output_path}")
    
    # Print detailed skip reasons
    print("\nReasons for skipping entries:")
    for reason, count in skip_reasons.items():
        print(f"  {reason.replace('_', ' ').title()}: {count}")
    
    return len(dataset), len(filtered_dataset), skipped_count

if __name__ == "__main__":
    # Define file paths
    data_dir = Path("../data")
    input_path = data_dir / "bug_localization_dataset_updated.json"
    json_output_path = data_dir / "bug_localization_dataset_filtered.json"
    csv_output_path = data_dir / "bug_localization_dataset_filtered.csv"
    
    # Filter the dataset
    total, filtered, skipped = filter_dataset(input_path, json_output_path, csv_output_path)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Dataset Filtering Summary:")
    print(f"  Original dataset:  {total} entries")
    print(f"  Filtered dataset:  {filtered} entries")
    print(f"  Skipped entries:   {skipped} entries")
    print("="*50)