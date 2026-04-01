import pandas as pd
import re
import os
import glob
from typing import Tuple

def analyze_dataset(file_path: str) -> Tuple[int, int, int, int, float, int, int, float, int]:
    """
    Analyze a dataset and return statistics.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Tuple containing:
        - Number of rows
        - Token count for the first column
        - Count of numbers in the second column
        - Min token count per row
        - Average token count per row
        - Max token count per row
        - Min line count per row (first column)
        - Average line count per row (first column)
        - Max line count per row (first column)
    """
    # Read the dataset without header
    df = pd.read_csv(file_path, header=None)
    
    # Get number of rows
    num_rows = len(df)
    
    # Get token count for first column and per-row statistics
    first_column = df.iloc[:, 0].astype(str)
    token_counts_per_row = [len(str(cell).split()) for cell in first_column]
    
    total_token_count = sum(token_counts_per_row)
    min_tokens = min(token_counts_per_row) if token_counts_per_row else 0
    avg_tokens = sum(token_counts_per_row) / len(token_counts_per_row) if token_counts_per_row else 0
    max_tokens = max(token_counts_per_row) if token_counts_per_row else 0
    
    # Get line count for first column and per-row statistics
    line_counts_per_row = [len(str(cell).split('\n')) for cell in first_column]
    min_lines = min(line_counts_per_row) if line_counts_per_row else 0
    avg_lines = sum(line_counts_per_row) / len(line_counts_per_row) if line_counts_per_row else 0
    max_lines = max(line_counts_per_row) if line_counts_per_row else 0
    
    # Count numbers in second column
    second_column = df.iloc[:, 1].astype(str)
    number_count = 0
    for cell in second_column:
        # Find all numbers (integers and floats) in the cell
        numbers = re.findall(r'-?\d+\.?\d*', str(cell))
        number_count += len(numbers)
    
    return num_rows, total_token_count, number_count, min_tokens, avg_tokens, max_tokens, min_lines, avg_lines, max_lines

def main():
    """Main function to analyze all CSV files in the data/ folder."""
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        print(f"Error: {data_folder} folder not found")
        return
    
    # Find all CSV files in data/ folder and subdirectories
    csv_files = glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {data_folder} folder")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) in {data_folder} folder:")
    print("=" * 60)
    
    # Track overall statistics across all datasets
    all_row_counts = []
    
    for file_path in csv_files:
        print(f"\nAnalyzing: {file_path}")
        try:
            rows, tokens, numbers, min_tokens, avg_tokens, max_tokens, min_lines, avg_lines, max_lines = analyze_dataset(file_path)
            all_row_counts.append(rows)
            print(f"  Number of rows: {rows}")
            print(f"  Token count (first column): {tokens}")
            print(f"  Number count (second column): {numbers}")
            print(f"  Min tokens per row: {min_tokens}")
            print(f"  Avg tokens per row: {avg_tokens:.2f}")
            print(f"  Max tokens per row: {max_tokens}")
            print(f"  Min lines per row (first column): {min_lines}")
            print(f"  Avg lines per row (first column): {avg_lines:.2f}")
            print(f"  Max lines per row (first column): {max_lines}")
        except Exception as e:
            print(f"  Error analyzing dataset: {e}")
    
    # Print overall row count statistics
    if all_row_counts:
        print("\n" + "=" * 60)
        print("OVERALL ROW COUNT STATISTICS:")
        print(f"  Min rows across datasets: {min(all_row_counts)}")
        print(f"  Avg rows across datasets: {sum(all_row_counts) / len(all_row_counts):.2f}")
        print(f"  Max rows across datasets: {max(all_row_counts)}")
        print(f"  Total rows across all datasets: {sum(all_row_counts)}")

if __name__ == "__main__":
    main()
