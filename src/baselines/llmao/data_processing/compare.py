import pandas as pd
import pathlib
import json
import csv
from typing import List, Dict, Any

def analyze_csv_structure(csv_path: pathlib.Path) -> Dict[str, Any]:
    """Analyze the structure and format of a CSV file."""
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}
    
    try:
        # Read first few rows to analyze structure
        df = pd.read_csv(csv_path, nrows=10)
        
        # Get basic info
        info = {
            "file_path": str(csv_path),
            "columns": list(df.columns),
            "num_columns": len(df.columns),
            "dtypes": df.dtypes.to_dict(),
        }
        
        # Try to get total row count
        try:
            total_rows = sum(1 for _ in open(csv_path, 'r', encoding='utf-8'))
            info["total_rows"] = total_rows - 1  # subtract header
        except:
            info["total_rows"] = "Could not determine"
        
        return info
        
    except Exception as e:
        return {"error": f"Error analyzing {csv_path}: {str(e)}"}

def compare_csv_formats():
    """Compare CSV files from defects4j and beetlebox directories."""
    
    # Define paths to check
    defects4j_path = pathlib.Path("data/defects4j")
    beetlebox_path = pathlib.Path("beetlebox")
    
    print("=== CSV Format Comparison ===\n")
    
    # Find CSV files in both directories
    defects4j_csvs = list(defects4j_path.glob("*.csv")) if defects4j_path.exists() else []
    beetlebox_csvs = list(beetlebox_path.glob("*.csv")) if beetlebox_path.exists() else []
    
    print(f"Found {len(defects4j_csvs)} CSV files in defects4j/")
    print(f"Found {len(beetlebox_csvs)} CSV files in beetlebox/")
    print()
    
    # Analyze defects4j files
    defects4j_analyses = []
    if defects4j_csvs:
        print("=== DEFECTS4J FILES ===")
        for csv_file in defects4j_csvs:
            print(f"\nAnalyzing: {csv_file.name}")
            print("-" * 50)
            analysis = analyze_csv_structure(csv_file)
            
            if "error" in analysis:
                print(f"Error: {analysis['error']}")
                continue
                
            defects4j_analyses.append(analysis)
            print(f"Columns ({analysis['num_columns']}): {analysis['columns']}")
            print(f"Total rows: {analysis['total_rows']}")
            print(f"Data types: {analysis['dtypes']}")
    
    # Analyze beetlebox files
    beetlebox_analyses = []
    if beetlebox_csvs:
        print("\n\n=== BEETLEBOX FILES ===")
        for csv_file in beetlebox_csvs:
            print(f"\nAnalyzing: {csv_file.name}")
            print("-" * 50)
            analysis = analyze_csv_structure(csv_file)
            
            if "error" in analysis:
                print(f"Error: {analysis['error']}")
                continue
                
            beetlebox_analyses.append(analysis)
            print(f"Columns ({analysis['num_columns']}): {analysis['columns']}")
            print(f"Total rows: {analysis['total_rows']}")
            print(f"Data types: {analysis['dtypes']}")
    
    # Compare formats if both exist
    if defects4j_analyses and beetlebox_analyses:
        print("\n\n=== FORMAT COMPARISON ===")
        
        # Get representative files for comparison
        defects4j_analysis = defects4j_analyses[0]
        beetlebox_analysis = beetlebox_analyses[0]
        
        print("Column comparison:")
        print(f"  Defects4J columns: {defects4j_analysis['columns']}")
        print(f"  Beetlebox columns: {beetlebox_analysis['columns']}")
        
        # Check if columns match
        if defects4j_analysis['columns'] == beetlebox_analysis['columns']:
            print("  ✓ Column names match exactly")
        else:
            print("  ✗ Column names differ")
            defects4j_cols = set(defects4j_analysis['columns'])
            beetlebox_cols = set(beetlebox_analysis['columns'])
            
            common = defects4j_cols & beetlebox_cols
            defects4j_only = defects4j_cols - beetlebox_cols
            beetlebox_only = beetlebox_cols - defects4j_cols
            
            if common:
                print(f"    Common columns: {list(common)}")
            if defects4j_only:
                print(f"    Defects4J only: {list(defects4j_only)}")
            if beetlebox_only:
                print(f"    Beetlebox only: {list(beetlebox_only)}")
        
        # Compare data types
        print("\nData type comparison:")
        if defects4j_analysis['dtypes'] == beetlebox_analysis['dtypes']:
            print("  ✓ Data types match exactly")
        else:
            print("  ✗ Data types differ")
            for col in set(defects4j_analysis['columns'] + beetlebox_analysis['columns']):
                d4j_type = defects4j_analysis['dtypes'].get(col, 'N/A')
                bb_type = beetlebox_analysis['dtypes'].get(col, 'N/A')
                if d4j_type != bb_type:
                    print(f"    {col}: Defects4J={d4j_type}, Beetlebox={bb_type}")

if __name__ == "__main__":
    compare_csv_formats()
