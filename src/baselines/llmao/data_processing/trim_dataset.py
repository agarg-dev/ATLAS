import pandas as pd

def trim_dataset():
    """Remove entries from code_bugline.csv that have more than 128 lines in column 1"""
    
    # Read the CSV file
    df = pd.read_csv('./llmao_data/code_bugline.csv')
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Filter out rows where column 1 has more than 128 lines
    df_filtered = df[df.iloc[:, 0].astype(str).apply(lambda x: len(x.split('\n')) <= 828)]
    
    print(f"Filtered dataset size: {len(df_filtered)} rows")
    print(f"Removed {len(df) - len(df_filtered)} rows")
    
    # Save the filtered dataset
    df_filtered.to_csv('./llmao_data/code_bugline_trimmed.csv', index=False)
    print("Saved trimmed dataset to 'code_bugline_trimmed.csv'")

if __name__ == "__main__":
    trim_dataset()
