import pandas as pd

def grab_first_samples(input_file, output_file, num_samples=250):
    try:
        df = pd.read_csv(input_file, header=None)
        df_first = df.head(num_samples)
        df_first.to_csv(output_file, index=False, header=False)
        print(f"Extracted first {num_samples} samples from {input_file} to {output_file}")
    except Exception as e:
        print(f"Error while processing files: {e}")

def main():
    input_file = "data_processing/llmao_data/code_bugline_1idx_1000.csv"
    output_file = "data_processing/llmao_data/code_bugline_small_250_1idx.csv"
    grab_first_samples(input_file, output_file)

if __name__ == "__main__":
    main()
