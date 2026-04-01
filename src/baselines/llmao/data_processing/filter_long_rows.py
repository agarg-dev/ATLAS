import csv
import json

MAX_LEN = 130  # Assuming this is the max length based on the context

def filter_long_rows(input_file, output_file):
    """
    Filter out rows that are too long based on the row_processer logic.
    Keeps rows where:
    1. sample_shape + 1 <= MAX_LEN
    2. native_sample_size == (sample_shape + 1)
    """
    filtered_count = 0
    total_count = 0
    
    # Increase the field size limit to handle large CSV fields
    csv.field_size_limit(1000000)  # Set to 1MB limit
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            total_count += 1
            
            try:
                decoded_program = row[0]
                label = json.loads(row[1])
            except:
                print(f"Skipping row {total_count}: Failed to parse")
                continue
            
            # Calculate sample_shape (number of lines - 1, similar to hidden_states.size()[0])
            lines = decoded_program.split("\n")
            native_sample_size = len(lines)
            sample_shape = native_sample_size - 1  # Assuming this matches hidden_states.size()[0]
            
            # Apply the same filtering logic as row_processer
            if sample_shape + 1 > MAX_LEN or native_sample_size != (sample_shape + 1):
                continue
            
            # If we get here, the row passes the filter
            writer.writerow(row)
            filtered_count += 1
    
    print(f"Filtered {total_count} rows down to {filtered_count} rows")
    print(f"Saved filtered data to {output_file}")

def main():
    input_file = "data_processing/llmao_data/code_bugline_1idx_full.csv"
    output_file = "data_processing/llmao_data/code_bugline_1idx_filtered.csv"
    filter_long_rows(input_file, output_file)

if __name__ == "__main__":
    main()
