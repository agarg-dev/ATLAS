import json
from collections import defaultdict

def count_unique_files_in_beetlebox(num_items=1000):
    """
    Read beetlebox.json and count unique files for the first X items.
    
    Args:
        num_items (int): Number of items to process from the beginning
    
    Returns:
        int: Number of unique files found
    """
    unique_files = set()
    
    try:
        with open('data_processing/llmao_data/beetlebox.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process first X items
        items_to_process = data[:num_items] if len(data) > num_items else data
        
        for item in items_to_process:
            # Extract file path and name. Prefer explicit path + filename combo if available,
            # otherwise fall back to single-field heuristics.

            # 1. LLMAO / Beetlebox schema (path_to_buggy_file + buggy_file_name)
            if "path_to_buggy_file" in item and "buggy_file_name" in item:
                combined = f"{item['path_to_buggy_file'].rstrip('/')}/{item['buggy_file_name']}"
                unique_files.add(combined)
                continue  # done with this item

            # 2. Generic single-field look-ups
            file_info = None
            for key in [
                "file",
                "filename",
                "path",
                "source_file",
                "file_path",
            ]:
                if key in item:
                    file_info = item[key]
                    break

            if file_info:
                unique_files.add(file_info)
        
        print(f"Processed {len(items_to_process)} items")
        print(f"Found {len(unique_files)} unique files")
        
        return len(unique_files)
        
    except FileNotFoundError:
        print("Error: beetlebox.json file not found")
        return 0
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    # Count unique files for first 100 items by default
    count_unique_files_in_beetlebox(1000)
