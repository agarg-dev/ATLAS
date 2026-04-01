import torch
import argparse
import os

def observe_pt_file(file_path):
    """
    Load and observe the contents of a .pt file
    
    Args:
        file_path (str): Path to the .pt file
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    try:
        # Load the .pt file
        data = torch.load(file_path, map_location='cpu')
        
        print(f"=== Observing {file_path} ===")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys:")
            for key, value in data.items():
                print(f"  '{key}': {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                if hasattr(value, 'dtype'):
                    print(f"    Dtype: {value.dtype}")
                if isinstance(value, (int, float, str, bool)):
                    print(f"    Value: {value}")
                elif hasattr(value, '__len__') and len(value) < 10:
                    print(f"    Value: {value}")
                print()
        
        elif isinstance(data, torch.Tensor):
            print(f"Tensor shape: {data.shape}")
            print(f"Tensor dtype: {data.dtype}")
            print(f"Tensor device: {data.device}")
            print(f"Tensor requires_grad: {data.requires_grad}")
            print(f"First few values: {data.flatten()[:10]}")
            print(f"Min value: {data.min().item()}")
            print(f"Max value: {data.max().item()}")
            print(f"Mean value: {data.mean().item()}")
        
        elif isinstance(data, list):
            print(f"List with {len(data)} elements")
            for i, item in enumerate(data[:5]):  # Show first 5 items
                print(f"  [{i}]: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"    Shape: {item.shape}")
            if len(data) > 5:
                print(f"  ... and {len(data) - 5} more items")
        
        else:
            print(f"Data: {data}")
            if hasattr(data, '__dict__'):
                print("Attributes:")
                for attr, value in data.__dict__.items():
                    print(f"  {attr}: {type(value)}")
    
    except Exception as e:
        print(f"Error loading file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Observe contents of a .pt file")
    parser.add_argument("file_path", help="Path to the .pt file to observe")
    print(parser.parse_args())
    args = parser.parse_args()
    observe_pt_file(args.file_path)

if __name__ == "__main__":
    main()
