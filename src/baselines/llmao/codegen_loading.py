import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from codegen import CodeGenPass
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import json
from pynvml import *
import csv
import sys
import time

torch.set_printoptions(profile="full")
csv.field_size_limit(2**31 - 1)
MAX_LEN = 128


class _CSVRowIterableDataset(IterableDataset):
    """Stream rows from every ``*.csv`` under a directory (replaces torchdata.datapipes)."""

    def __init__(self, root: str, row_processor):
        super().__init__()
        self.root = root
        self.row_processor = row_processor

    def __iter__(self):
        paths = sorted(
            os.path.join(self.root, name)
            for name in os.listdir(self.root)
            if name.endswith(".csv")
        )
        info = torch.utils.data.get_worker_info()
        if info is not None and info.num_workers > 0:
            paths = paths[info.id :: info.num_workers]
        for path in paths:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.reader(fh):
                    out = self.row_processor(row)
                    if out is not None:
                        yield out


class CSVDataLoader:
    def __init__(self, root, dim_model=1024, pretrain_type='350M'):
        self.root = root
        self.codegen_trainer = CodeGenPass()
        self.pretrain_type = pretrain_type
        print(f"Initializing CSVDataLoader with pretrain_type: {self.pretrain_type}")
        self.model, self.tokenizer = self.codegen_trainer.setup_model(
            type=self.pretrain_type)
        self.device_0 = self.codegen_trainer.codegen_device
        print(f"Using device for CodeGen I/O: {self.device_0}")
        print(f"Model loaded successfully. Tokenizer vocab size: {len(self.tokenizer)}")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.dim_model = dim_model
        print(f"Model initialized with dim_model: {self.dim_model}")

    def _newline_token_ids(self, tok):
            """Return every token id that the tokenizer ever produces for a newline char."""
            ids = {tok.encode("\n", add_special_tokens=False)[0]}          # single  '\n'
            ids.update(tok.encode("\n\n", add_special_tokens=False))       # double '\n\n'
            ids.update(tok.encode("\r\n", add_special_tokens=False))       # windows line-end
            return list(ids)

    def get_hidden_state(self, decoded_program):
        # 1. Cache newline ids once
        if not hasattr(self, "_nl_ids"):
            self._nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            # `encode` on one char can return >1 ids when using GPT-2-style BPE

        # 2. Tokenise and move to GPU
        ids = self.tokenizer(decoded_program,
                            return_tensors="pt",
                            truncation=True,
                            max_length=20_000).input_ids.to(self.device_0)

        vectors = []
        with torch.no_grad():
            for chunk in torch.split(ids, 2048, dim=1):
                out = self.model(chunk,
                                output_hidden_states=True,
                                return_dict=True)

                # 3️⃣  hidden states are in `out.hidden_states`
                last_h = out.hidden_states[-1].squeeze(0)          # [seq_len, D]

                # 4️⃣  grab all newline positions in *this* chunk
                nl_mask = torch.isin(chunk.squeeze(0),              # [seq_len]
                                    torch.tensor(self._nl_ids,
                                                device=chunk.device))
                if nl_mask.any():
                    vectors.append(last_h[nl_mask])                 # [#nl_in_chunk, D]

        if not vectors:
            raise ValueError("No newline tokens found")

        line_h = torch.cat(vectors, dim=0)                          # [num_lines, D]

        # 5️⃣  keep loader happy: clamp / pad so the count matches splitlines()
        expected = len(decoded_program.splitlines())
        if line_h.size(0) > expected:
            line_h = line_h[:expected]
        elif line_h.size(0) < expected:
            pad = line_h[-1:].repeat(expected - line_h.size(0), 1)
            line_h = torch.cat([line_h, pad], dim=0)

        return line_h

    def row_processer(self, row):
        """
        Process a single CSV row to create training data for the model.
        
        This function:
        1. Extracts and cleans the program code from the first column
        2. Parses the bug line labels from the second column (JSON format)
        3. Generates hidden states using the CodeGen model
        4. Validates sample dimensions and filters out invalid samples
        5. Applies padding to standardize sequence length
        6. Creates binary labels for newline tokens (bug lines)
        7. Generates attention masks for proper model training
        
        Args:
            row: CSV row containing [program_code, bug_line_labels]
            
        Returns:
            tuple: (hidden_states, labels, attention_mask, program_code, label_string) or None if invalid
        """
        # Step 1: Extract and clean program code from CSV row
        try:
            decoded_program = row[0]  # First column contains the program code
            # Normalize line endings and remove trailing newlines
            decoded_program = decoded_program.replace("\r\n", "\n").rstrip("\n")
            # Parse bug line labels from JSON format in second column
            label = json.loads(row[1])
        except Exception as e:
            print(f"Row parsing failed - invalid CSV format or JSON: {e}")
            return None
        
        # Step 2: Generate hidden states from the program code using CodeGen model
        hidden_states = self.get_hidden_state(decoded_program=decoded_program)
        
        # Step 3: Validate sample dimensions
        sample_shape = list(hidden_states.size())[0]  # Number of lines with hidden states
        native_sample_size = len(decoded_program.split("\n"))  # Actual number of lines in code
        
        # Filter out samples that are too long or have dimension mismatches
        if sample_shape > MAX_LEN:
            print(f"Sample filtered out - too long: {sample_shape} > {MAX_LEN}")
            return None
        if native_sample_size != (sample_shape):
            print(f"Sample filtered out - dimension mismatch: native_size={native_sample_size}, hidden_states_size={sample_shape}")
            return None
        
        # Step 4: Apply padding to standardize sequence length to MAX_LEN
        # Create zero padding for remaining positions
        sample_padding = torch.zeros(
            MAX_LEN - sample_shape, self.dim_model).to(self.device_0)
        
        # Concatenate original hidden states with padding
        final_hidden_states = torch.cat(
            [hidden_states, sample_padding], axis=0)
        
        # Step 5: Create binary tensor for newline tokens (bug line labels)
        # Initialize all positions as non-buggy (0)
        NL_tokens = np.zeros(MAX_LEN)
        try:
            # Set bug line positions to 1 based on labels
            NL_tokens[label] = np.ones(len(label))
        except Exception as e:
            print(f'Sample filtered out - label shape/indexing error: {e}')
            return None
        
        # Convert to tensor and move to GPU
        NL_tokens = torch.tensor(NL_tokens)
        NL_tokens = NL_tokens.to(self.device_0)
        
        # Step 6: Create attention mask to ignore padded positions during training
        # 1 for actual content, 0 for padded positions
        attention_mask = torch.cat(
            [torch.ones(sample_shape), torch.zeros(MAX_LEN - sample_shape)], axis=0
        ).to(self.device_0)
        
        # Step 7: Return processed sample as tuple
        output = (final_hidden_states, NL_tokens, attention_mask, decoded_program, row[1])
        return output

    def data_load(self):
        print("########################################################")
        print("DATA LOADING STARTED")
        print("########################################################")
        print(f"Starting data loading from root: {self.root}")

        if not os.path.exists(self.root):
            print(f"ERROR: Root directory does not exist: {self.root}")
            return None

        dataset = _CSVRowIterableDataset(self.root, self.row_processer)

        print("Data loading pipeline setup complete.")
        return dataset

def save_data():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", help="Path to data root")
    ap.add_argument("data_name", help="Name of dataset")
    ap.add_argument("biggest_model", help="")
    args = ap.parse_args()
    data_path = args.data_path
    data_name = args.data_name
    biggest_model = int(args.biggest_model)

    if biggest_model:
        pretrain_types = ['16B']
    else:
        pretrain_types = ['350M']

    for pretrain_type in pretrain_types:
        if pretrain_type == '350M':
            dim_model = 1024
        elif pretrain_type == '2B':
            dim_model = 2560
        elif pretrain_type == '6B':
            dim_model = 4096
        elif pretrain_type == '16B':
            dim_model = 6144
        print(f'Loading {pretrain_type} codegen states on {data_name}')

        # Data loading
        current_path = os.getcwd()
        root = os.path.join(current_path, data_path, data_name)
        
        total_samples = 0
        csv_files = [f for f in os.listdir(root) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files. Counting rows for ETA estimation...")
        for file in csv_files:
            try:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    total_samples += sum(1 for row in reader)
            except Exception as e:
                print(f"Couldn't read {file} for row counting: {e}")
        print(f"Estimated total samples: {total_samples}")

        data = CSVDataLoader(
            root=root,
            dim_model=dim_model,
            pretrain_type=pretrain_type,
        )
        datapipe = data.data_load()
        data_loaded = DataLoader(
            dataset=datapipe, batch_size=1, drop_last=True
        )

        print("########################################################")
        print("DATA LOADING COMPLETED")
        print("########################################################")

        try:
            first_batch = next(iter(data_loaded))
            print("✅ DataLoader has data! Example batch shapes:")
            print(f"  input: {first_batch[0][0].shape}")
            print(f"  label: {first_batch[1][0].shape}")
            print(f"  mask:  {first_batch[2][0].shape}")
        except StopIteration:
            print("⚠️  DataLoader is empty — no batches to process!")
            return  # or sys.exit(1), depending on how you want to bail out


        # Define base paths for outputs
        current_path = os.getcwd()
        states_base_path = os.path.join(current_path, data_path, 'codegen_states')
        instances_base_path = os.path.join(current_path, data_path, 'codegen_instances_csv')

        # Define specific directories for this run
        tensors_save_dir = os.path.join(states_base_path, f"{data_name}_{pretrain_type}")
        csv_save_dir = os.path.join(instances_base_path, f"{data_name}_{pretrain_type}")

        # Create directories if they don't exist
        os.makedirs(tensors_save_dir, exist_ok=True)
        os.makedirs(csv_save_dir, exist_ok=True)
        
        start_time = time.time()
        processed_count = 0
        for batch_iter, batch in enumerate(data_loaded):
            # Unpack batch contents. Since batch_size=1, we index [0] to get the item.
            input_tensor = batch[0][0].detach()
            label_tensor = batch[1][0].detach()
            mask_tensor = batch[2][0].detach()
            program_code = batch[3][0]
            label_string = batch[4][0]
            
            processed_count += 1
            if processed_count % 10 == 0:
                elapsed_time = time.time() - start_time
                samples_per_second = processed_count / elapsed_time
                remaining_samples = total_samples - processed_count
                if samples_per_second > 0:
                    eta_seconds = remaining_samples / samples_per_second
                    eta_minutes = eta_seconds / 60
                    print(f"Processed {processed_count}/{total_samples} samples. "
                          f"({samples_per_second:.2f} samples/sec). ETA: {eta_minutes:.2f} minutes.")

            # Save the processed tensors to a .pt file
            hidden_layer_dict = {'input': input_tensor, 'label': label_tensor, 'mask': mask_tensor}
            tensor_file_path = os.path.join(tensors_save_dir, f"{batch_iter}.pt")
            torch.save(hidden_layer_dict, tensor_file_path)

            # Save the original data to a .csv file
            csv_file_path = os.path.join(csv_save_dir, f"{batch_iter}.csv")
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([program_code, label_string])

        print('Finished preloading {} samples'.format(processed_count))


if __name__ == "__main__":
    save_data()