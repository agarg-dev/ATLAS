import contextlib
import os
import sys

# Ensure the repo root is on sys.path so codegen.py etc. can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------
# Lock the process to a single GPU before torch loads CUDA context
# ------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from codegen import CodeGenPass, cuda_kernels_available
import argparse
import numpy as np
import torch
import json
from pynvml import *
import csv
import time

# Use a safer large value that works on all platforms (typically 2 GB)
csv.field_size_limit(2**31 - 1)

torch.set_printoptions(profile="full")

MAX_LEN = 128
CHUNK = 2048  # CodeGen maximum context window size
BATCH = 6     # Tune according to available GPU memory

if cuda_kernels_available():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        gpu_name = nvmlDeviceGetName(handle)
        print(f"Using GPU: {gpu_name}")
    except Exception:
        print("NVML GPU name lookup skipped.")
else:
    print("CUDA kernels not usable on this machine (or LLMAO_DEVICE=cpu); running on CPU if needed.")



class CSVDataLoader:
    def __init__(self, root, dim_model=1024, pretrain_type='350M'):
        self.root = root
        self.codegen_trainer = CodeGenPass()
        self.pretrain_type = pretrain_type
        self.model, self.tokenizer = self.codegen_trainer.setup_model(
            type=self.pretrain_type
        )
        self.device_0 = self.codegen_trainer.codegen_device
        # ensure tokenizer has a pad token (required for batch padding)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        # ----------------------------------------------------
        # Inference-only setup
        self.model.eval()
        if str(self.device_0).startswith("cuda"):
            self.model.half()  # FP16 on GPU; model already placed by device_map
        for p in self.model.parameters():
            p.requires_grad_(False)
        # ----------------------------------------------------
        self.dim_model = dim_model

    def get_hidden_state(self, decoded_program):
        input_ids = self.tokenizer(
            decoded_program, return_tensors="pt", truncation=True, max_length=20000).input_ids

        input_ids = input_ids.to(self.device_0)
        split_input_ids = torch.split(input_ids, 2048, 1)
        hidden_states = []
        for input_id in split_input_ids:
            outputs = self.model(input_ids=input_id)[2]
            outputs = [h.detach() for h in outputs]
            attention_hidden_states = outputs[1:]
            hidden_state = attention_hidden_states[-1]
            nl_indices = torch.where((input_id == 198) | (input_id == 628))
            if len(nl_indices) > 1:
                nl_index = nl_indices[1]
            else:
                nl_index = nl_indices[0]
            nl_final_attention_states = hidden_state[torch.arange(
                hidden_state.size(0)), nl_index]
            hidden_states.append(nl_final_attention_states)
        final_attention_states = torch.cat(hidden_states, axis=0)
        return final_attention_states

    # #########################################################
    # New batched forward pass – much faster than per-sample
    # #########################################################
    @torch.no_grad()
    def get_hidden_states_batch(self, decoded_programs):
        """Compute final hidden states for a mini-batch of decoded programs.

        Args
        ----
        decoded_programs : list[str]
            Raw program strings (len <= BATCH).
        Returns
        -------
        list[Tensor]
            One tensor per program – shape (n_lines, dim_model).
        """
        # 1) Tokenise the whole batch at once
        tok_out = self.tokenizer(
            decoded_programs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=20000,
        )
        input_ids = tok_out.input_ids.to(self.device_0)  # (N, Lmax)

        # 2) Split every sequence into 2048-token chunks *in parallel*
        n_chunks = (input_ids.size(1) + CHUNK - 1) // CHUNK
        agg_hidden = [[] for _ in range(len(decoded_programs))]  # gather per sample

        amp_ctx = (
            torch.cuda.amp.autocast()
            if str(self.device_0).startswith("cuda")
            else contextlib.nullcontext()
        )
        with amp_ctx:
            for c in range(n_chunks):
                chunk = input_ids[:, c * CHUNK:(c + 1) * CHUNK]
                if chunk.size(1) == 0:
                    break  # no tokens left

                out = self.model(
                    input_ids=chunk,
                    use_cache=False,
                    output_hidden_states=True,
                    output_attentions=False,
                )
                h_last = out.hidden_states[-1]  # (N, chunk_len, D) – last layer

                # Identify newline tokens (198 and 628 in CodeGen tokenizer)
                nl_mask = (chunk == 198) | (chunk == 628)  # (N, chunk_len)
                # Ensure at least one True per row – fallback to last token
                nl_mask[:, -1] |= (~nl_mask).all(dim=1)

                # Gather hidden states at NL positions
                r_idx, c_idx = nl_mask.nonzero(as_tuple=True)
                gathered = h_last[r_idx, c_idx]  # (total_NL_in_chunk, D)

                for r, vec in zip(r_idx.tolist(), gathered):
                    agg_hidden[r].append(vec)

        # 3) Concatenate per program
        final = [torch.stack(v, dim=0) if len(v) else torch.empty(
                 0, self.dim_model, device=self.device_0) for v in agg_hidden]
        return final

    def iter_csv_rows(self):
        """Yield ``(program_text, label_json)`` for every row in every ``*.csv`` under ``root``."""
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(self.root, name)
            with open(path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.reader(fh):
                    if len(row) >= 2:
                        yield row[0], row[1]


def save_data():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", help="Path to data root")
    ap.add_argument("data_name", help="Name of dataset")
    ap.add_argument("biggest_model", help="")
    args = ap.parse_args()
    data_path = args.data_path
    data_name = args.data_name
    biggest_model = int(args.biggest_model)

    def format_seconds(seconds):
        """Convert seconds to H:MM:SS string."""
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours}:{mins:02d}:{secs:02d}"

    if biggest_model:
        pretrain_types = ['16B']
    else:
        pretrain_types = ['350M', '2B', '6B']
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
        root = f'{current_path}/{data_path}/{data_name}'
        # Count total lines (samples) across all CSV files for progress tracking
        total_samples = 0
        for filename in os.listdir(root):
            if filename.endswith(".csv"):
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    total_samples += sum(1 for _ in f)
        print(f"Total samples to process: {total_samples}")

        data = CSVDataLoader(
            root=root,
            dim_model=dim_model,
            pretrain_type=pretrain_type,
        )
        row_iter = data.iter_csv_rows()
        save_path = f'{current_path}/{data_path}/codegen_states'
        try:
            os.mkdir(save_path)
        except OSError:
            pass 
        os.chdir(save_path)
        if not os.path.isdir(f"{data_name}_{pretrain_type}"):
            os.mkdir(f"{data_name}_{pretrain_type}")

        
        start_time = time.time()
        batch_programs = []
        batch_labels = []
        processed = 0
        rows_iterated = 0

        def flush():
            nonlocal processed, batch_programs, batch_labels
            if not batch_programs:
                return
            hidden_list = data.get_hidden_states_batch(batch_programs)
            for i, (hidden_states, lbl_json) in enumerate(zip(hidden_list, batch_labels)):
                sample_shape = hidden_states.size(0)
                native_sample_size = len(batch_programs[i].split("\n"))
                if sample_shape + 1 > MAX_LEN or native_sample_size != sample_shape + 1:
                    continue  # skip invalid sample

                pad = torch.zeros(MAX_LEN - sample_shape, data.dim_model, device=data.device_0)
                final_hidden = torch.cat([hidden_states, pad], 0)

                NL_tokens = torch.zeros(MAX_LEN, device=data.device_0)
                try:
                    label_idx = json.loads(lbl_json)
                    NL_tokens[label_idx] = 1
                except Exception:
                    continue

                mask = torch.cat([
                    torch.ones(sample_shape, device=data.device_0),
                    torch.zeros(MAX_LEN - sample_shape, device=data.device_0)
                ], 0)

                hidden_layer_dict = {
                    'input': final_hidden.detach().cpu(),
                    'label': NL_tokens.detach().cpu(),
                    'mask': mask.detach().cpu(),
                }
                torch.save(hidden_layer_dict, f"{data_name}_{pretrain_type}/{processed}.pt")
                processed += 1
            batch_programs.clear()
            batch_labels.clear()

        # ------------------------------------------------------
        print("Starting batch processing...")
        for prog, lbl_json in row_iter:
            rows_iterated += 1
            batch_programs.append(prog)
            batch_labels.append(lbl_json)
            if len(batch_programs) == BATCH:
                flush()
                # --- Progress Update ---
                elapsed = time.time() - start_time
                time_per_row = elapsed / rows_iterated if rows_iterated > 0 else 0
                remaining_rows = total_samples - rows_iterated
                eta_seconds = time_per_row * remaining_rows
                print(
                    f"[{rows_iterated}/{total_samples}] Saved: {processed} | "
                    f"Elapsed: {format_seconds(elapsed)} | "
                    f"ETA: {format_seconds(eta_seconds)}"
                )

        flush()  # leftover
        total_elapsed = time.time() - start_time
        print(f"Finished preloading {processed} samples in {format_seconds(total_elapsed)}")


if __name__ == "__main__":
    save_data()