import os
import json
import re
import numpy as np
import faiss
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
import random
import torch.nn.functional as F


class CodeBERTEmbedder:
    def __init__(self, 
                 tokenizer_path="../models/codebert/codebert_tokenizer", 
                 model_ckpt_path="../models/codebert/best_codebert_triplet.pt",
                 device="cuda"):

        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        base_model = RobertaModel.from_pretrained("microsoft/codebert-base")

        raw_ckpt = torch.load(model_ckpt_path, map_location=device)
        cleaned_ckpt = {k.replace("model.", ""): v for k, v in raw_ckpt.items()}
        base_model.load_state_dict(cleaned_ckpt)

        self.model = base_model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def _sliding_window_embed(self, text, max_tokens=512, stride=510, max_chunks=12):
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        chunks = [input_ids[i:i + max_tokens - 2] for i in range(0, len(input_ids), stride)]
        chunks = chunks[:max_chunks]

        embeddings = []
        for chunk in chunks:
            ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
            mask = [1] * len(ids)
            pad_len = max_tokens - len(ids)
            ids += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len

            input_tensor = {
                "input_ids": torch.tensor([ids], device=self.device),
                "attention_mask": torch.tensor([mask], device=self.device)
            }

            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type if self.device.type == "cuda" else "cpu"):
                    output = self.model(**input_tensor)
                    cls_embedding = output.last_hidden_state[:, 0, :]
                    embeddings.append(cls_embedding.squeeze(0))

        stacked = torch.stack(embeddings)
        mean_pooled = stacked.mean(dim=0)
        return F.normalize(mean_pooled, dim=0).cpu().numpy().astype(np.float32)

    def embed_stack_trace(self, stack_trace):
        return self._sliding_window_embed(stack_trace)


def load_triplet_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_faiss_index(index_path, dim):
    index = faiss.read_index(index_path)
    assert index.d == dim, f"Expected dimension {dim}, got {index.d}"
    return index


def search_similar_files(index, query_embedding, top_k=3):
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return indices[0].tolist(), distances[0].tolist()


def create_bug_dataset_entry(triplet, candidate_ids, bug_location, triplets, metadata, stack_trace_embedding):

    file_contents = []
    for cid in candidate_ids:
        candidate_meta = metadata[cid]
        candidate_triplet_id = candidate_meta["triplet_id"]
        candidate_type = candidate_meta["type"]
        candidate_code = triplets[candidate_triplet_id][candidate_type]
        file_contents.append(candidate_code)

    return {
        'issue_id': triplet.get("issue_id"),
        'buggy_file_name': triplet.get("buggy_file_name"),
        'stack_trace_embedding': stack_trace_embedding.tolist(),
        'file_contents': file_contents,
        'correct_file_idx': bug_location,
        'buggy_function_name': triplet.get("buggy_function_name"),
        'buggy_line_number': triplet.get("buggy_line_number")
    }


def filter_json_datasets(input_files, output_file):
    total_count = bad_invalid = kept = 0
    filtered = []

    for fp in input_files:
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found, skip")
            continue
        data = json.load(open(fp))
        total_count += len(data)

        for item in data:
            if (item['correct_file_idx'] == -1 or
                item['buggy_line_number'] == -1 or item['buggy_line_number'] == []):
                bad_invalid += 1
                continue

            filtered.append(item)
            kept += 1

    json.dump(filtered, open(output_file, "w"), indent=2)
    return dict(total_items=total_count,
                items_removed_invalid=bad_invalid,
                items_kept=kept)


def main():
    triplet_path = "../data/triplet_dataset.json"
    index_path = "../data/faiss_index_codebert/faiss_index.bin"
    metadata_path = "../data/faiss_index_codebert/faiss_metadata.json"
    output_dir = Path("../data/hierarchical_dataset")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f" Loading triplets from {triplet_path}")
    triplets = load_triplet_dataset(triplet_path)
    print(f"Loaded {len(triplets)} triplets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = CodeBERTEmbedder(
        tokenizer_path="../models/codebert/codebert_tokenizer",
        model_ckpt_path="../models/codebert/best_codebert_triplet.pt",
        device=device)

    print(f"Using CodeBERT on device: {embedder.device}")

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    pos_index = {
        (m["issue_id"], m["type"]): i
        for i, m in enumerate(metadata)
        if m["type"] == "positive"
    }

    train_bugs, temp_bugs = train_test_split(list(enumerate(triplets)), test_size=0.3, random_state=42)
    val_bugs, test_bugs = train_test_split(temp_bugs, test_size=0.5, random_state=42)
    top_k = 50

    def process_split(split_bugs, split_name):
        retrieval_hit = 0
        retrieval_total = 0
        dataset = []

        for triplet_id, triplet in tqdm(split_bugs, desc=f"Processing {split_name} bugs"):
            stack_trace_embedding = embedder.embed_stack_trace(triplet['anchor'])
            candidate_ids, _ = search_similar_files(index, stack_trace_embedding, top_k=top_k)
            bug_id = pos_index.get((triplet["issue_id"], "positive"))
            retrieval_total += 1
            bug_location = -1
            for i,cid in enumerate(candidate_ids):
                if cid == bug_id:
                    retrieval_hit += 1
                    bug_location = i
                    break
            if bug_location == -1:
                candidate_ids.append(bug_id)
                bug_location = len(candidate_ids) - 1

            entry = create_bug_dataset_entry(triplet, candidate_ids,bug_location, triplets, metadata, stack_trace_embedding)
            dataset.append(entry)

        k_count = sum(len(entry['file_contents']) == top_k for entry in dataset)
        k_plus_1_count = sum(len(entry['file_contents']) == top_k + 1 for entry in dataset)

        print(f"Candidate count stats for {split_name}:")
        print(f"    Top-{top_k} exact match count      : {k_count}")
        print(f"    Top-{top_k + 1} (added positive)    : {k_plus_1_count}")

        hit_rate = retrieval_hit / retrieval_total if retrieval_total else 0
        print(f" Top-{top_k} retrieval hit rate for {split_name}: "
              f"{retrieval_hit}/{retrieval_total} = {hit_rate:.2%}")

        json_path = output_dir / f"{split_name}.json"
        with open(json_path, "w") as f:
            json.dump(dataset, f)
        print(f"Saved {split_name} dataset with {len(dataset)} entries")

        filt_path = output_dir / f"{split_name}_filtered.json"
        stats = filter_json_datasets([json_path], filt_path)
        print(f" {split_name} filter → kept {stats['items_kept']} / "
              f"{stats['total_items']}  (removed {stats['items_removed_invalid']}) "
              f"→ {filt_path}")

    process_split(train_bugs, "train")
    process_split(val_bugs,  "val")
    process_split(test_bugs,  "test")


if __name__ == "__main__":
    main()
