import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
import torch
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



    def embed_stack_trace(self, text, max_tokens=512, stride=510, max_chunks=12):
        
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(input_ids) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        chunks = [input_ids[i:i + max_tokens - 2] for i in range(0, len(input_ids), stride)]
        chunks = chunks[:max_chunks]

        pooled = torch.zeros(self.embedding_dim, device=self.device)
        total = 0

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
                output = self.model(**input_tensor).last_hidden_state[:, 0, :]
                pooled += output.squeeze(0)
                total += 1

            del input_tensor, output
            torch.cuda.empty_cache()

        if total == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        pooled /= total
        pooled = F.normalize(pooled.unsqueeze(0), dim=1)
        return pooled.cpu().numpy()[0].astype(np.float32)



def load_triplet_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_faiss_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)

def search_similar_files(index, query_embedding, top_k=3):
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return indices[0].tolist(), distances[0].tolist()

def mine_hard_negatives(triplets, index, metadata, embedder, top_k=3):
    issue_to_index = {
        (m["issue_id"], m["type"]): i
        for i, m in enumerate(metadata)
    }

    new_triplets = []
    for triplet in tqdm(triplets, desc="Mining hard negatives"):
        anchor = triplet["anchor"]
        positive = triplet["positive"]
        issue_id = triplet["issue_id"]
        repo_name = triplet.get("repo_name")
        buggy_file_name = triplet.get("buggy_file_name")
        buggy_function_name = triplet.get("buggy_function_name")
        buggy_line_number = triplet.get("buggy_line_number")
        path_to_buggy_file = triplet.get("path_to_buggy_file")

        query_embedding = embedder.embed_stack_trace(anchor)

        candidate_ids, _ = search_similar_files(index, query_embedding, top_k=top_k)
        correct_index = issue_to_index.get((issue_id, "positive"))

        hard_neg = None
        for cid in candidate_ids:
            if cid == correct_index:
                continue
            meta = metadata[cid]
            if meta["issue_id"] == issue_id:
                continue
            hard_neg = triplets[meta["triplet_id"]][meta["type"]]
            break

        if hard_neg:
            new_triplets.append({
                "issue_id": issue_id,
                "repo_name": repo_name,
                "buggy_file_name": buggy_file_name,
                "buggy_function_name": buggy_function_name,
                "buggy_line_number": buggy_line_number,
                "path_to_buggy_file": path_to_buggy_file,
                "anchor": anchor,
                "positive": positive,
                "negative": hard_neg
            })

    return new_triplets

def main():
    triplet_path = "../data/triplet_dataset.json"
    index_path = "../data/faiss_index_codebert/faiss_index.bin"
    metadata_path = "../data/faiss_index_codebert/faiss_metadata.json"
    output_path = "../data/triplet_dataset_hardneg.json"

    print(f"Loading triplets from {triplet_path}")
    triplets = load_triplet_dataset(triplet_path)

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)

    print(f"Loading metadata from {metadata_path}")
    metadata = load_faiss_metadata(metadata_path)

    embedder = CodeBERTEmbedder(
        tokenizer_path="../models/codebert/codebert_tokenizer",
        model_ckpt_path="../models/codebert/best_codebert_triplet.pt",
        device="cuda"
    )

    print("Starting hard negative mining...")
    hardneg_triplets = mine_hard_negatives(triplets, index, metadata, embedder)

    print(f"Mined {len(hardneg_triplets)} hard triplets. Saving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(hardneg_triplets, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
