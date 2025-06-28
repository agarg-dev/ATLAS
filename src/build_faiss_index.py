import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F


class CodeBERTEmbedder:
    """Embed long input sequences using a sliding window over CodeBERT."""

    def __init__(self, model_ckpt_path, tokenizer_path, device=None):
        from transformers import RobertaTokenizer, RobertaModel
        import torch.nn.functional as F
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        raw_ckpt = torch.load(model_ckpt_path, map_location=self.device)
        cleaned_ckpt = {k.replace("model.", ""): v for k, v in raw_ckpt.items()}
        self.model.load_state_dict(cleaned_ckpt)

        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def sliding_window_embed(self, text: str, max_tokens=512, stride=510, max_chunks=12) -> np.ndarray:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not input_ids:
            return np.zeros((self.embedding_dim,), dtype=np.float32)

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
                cls_embedding = self.model(**input_tensor).last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze(0))

        stacked = torch.stack(embeddings)
        mean_pooled = stacked.mean(dim=0)
        return F.normalize(mean_pooled, dim=0).cpu().numpy().astype(np.float32)


def load_triplet_dataset(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def process_files_for_index(triplets, embedder):
    """Embed unique positives and all negatives from the dataset with rich metadata."""
    embeddings = []
    metadata = []
    seen_issue_ids = set()

    for i, triplet in enumerate(tqdm(triplets, desc="Embedding buggy and non-buggy files")):
        try:
            issue_id = triplet["issue_id"]

            if issue_id not in seen_issue_ids:
                pos_emb = embedder.sliding_window_embed(triplet["positive"])
                embeddings.append(pos_emb)
                metadata.append({
                    "type": "positive",
                    "triplet_id": i,
                    "issue_id": issue_id,
                    "repo": triplet.get("repo_name"),
                    "file": triplet.get("buggy_file_name"),
                    "function": triplet.get("buggy_function_name"),
                    "line": triplet.get("buggy_line_number"),
                    "path": triplet.get("path_to_buggy_file")
                })
                seen_issue_ids.add(issue_id)

            neg_emb = embedder.sliding_window_embed(triplet["negative"])
            embeddings.append(neg_emb)
            metadata.append({
                "type": "negative",
                "triplet_id": i,
                "issue_id": issue_id,
                "repo": triplet.get("repo_name"),
                "file": None,
                "function": None,
                "line": None,
                "path": None
            })

        except Exception as e:
            print(f"Skipping triplet {i} due to error: {e}")
            continue

    return np.array(embeddings, dtype=np.float32), metadata


def create_faiss_index(embeddings, metadata, output_dir):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))

    with open(os.path.join(output_dir, "faiss_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    print(f"Indexed {index.ntotal} vectors to: {output_dir} using cosine similarity")


def main():
    triplet_path = "../data/triplet_dataset.json"
    output_dir = "../data/faiss_index_codebert"
    tokenizer_path = "../models/codebert/codebert_tokenizer"
    model_ckpt_path = "../models/codebert/best_codebert_triplet.pt"

    print(f"Loading triplets from: {triplet_path}")
    triplets = load_triplet_dataset(triplet_path)
    print(f"Loaded {len(triplets)} triplets")

    embedder = CodeBERTEmbedder(
        model_ckpt_path=model_ckpt_path,
        tokenizer_path=tokenizer_path
    )
    print(f"Using CodeBERT on device: {embedder.device}")

    embeddings, metadata = process_files_for_index(triplets, embedder)

    print(f"Prepared {len(embeddings)} vectors for FAISS")
    create_faiss_index(embeddings, metadata, output_dir)

    print("Indexing complete.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
