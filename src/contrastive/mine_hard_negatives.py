import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn.functional as F


class CodeBERTEmbedder:
    def __init__(self,
                 tokenizer_path="../../models/codebert/codebert_tokenizer",
                 model_ckpt_path="../../models/codebert/best_codebert_triplet.pt",
                 device="cuda"):

        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        base_model = RobertaModel.from_pretrained("microsoft/codebert-base")

        raw_ckpt = torch.load(model_ckpt_path, map_location=device, weights_only=True)
        cleaned_ckpt = {k.replace("model.", ""): v for k, v in raw_ckpt.items() if k.startswith("model.")}
        base_model.load_state_dict(cleaned_ckpt)

        self.model = base_model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def embed_stack_trace(self, text, max_tokens=512, stride=510, max_chunks=50):
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

        if total == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        pooled /= total
        pooled = F.normalize(pooled.unsqueeze(0), dim=1)
        return pooled.cpu().numpy()[0].astype(np.float32)


def load_triplet_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_per_repo_indices(index_dir):
    """Load per-repo FAISS indices, metadata, and file contents.

    Detects per-repo layout: if index_dir contains subdirectories with
    faiss_index.bin, loads each as a separate repo index.

    Returns:
        repo_indices: {repo_name: faiss.Index}
        repo_metadata: {repo_name: metadata_dict}
        repo_contents: {repo_name: [file_content_strings]}
    """
    repo_indices = {}
    repo_metadata = {}
    repo_contents = {}

    index_path = Path(index_dir)

    for repo_dir in sorted(index_path.iterdir()):
        if not repo_dir.is_dir():
            continue
        idx_file = repo_dir / "faiss_index.bin"
        meta_file = repo_dir / "faiss_metadata.json"
        contents_file = repo_dir / "file_contents.json"

        if not idx_file.exists() or not meta_file.exists():
            continue

        repo_name = repo_dir.name
        repo_indices[repo_name] = faiss.read_index(str(idx_file))
        with open(meta_file) as f:
            repo_metadata[repo_name] = json.load(f)
        if contents_file.exists():
            with open(contents_file) as f:
                repo_contents[repo_name] = json.load(f)
        else:
            repo_contents[repo_name] = []

        print(f"  Loaded {repo_name}: {repo_indices[repo_name].ntotal} vectors")

    return repo_indices, repo_metadata, repo_contents


def search_similar_files(index, query_embedding, top_k=10):
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return indices[0].tolist(), distances[0].tolist()


def mine_hard_negatives(triplets, repo_indices, repo_metadata, repo_contents, embedder, top_k=10, num_hard_negs=1):
    """Mine hard negatives using per-repo FAISS indices.

    For each triplet, queries the correct repo's index and collects up to
    num_hard_negs hard negatives, emitting one output triplet per negative.
    Skips candidates whose vector_id is in issue_vectors[issue_id] (same-issue positives).
    Content is looked up from repo_contents (file_contents.json).
    """
    new_triplets = []
    skipped_no_repo = 0
    skipped_no_neg = 0

    # Build per-issue positive path sets for path-based exclusion
    # (catches same-file-different-SHA that slips past vector ID check)
    issue_positive_paths = defaultdict(set)
    for t in triplets:
        pp = t.get("positive_path", "")
        if pp:
            issue_positive_paths[t["issue_id"]].add(pp)

    for triplet in tqdm(triplets, desc="Mining hard negatives"):
        anchor = triplet["anchor"]
        positive = triplet["positive"]
        issue_id = triplet["issue_id"]
        repo_name = triplet.get("repo_name")

        if repo_name not in repo_indices:
            skipped_no_repo += 1
            continue

        index = repo_indices[repo_name]
        meta = repo_metadata[repo_name]
        contents = repo_contents.get(repo_name, [])
        issue_vecs = set(meta.get("issue_vectors", {}).get(str(issue_id), []))
        excluded_paths = issue_positive_paths.get(issue_id, set())
        vectors = meta.get("vectors", [])

        query_embedding = embedder.embed_stack_trace(anchor)
        candidate_ids, _ = search_similar_files(index, query_embedding, top_k=top_k)

        hard_negs = []
        for cid in candidate_ids:
            if cid < 0:
                continue
            if cid in issue_vecs:
                continue
            # Path-based exclusion: skip candidates that are the same file
            # as any positive for this issue (catches different-SHA versions)
            if cid < len(vectors):
                cand_path = vectors[cid].get("filepath", "").replace("[positive] ", "")
                if cand_path in excluded_paths:
                    continue
            if cid < len(contents):
                content = contents[cid]
                if content and content.strip():
                    path = vectors[cid].get("filepath", "").replace("[positive] ", "") if cid < len(vectors) else ""
                    hard_negs.append((content, path))
                    if len(hard_negs) == num_hard_negs:
                        break

        if hard_negs:
            base = {
                "issue_id": issue_id,
                "repo_name": repo_name,
                "buggy_file_name": triplet.get("buggy_file_name"),
                "buggy_function_name": triplet.get("buggy_function_name"),
                "buggy_line_number": triplet.get("buggy_line_number"),
                "path_to_buggy_file": triplet.get("path_to_buggy_file"),
                "before_fix_sha": triplet.get("before_fix_sha"),
                "anchor": anchor,
                "positive": positive,
                "positive_path": triplet.get("positive_path", ""),
            }
            if "split" in triplet:
                base["split"] = triplet["split"]
            for hard_neg, hard_neg_path in hard_negs:
                new_triplets.append({**base, "negative": hard_neg, "negative_path": hard_neg_path})
        else:
            skipped_no_neg += 1

    print(f"Skipped {skipped_no_repo} triplets (repo not in index), "
          f"{skipped_no_neg} triplets (no valid hard negative found)")
    return new_triplets


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mine hard negatives using per-repo FAISS indices")
    parser.add_argument("--triplet_path", type=str, default="../../data/triplet_dataset.json")
    parser.add_argument("--index_dir", type=str, default="../../data/faiss_index_codebert",
                        help="Directory containing per-repo FAISS subdirectories")
    parser.add_argument("--output_path", type=str, default="../../data/triplet_dataset_hardneg.json")
    parser.add_argument("--tokenizer_path", type=str, default="../../models/codebert/codebert_tokenizer")
    parser.add_argument("--model_ckpt_path", type=str, default="../../models/codebert/best_codebert_triplet.pt",
                        help="Path to fine-tuned CodeBERT checkpoint.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of hard negative candidates to search")
    parser.add_argument("--num_hard_negs", type=int, default=1, help="Hard negatives to mine per triplet (default 1; use 3 for richer training)")
    args = parser.parse_args()

    print(f"Loading triplets from {args.triplet_path}")
    triplets = load_triplet_dataset(args.triplet_path)
    print(f"Loaded {len(triplets)} triplets")

    print(f"Loading per-repo FAISS indices from {args.index_dir}")
    repo_indices, repo_metadata, repo_contents = load_per_repo_indices(args.index_dir)
    print(f"Loaded indices for {len(repo_indices)} repos")

    embedder = CodeBERTEmbedder(
        tokenizer_path=args.tokenizer_path,
        model_ckpt_path=args.model_ckpt_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Starting hard negative mining...")
    hardneg_triplets = mine_hard_negatives(
        triplets, repo_indices, repo_metadata, repo_contents,
        embedder, top_k=args.top_k, num_hard_negs=args.num_hard_negs
    )

    print(f"Mined {len(hardneg_triplets)} hard triplets. Saving to: {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(hardneg_triplets, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
