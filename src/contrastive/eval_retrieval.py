"""Evaluate retrieval performance: Hit@K, Recall@K, and MRR@K per split."""

import argparse
import json
import numpy as np
import faiss
import torch
from pathlib import Path
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F
from tqdm import tqdm


class CodeBERTEmbedder:
    """Embed text using CodeBERT with sliding window."""

    def __init__(self, model_ckpt_path=None, tokenizer_path="../../models/codebert/codebert_tokenizer",
                 device=None):
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if model_ckpt_path and model_ckpt_path.lower() != "none":
            print(f"Loading fine-tuned weights from: {model_ckpt_path}")
            raw_ckpt = torch.load(model_ckpt_path, map_location=self.device, weights_only=True)
            cleaned_ckpt = {k.replace("model.", ""): v for k, v in raw_ckpt.items() if k.startswith("model.")}
            self.model.load_state_dict(cleaned_ckpt)
        else:
            print("Using base CodeBERT (no fine-tuning)")

        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def embed(self, text, max_tokens=512, stride=510, max_chunks=50):
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not input_ids:
            return np.zeros(self.embedding_dim, dtype=np.float32)

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
                cls = self.model(**input_tensor).last_hidden_state[:, 0, :]
                embeddings.append(cls.squeeze(0))

        stacked = torch.stack(embeddings)
        mean_pooled = stacked.mean(dim=0)
        return F.normalize(mean_pooled, dim=0).cpu().numpy().astype(np.float32)


def load_per_repo_indices(index_dir):
    """Load per-repo FAISS indices and metadata."""
    repo_indices = {}
    repo_metadata = {}

    for repo_dir in sorted(Path(index_dir).iterdir()):
        if not repo_dir.is_dir():
            continue
        idx_file = repo_dir / "faiss_index.bin"
        meta_file = repo_dir / "faiss_metadata.json"
        if not idx_file.exists() or not meta_file.exists():
            continue

        repo_name = repo_dir.name
        repo_indices[repo_name] = faiss.read_index(str(idx_file))
        with open(meta_file) as f:
            repo_metadata[repo_name] = json.load(f)

    return repo_indices, repo_metadata


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval Hit@K and Recall@K (Table 4)")
    parser.add_argument("--triplet_path", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True,
                        help="Directory with per-repo FAISS subdirectories")
    parser.add_argument("--model_ckpt", type=str, default="none",
                        help="Path to fine-tuned checkpoint, or 'none' for base CodeBERT")
    parser.add_argument("--tokenizer_path", type=str, default="../../models/codebert/codebert_tokenizer")
    parser.add_argument("--max_k", type=int, default=50, help="Maximum K for retrieval")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"],
                        help="Which split to evaluate (default: test). Uses 'split' field in triplets.")
    args = parser.parse_args()

    K_VALUES = [3, 5, 10, 20, 50]

    print(f"Loading triplets from {args.triplet_path}")
    with open(args.triplet_path) as f:
        triplets = json.load(f)
    print(f"Loaded {len(triplets)} triplets")

    has_split = any("split" in t for t in triplets)
    if has_split and args.split != "all":
        before = len(triplets)
        triplets = [t for t in triplets if t.get("split") == args.split]
        print(f"Filtered to {len(triplets)} {args.split}-only triplets (from {before})")

    print(f"Loading per-repo indices from {args.index_dir}")
    repo_indices, repo_metadata = load_per_repo_indices(args.index_dir)
    print(f"Loaded {len(repo_indices)} repo indices")

    ckpt = args.model_ckpt if args.model_ckpt.lower() != "none" else None
    embedder = CodeBERTEmbedder(
        model_ckpt_path=ckpt,
        tokenizer_path=args.tokenizer_path
    )
    print(f"Using device: {embedder.device}")

    # Build positive_index and issue_vectors lookups per repo
    repo_pos_index = {}
    repo_issue_vectors = {}
    for repo_name, meta in repo_metadata.items():
        pos_idx = {}
        pi = meta.get("positive_index", {})
        for key_str, vid in pi.items():
            parts = key_str.split("|||", 1)
            if len(parts) == 2:
                pos_idx[(parts[0], parts[1])] = vid
        repo_pos_index[repo_name] = pos_idx
        # issue_vectors: issue_id -> set of all correct vector IDs for that issue
        iv = meta.get("issue_vectors", {})
        repo_issue_vectors[repo_name] = {iid: set(vids) for iid, vids in iv.items()}

    # Evaluate
    hit_counts = {k: 0 for k in K_VALUES}
    total = 0
    skipped = 0
    best_ranks = []  # best rank (0-indexed) of any correct file per triplet, -1 if not found

    # For Recall@K: group by issue_id
    issue_buggy_files = defaultdict(set)   # issue_id -> set of (repo, buggy_file_name)
    issue_retrieved = defaultdict(lambda: defaultdict(set))  # issue_id -> {k: set of retrieved buggy files}

    for triplet in tqdm(triplets, desc="Evaluating retrieval"):
        repo_name = triplet.get("repo_name")
        issue_id = triplet["issue_id"]
        buggy_file = triplet["buggy_file_name"]

        if repo_name not in repo_indices:
            skipped += 1
            continue

        index = repo_indices[repo_name]
        pos_idx = repo_pos_index.get(repo_name, {})
        bug_key = (str(issue_id), buggy_file)
        bug_vid = pos_idx.get(bug_key)

        # All correct vector IDs for this issue (any buggy file counts as a hit)
        all_correct_vids = repo_issue_vectors.get(repo_name, {}).get(str(issue_id), set())
        if bug_vid is not None:
            all_correct_vids = all_correct_vids | {bug_vid}

        query_emb = embedder.embed(triplet["anchor"])
        query_vec = np.array([query_emb], dtype=np.float32)
        distances, indices = index.search(query_vec, args.max_k)
        retrieved = indices[0].tolist()

        total += 1
        issue_buggy_files[issue_id].add(bug_key)

        best_rank = next((i for i, vid in enumerate(retrieved) if vid in all_correct_vids), -1)
        best_ranks.append(best_rank)

        for k in K_VALUES:
            top_k = set(retrieved[:k])
            # Issue-level hit: any correct file for this issue found in top-K
            if all_correct_vids & top_k:
                hit_counts[k] += 1
                issue_retrieved[issue_id][k].add(bug_key)

    # Compute metrics
    found_ranks = [r for r in best_ranks if r >= 0]
    not_found = total - len(found_ranks)

    # MRR@K: reciprocal rank if found within top-K, else 0
    mrr_at_k = {}
    for k in K_VALUES:
        mrr_at_k[k] = float(np.mean([1.0 / (r + 1) if 0 <= r < k else 0.0 for r in best_ranks])) if best_ranks else 0.0
    mrr = mrr_at_k[args.max_k]  # overall MRR = MRR@max_k

    print(f"\nSkipped {skipped} triplets (repo not in index)")
    print(f"Evaluated {total} triplets\n")

    print(f"{'K':>5}  {'Hit@K':>10}  {'Recall@K':>10}  {'MRR@K':>10}")
    print("-" * 44)

    for k in K_VALUES:
        hit_rate = hit_counts[k] / total if total else 0.0

        # Recall@K: per-issue average
        recalls = []
        for issue_id, buggy_set in issue_buggy_files.items():
            found = issue_retrieved[issue_id].get(k, set())
            recalls.append(len(found) / len(buggy_set) if buggy_set else 0.0)
        avg_recall = np.mean(recalls) if recalls else 0.0

        print(f"{k:>5}  {hit_rate:>10.4f}  {avg_recall:>10.4f}  {mrr_at_k[k]:>10.4f}")

    print(f"\nMRR@{args.max_k} (issue-level): {mrr:.4f}")
    print(f"Not found in top-{args.max_k}: {not_found}/{total} ({not_found/total:.2%})" if total else "")

    # Rank histogram — shows where correct files land (to diagnose Hit@3 vs Hit@5 cliff)
    print(f"\nRank histogram (first correct file position, 0-indexed):")
    buckets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50)]
    for lo, hi in buckets:
        count = sum(1 for r in found_ranks if lo <= r < hi)
        bar = "#" * min(count * 40 // max(total, 1), 40)
        label = f"rank {lo}" if hi == lo + 1 else f"rank {lo}-{hi-1}"
        print(f"  {label:>12}: {count:>4}  {bar}")

    # Save results
    results = {
        "model_ckpt": args.model_ckpt,
        "triplet_path": args.triplet_path,
        "index_dir": args.index_dir,
        "total_triplets": total,
        "skipped": skipped,
    }
    for k in K_VALUES:
        results[f"hit_at_{k}"] = hit_counts[k] / total if total else 0.0
    for k in K_VALUES:
        recalls = []
        for issue_id, buggy_set in issue_buggy_files.items():
            found = issue_retrieved[issue_id].get(k, set())
            recalls.append(len(found) / len(buggy_set) if buggy_set else 0.0)
        results[f"recall_at_{k}"] = float(np.mean(recalls)) if recalls else 0.0
    results["mrr"] = float(mrr)
    for k in K_VALUES:
        results[f"mrr_at_{k}"] = mrr_at_k[k]
    results["not_found"] = not_found
    results["rank_histogram"] = {
        f"{lo}-{hi-1}": sum(1 for r in found_ranks if lo <= r < hi)
        for lo, hi in buckets
    }

    output_path = Path(args.index_dir) / f"eval_retrieval_results_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
