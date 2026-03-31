import os
import json
import numpy as np
import faiss
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from collections import defaultdict
import random
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

    def _sliding_window_embed(self, text, max_tokens=512, stride=510, max_chunks=50):
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


def load_per_repo_indices(index_dir):
    """Load per-repo FAISS indices, metadata, and file contents.

    Returns:
        repo_indices: {repo_name: faiss.Index}
        repo_metadata: {repo_name: metadata_dict}
        repo_contents: {repo_name: [file_content_strings]}
        repo_pos_index: {repo_name: {(issue_id, buggy_file_name): local_vector_id}}
    """
    repo_indices = {}
    repo_metadata = {}
    repo_contents = {}
    repo_pos_index = {}

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

        # Build positive_index lookup: (issue_id, buggy_file_name) -> local_vector_id
        pos_idx = {}
        pi = repo_metadata[repo_name].get("positive_index", {})
        for key_str, vid in pi.items():
            # Keys are stored as "issue_id|||buggy_file_name"
            parts = key_str.split("|||", 1)
            if len(parts) == 2:
                pos_idx[(parts[0], parts[1])] = vid
        repo_pos_index[repo_name] = pos_idx

        print(f"  Loaded {repo_name}: {repo_indices[repo_name].ntotal} vectors, "
              f"{len(pos_idx)} positive mappings")

    return repo_indices, repo_metadata, repo_contents, repo_pos_index


def search_similar_files(index, query_embedding, top_k=50):
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return indices[0].tolist(), distances[0].tolist()


def create_bug_dataset_entry(triplet, candidate_ids, bug_location, repo_contents, repo_name, stack_trace_embedding, retrieved=True, repo_metadata=None):
    """Create a hierarchical dataset entry with file contents from repo file_contents.json."""
    contents = repo_contents.get(repo_name, [])
    file_contents = []
    for cid in candidate_ids:
        if 0 <= cid < len(contents):
            file_contents.append(contents[cid])
        else:
            file_contents.append("")

    # Extract file paths from FAISS metadata (needed for path-aware RL embeddings)
    file_paths = []
    if repo_metadata is not None:
        vectors = repo_metadata.get(repo_name, {}).get("vectors", [])
        for cid in candidate_ids:
            if 0 <= cid < len(vectors):
                fp = vectors[cid].get("filepath", "")
                # Strip "[positive] " prefix added by build_faiss_index.py
                fp = fp.replace("[positive] ", "")
                file_paths.append(fp)
            else:
                file_paths.append("")

    entry = {
        'issue_id': triplet.get("issue_id"),
        'buggy_file_name': triplet.get("buggy_file_name"),
        'stack_trace_embedding': stack_trace_embedding.tolist(),
        'file_contents': file_contents,
        'file_paths': file_paths,
        'correct_file_idx': bug_location,
        'buggy_function_name': triplet.get("buggy_function_name"),
        'buggy_line_number': triplet.get("buggy_line_number"),
        'retrieved': retrieved,
    }
    # Pass through aggregated multi-line/multi-function info if present
    if 'buggy_lines' in triplet:
        entry['buggy_lines'] = triplet['buggy_lines']
    if 'buggy_functions' in triplet:
        entry['buggy_functions'] = triplet['buggy_functions']
    return entry


def filter_json_datasets(input_files, output_file):
    total_count = bad_invalid = kept = 0
    filtered = []

    for fp in input_files:
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found, skip")
            continue
        with open(fp) as f:
            data = json.load(f)
        total_count += len(data)

        for item in data:
            if (item['correct_file_idx'] == -1 or
                item['buggy_line_number'] == -1 or item['buggy_line_number'] == []):
                bad_invalid += 1
                continue

            filtered.append(item)
            kept += 1

    with open(output_file, "w") as out_f:
        json.dump(filtered, out_f, indent=2)
    return dict(total_items=total_count,
                items_removed_invalid=bad_invalid,
                items_kept=kept)


def print_split_statistics(train_bugs, val_bugs, test_bugs):
    """Print and return split statistics for the paper."""
    stats = {}
    for name, bugs in [("Train", train_bugs), ("Val", val_bugs), ("Test", test_bugs)]:
        issues = set(t["issue_id"] for _, t in bugs)
        repos = set(t["repo_name"] for _, t in bugs)
        stats[name.lower()] = {
            "issues": len(issues),
            "triplets": len(bugs),
            "repos": len(repos),
        }
        print(f"  {name:5s}: {len(issues):>4} issues, {len(bugs):>5} triplets, {len(repos):>3} repos")
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build hierarchical retrieval dataset for RL training")
    parser.add_argument("--triplet_path", type=str, default="../../data/triplet_dataset.json")
    parser.add_argument("--index_dir", type=str, default="../../data/faiss_index_codebert",
                        help="Directory containing per-repo FAISS subdirectories")
    parser.add_argument("--output_dir", type=str, default="../../data/new_hierarchical_dataset")
    parser.add_argument("--tokenizer_path", type=str, default="../../models/codebert/codebert_tokenizer")
    parser.add_argument("--model_ckpt_path", type=str, default="../../models/codebert/best_codebert_triplet.pt",
                        help="Path to fine-tuned CodeBERT checkpoint.")
    parser.add_argument("--top_k", type=int, default=50, help="Number of candidate files to retrieve per bug")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading triplets from {args.triplet_path}")
    triplets = load_triplet_dataset(args.triplet_path)
    print(f"Loaded {len(triplets)} triplets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = CodeBERTEmbedder(
        tokenizer_path=args.tokenizer_path,
        model_ckpt_path=args.model_ckpt_path,
        device=device)

    print(f"Using CodeBERT on device: {embedder.device}")

    print(f"Loading per-repo FAISS indices from {args.index_dir}")
    repo_indices, repo_metadata, repo_contents, repo_pos_index = load_per_repo_indices(args.index_dir)
    print(f"Loaded indices for {len(repo_indices)} repos")

    # issue_vectors: repo -> issue_id -> set of all correct vector IDs for that issue
    repo_issue_vectors = {
        repo_name: {iid: set(vids) for iid, vids in meta.get("issue_vectors", {}).items()}
        for repo_name, meta in repo_metadata.items()
    }

    # Issue-level split (uses "split" field from triplet dataset)
    train_bugs = [(i, t) for i, t in enumerate(triplets) if t.get("split") == "train"]
    val_bugs = [(i, t) for i, t in enumerate(triplets) if t.get("split") == "val"]
    test_bugs = [(i, t) for i, t in enumerate(triplets) if t.get("split") == "test"]

    n_train_issues = len(set(t["issue_id"] for _, t in train_bugs))
    n_val_issues = len(set(t["issue_id"] for _, t in val_bugs))
    n_test_issues = len(set(t["issue_id"] for _, t in test_bugs))
    total_split_issues = n_train_issues + n_val_issues + n_test_issues
    print(f"\nIssue-level split (no data leakage):")
    print(f"  Unique issues: {total_split_issues} total → "
          f"{n_train_issues} train / {n_val_issues} val / {n_test_issues} test")
    split_stats = print_split_statistics(train_bugs, val_bugs, test_bugs)

    # Verify no issue overlap
    train_issue_set = set(t["issue_id"] for _, t in train_bugs)
    val_issue_set   = set(t["issue_id"] for _, t in val_bugs)
    test_issue_set  = set(t["issue_id"] for _, t in test_bugs)
    assert not (train_issue_set & val_issue_set),  "Train/val issue overlap!"
    assert not (train_issue_set & test_issue_set), "Train/test issue overlap!"
    assert not (val_issue_set   & test_issue_set), "Val/test issue overlap!"

    top_k = args.top_k
    K_VALUES = [3, 5, 10, 20, 50]

    def process_split(split_bugs, split_name):
        # Deduplicate by (issue_id, buggy_file_name) but aggregate multiple
        # buggy lines/functions so the RL agent can train on all of them.
        grouped = {}
        for i, t in split_bugs:
            key = (t["issue_id"], t["buggy_file_name"])
            if key not in grouped:
                grouped[key] = (i, t, [], [])
            _, _, lines, funcs = grouped[key]
            ln = t.get("buggy_line_number")
            fn = t.get("buggy_function_name")
            if ln is not None and ln != -1 and ln not in lines:
                lines.append(ln)
            if fn and fn not in funcs:
                funcs.append(fn)

        deduped = []
        for key, (i, t, lines, funcs) in grouped.items():
            # Enrich triplet with aggregated multi-line/multi-function info
            t["buggy_lines"] = lines
            t["buggy_functions"] = [{"name": f} for f in funcs]
            deduped.append((i, t))
        if len(deduped) < len(split_bugs):
            print(f"  Deduplicated {split_name}: {len(split_bugs)} → {len(deduped)} triplets "
                  f"(aggregated buggy lines/functions)")
        split_bugs = deduped

        retrieval_hit = 0
        retrieval_total = 0
        fallback_insertions = 0
        dataset = []
        issue_hit_ranks = []  # best rank of any correct file per entry (for issue-level Hit@K)
        skipped_no_repo = 0

        for triplet_id, triplet in tqdm(split_bugs, desc=f"Processing {split_name} bugs"):
            repo_name = triplet.get("repo_name")
            if repo_name not in repo_indices:
                skipped_no_repo += 1
                continue

            index = repo_indices[repo_name]
            pos_idx = repo_pos_index.get(repo_name, {})

            stack_trace_embedding = embedder.embed_stack_trace(triplet['anchor'])
            candidate_ids, _ = search_similar_files(index, stack_trace_embedding, top_k=top_k)

            # Find the specific buggy file's vector ID (used for RL correct_file_idx)
            issue_id = triplet["issue_id"]
            bug_key = (str(issue_id), triplet["buggy_file_name"])
            bug_id = pos_idx.get(bug_key)

            # All correct vector IDs for this issue (any counts as a retrieval hit)
            all_correct_vids = repo_issue_vectors.get(repo_name, {}).get(str(issue_id), set())
            if bug_id is not None:
                all_correct_vids = all_correct_vids | {bug_id}

            retrieval_total += 1
            bug_location = -1
            best_issue_rank = -1
            for i, cid in enumerate(candidate_ids):
                if cid == bug_id and bug_location == -1:
                    bug_location = i
                if cid in all_correct_vids and best_issue_rank == -1:
                    best_issue_rank = i

            if best_issue_rank >= 0:
                retrieval_hit += 1

            is_retrieved = True
            if bug_location == -1 and bug_id is not None:
                # Correct file not retrieved: evict a random candidate and insert the correct
                # file at that position. Keeps list at exactly top_k files and avoids the
                # positional bias of always appending at the end.
                replace_pos = random.randint(0, len(candidate_ids) - 1)
                candidate_ids[replace_pos] = bug_id
                bug_location = replace_pos
                fallback_insertions += 1
                is_retrieved = False

            issue_hit_ranks.append(best_issue_rank)
            entry = create_bug_dataset_entry(
                triplet, candidate_ids, bug_location,
                repo_contents, repo_name, stack_trace_embedding,
                retrieved=is_retrieved,
                repo_metadata=repo_metadata,
            )
            dataset.append(entry)

        if skipped_no_repo > 0:
            print(f"  Skipped {skipped_no_repo} triplets (repo not in index)")

        # Multi-K hit rate reporting (issue-level: any correct file counts)
        n = len(dataset)
        if n > 0:
            print(f"\n  Retrieval Hit@K for {split_name} (n={n}):")
            for k in K_VALUES:
                hits = sum(1 for r in issue_hit_ranks if 0 <= r < k)
                print(f"    Hit@{k:<3}: {hits}/{n} = {hits/n:.2%}")

        hit_rate = retrieval_hit / retrieval_total if retrieval_total else 0
        print(f"\n  Top-{top_k} retrieval hit rate for {split_name}: "
              f"{retrieval_hit}/{retrieval_total} = {hit_rate:.2%}")
        if retrieval_total:
            print(f"  Fallback insertions (correct not retrieved): "
                  f"{fallback_insertions}/{retrieval_total} = {fallback_insertions/retrieval_total:.2%}")

        json_path = output_dir / f"{split_name}.json"
        with open(json_path, "w") as f:
            json.dump(dataset, f)
        print(f"  Saved {split_name} dataset with {len(dataset)} entries")

        filt_path = output_dir / f"{split_name}_filtered.json"
        stats = filter_json_datasets([json_path], filt_path)
        print(f"  {split_name} filter -> kept {stats['items_kept']} / "
              f"{stats['total_items']}  (removed {stats['items_removed_invalid']}) "
              f"-> {filt_path}")

    process_split(train_bugs, "train")
    process_split(val_bugs,  "val")
    process_split(test_bugs,  "test")

    # Save split statistics
    split_stats_path = output_dir / "split_statistics.json"
    with open(split_stats_path, "w") as f:
        json.dump(split_stats, f, indent=2)
    print(f"\nSplit statistics saved to {split_stats_path}")


if __name__ == "__main__":
    main()
