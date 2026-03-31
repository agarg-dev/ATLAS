import ast
import os
import re
import json
import hashlib
import subprocess
import torch
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F


class _StripDocstrings(ast.NodeTransformer):
    """AST transformer that removes docstrings from modules, classes, and functions."""
    def _strip(self, node):
        self.generic_visit(node)
        if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
        return node
    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def filter_python_for_embedding(source):
    """Remove # comments and docstrings from Python source via AST round-trip.
    Falls back to raw source on SyntaxError (e.g. Python 2 syntax, encoding issues)."""
    try:
        tree = ast.parse(source)
        tree = _StripDocstrings().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return source


def filter_java_for_embedding(text):
    """Strip block comments, line comments, imports, package decls, and blank lines from Java source."""
    if not text:
        return text
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    lines = text.split('\n')
    filtered = []
    for line in lines:
        s = line.strip()
        if not s:                                       continue
        if s.startswith('//'):                          continue
        if s.startswith('import '):                     continue
        if s.startswith('package '):                    continue
        filtered.append(line)
    return '\n'.join(filtered)


def filter_for_embedding(text, filepath=""):
    """Route to Python or Java filter by file extension; returns text unchanged for other types."""
    if filepath.endswith(".py"):
        return filter_python_for_embedding(text)
    if filepath.endswith(".java"):
        return filter_java_for_embedding(text)
    return text


class CodeBERTEmbedder:
    """Embed long input sequences using a sliding window over CodeBERT."""

    def __init__(self, model_ckpt_path, tokenizer_path, device=None, use_path=False):
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_path = use_path

        if model_ckpt_path and model_ckpt_path.lower() != "none":
            raw_ckpt = torch.load(model_ckpt_path, map_location=self.device, weights_only=True)
            cleaned_ckpt = {k.replace("model.", ""): v for k, v in raw_ckpt.items() if k.startswith("model.")}
            self.model.load_state_dict(cleaned_ckpt)
        else:
            print("Using base CodeBERT (no fine-tuning)")

        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def sliding_window_embed(self, text: str, path: str = "", max_tokens=512, max_chunks=50) -> np.ndarray:
        """Embed text with an optional path prefix prepended to every chunk.

        Path is formatted as a comment ("# filepath") so it looks like a natural
        file header rather than a syntax error when prepended to code tokens.
        Content is filtered (docstrings/comments stripped) before this is called.
        """
        path_ids = self.tokenizer.encode(f"# {path}", add_special_tokens=False) if path else []
        content_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not content_ids:
            return np.zeros((self.embedding_dim,), dtype=np.float32)

        # max(1, ...) ensures we never produce an empty chunk even for unusually long paths.
        content_budget = max(1, max_tokens - 2 - len(path_ids))
        chunks = [content_ids[i:i + content_budget] for i in range(0, len(content_ids), content_budget)]
        chunks = chunks[:max_chunks]

        embeddings = []
        for chunk in chunks:
            ids = [self.tokenizer.cls_token_id] + path_ids + chunk + [self.tokenizer.sep_token_id]
            mask = [1] * len(ids)
            pad_len = max_tokens - len(ids)
            ids  += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len

            input_tensor = {
                "input_ids": torch.tensor([ids], device=self.device),
                "attention_mask": torch.tensor([mask], device=self.device)
            }

            with torch.no_grad():
                cls_embedding = self.model(**input_tensor).last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze(0))

        mean_pooled = torch.stack(embeddings).mean(dim=0)
        return F.normalize(mean_pooled, dim=0).cpu().numpy().astype(np.float32)


def build_rename_map(repo_path, from_sha, to_sha):
    """Detect file renames between two SHAs using git diff -M.

    Returns dict mapping old_path -> new_path for renamed files.
    Used to find replacement targets when a positive file's path doesn't
    exist at the indexed SHA because it was renamed between commits.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "-M", from_sha, to_sha],
            cwd=str(repo_path), capture_output=True, text=True, timeout=60
        )
        rename_map = {}
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0].startswith("R"):
                rename_map[parts[1]] = parts[2]
        return rename_map
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  WARNING: rename detection failed ({e})")
        return {}


def get_head_sha(repo_path):
    """Return the HEAD SHA of the repository at repo_path."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path), capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_repo_files(repo_path, sha, extension=".py"):
    """List all files at a given SHA using git ls-tree (no checkout needed).

    Returns list of (filepath, blob_hash) tuples.
    """
    result = subprocess.run(
        ["git", "ls-tree", "-r", sha],
        cwd=str(repo_path), capture_output=True, text=True
    )
    files = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        filepath = parts[1]
        if not filepath.endswith(extension):
            continue
        blob_hash = parts[0].split()[2]
        files.append((filepath, blob_hash))
    return files


def read_blob(repo_path, blob_hash):
    """Read file content from git object store (no checkout needed)."""
    result = subprocess.run(
        ["git", "cat-file", "-p", blob_hash],
        cwd=str(repo_path), capture_output=True
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        # Some files contain non-UTF-8 bytes (e.g. Latin-1 test fixtures);
        # fall back to lossy decoding so the pipeline keeps going.
        return result.stdout.decode("utf-8", errors="replace")


def content_hash(content):
    """MD5 hash of file content for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()


def load_triplet_dataset(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def build_repo_index(repo_name, repo_path, sha, triplets_for_repo, embedder, extension=".py"):
    """Build FAISS index for a single repo.

    1. List all files at `sha` via git ls-tree
    2. Read & embed each unique file (deduped by content hash)
    3. Add triplet positive files not already in repo (by content hash)
    4. Build metadata with positive_index and issue_vectors

    Returns (index, metadata_dict, file_contents_list) or None on failure.
    """
    # Step 1: Get all repo files at this SHA
    repo_files = get_repo_files(repo_path, sha, extension)
    if not repo_files:
        print(f"  WARNING: No {extension} files found for {repo_name} at {sha[:7]}")
        return None

    embeddings = []
    metadata_vectors = []
    file_contents = []
    seen_hashes = {}  # content_hash -> vector_id

    # Step 2: Embed all repo files
    print(f"  Embedding {len(repo_files)} repo files for {repo_name}...")
    for filepath, blob_hash in tqdm(repo_files, desc=f"  {repo_name} repo files", leave=False):
        file_content = read_blob(repo_path, blob_hash)
        if not file_content or not file_content.strip():
            continue

        chash = content_hash(file_content)
        if chash in seen_hashes:
            continue  # dedup by content

        vid = len(embeddings)
        seen_hashes[chash] = vid

        filtered_content = filter_for_embedding(file_content, filepath)
        emb = embedder.sliding_window_embed(filtered_content, path=filepath if embedder.use_path else "")
        embeddings.append(emb)
        metadata_vectors.append({
            "filepath": filepath,
            "content_hash": chash,
        })
        file_contents.append(file_content)  # store raw; filtered only used for embedding

    # Map positives into the index, replacing the repo-SHA version when content differs
    # to avoid near-duplicate vectors.
    positive_index = {}  # (issue_id, buggy_file_name) -> vector_id
    issue_vectors = {}   # issue_id -> [vector_ids]

    # Build filepath → vid lookup so we can replace repo versions in-place
    filepath_to_vid = {}
    for vid, meta in enumerate(metadata_vectors):
        filepath_to_vid[meta["filepath"]] = vid

    # Lazily-built rename maps for SHA pairs (triplet_sha -> {old_path: new_path})
    rename_maps = {}

    replaced_count = 0
    renamed_count = 0
    for t in triplets_for_repo:
        issue_id = str(t["issue_id"])  # normalize to str for consistent JSON keys
        buggy_file = t["buggy_file_name"]
        pos_content = t["positive"]
        pos_key = (issue_id, buggy_file)

        if pos_key in positive_index:
            # Already mapped this (issue, file) pair
            continue

        chash = content_hash(pos_content)
        if chash in seen_hashes:
            # Content already in index (from repo files or a previous positive)
            vid = seen_hashes[chash]
            positive_index[pos_key] = vid
        else:
            pos_filepath = t.get("positive_path") or f"{t.get('path_to_buggy_file', '')}/{buggy_file}"

            # Try direct path match first
            repo_vid = filepath_to_vid.get(pos_filepath)
            matched_path = pos_filepath

            # If not found, detect renames between triplet SHA and index SHA
            if repo_vid is None:
                triplet_sha = t.get("before_fix_sha", "")
                if triplet_sha and triplet_sha != sha:
                    if triplet_sha not in rename_maps:
                        rename_maps[triplet_sha] = build_rename_map(
                            repo_path, triplet_sha, sha)
                    renamed_to = rename_maps[triplet_sha].get(pos_filepath)
                    if renamed_to:
                        repo_vid = filepath_to_vid.get(renamed_to)
                        matched_path = renamed_to

            if repo_vid is not None:
                # Replace repo-SHA version with buggy (before_fix_sha) version
                old_hash = metadata_vectors[repo_vid]["content_hash"]
                if old_hash in seen_hashes and seen_hashes[old_hash] == repo_vid:
                    del seen_hashes[old_hash]
                seen_hashes[chash] = repo_vid

                filtered_pos = filter_for_embedding(pos_content, pos_filepath)
                emb = embedder.sliding_window_embed(filtered_pos, path=pos_filepath if embedder.use_path else "")
                embeddings[repo_vid] = emb
                metadata_vectors[repo_vid] = {"filepath": pos_filepath, "content_hash": chash}
                file_contents[repo_vid] = pos_content
                positive_index[pos_key] = repo_vid
                del filepath_to_vid[matched_path]  # don't replace this vid again
                if matched_path != pos_filepath:
                    renamed_count += 1
                else:
                    replaced_count += 1
            else:
                # File truly doesn't exist at repo SHA — add as new vector
                vid = len(embeddings)
                seen_hashes[chash] = vid

                filtered_pos = filter_for_embedding(pos_content, pos_filepath)
                emb = embedder.sliding_window_embed(filtered_pos, path=pos_filepath if embedder.use_path else "")
                embeddings.append(emb)
                metadata_vectors.append({
                    "filepath": pos_filepath,
                    "content_hash": chash,
                })
                file_contents.append(pos_content)  # store raw
                positive_index[pos_key] = vid

    # Build issue_vectors mapping
    for t in triplets_for_repo:
        issue_id = str(t["issue_id"])
        buggy_file = t["buggy_file_name"]
        pos_key = (issue_id, buggy_file)
        vid = positive_index.get(pos_key)
        if vid is not None:
            if issue_id not in issue_vectors:
                issue_vectors[issue_id] = []
            if vid not in issue_vectors[issue_id]:
                issue_vectors[issue_id].append(vid)

    if not embeddings:
        return None

    # Build FAISS index
    emb_array = np.array(embeddings, dtype=np.float32)
    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    # Serialize positive_index keys as strings for JSON
    positive_index_serializable = {
        f"{k[0]}|||{k[1]}": v for k, v in positive_index.items()
    }

    metadata_dict = {
        "vectors": metadata_vectors,
        "positive_index": positive_index_serializable,
        "issue_vectors": issue_vectors,
    }

    print(f"  {repo_name}: {index.ntotal} vectors ({len(repo_files)} repo files, "
          f"{len(positive_index)} positive mappings, {replaced_count} replaced, "
          f"{renamed_count} renamed, {len(seen_hashes)} unique contents)")

    return index, metadata_dict, file_contents


def rebuild_repo_index_from_cache(repo_name, cached_dir, triplets_for_repo, embedder):
    """Rebuild FAISS index by reusing file contents from a previous run.

    Loads file_contents.json and faiss_metadata.json from cached_dir/<repo>,
    re-embeds all contents with the current model, and rebuilds metadata.
    Skips all git operations — only the embedding model differs.
    Rename detection is NOT available here (no git); the initial build
    (build_repo_index) handles renames, and the cached structure is inherited.
    If rebuilding from a pre-rename-fix cache, regenerate from scratch first.

    Returns (index, metadata_dict, file_contents_list) or None on failure.
    """
    cached_path = Path(cached_dir) / repo_name
    contents_file = cached_path / "file_contents.json"
    meta_file = cached_path / "faiss_metadata.json"

    if not contents_file.exists() or not meta_file.exists():
        print(f"  WARNING: No cached data for {repo_name} in {cached_path}")
        return None

    with open(contents_file) as f:
        file_contents = json.load(f)
    with open(meta_file) as f:
        old_metadata = json.load(f)

    if not file_contents:
        return None

    # Rebuild seen_hashes and metadata_vectors from previous run's metadata.
    # old_metadata["vectors"] is a list indexed by vector_id, each entry has
    # {"filepath": ..., "content_hash": ...} — same order as file_contents.
    seen_hashes = {}
    metadata_vectors = []
    for vid, vec_meta in enumerate(old_metadata["vectors"]):
        seen_hashes[vec_meta["content_hash"]] = vid
        metadata_vectors.append(dict(vec_meta))  # copy so we can mutate

    # Build filepath → vid lookup for replacement (exclude old [positive] entries)
    filepath_to_vid = {}
    for vid, meta in enumerate(metadata_vectors):
        fp = meta["filepath"]
        if not fp.startswith("[positive]"):
            filepath_to_vid[fp] = vid

    # Process positives BEFORE re-embedding so replaced content gets the
    # correct embedding in one pass.  Same logic as build_repo_index.
    positive_index = {}
    replaced_count = 0
    for t in triplets_for_repo:
        issue_id = str(t["issue_id"])
        buggy_file = t["buggy_file_name"]
        pos_content = t["positive"]
        pos_key = (issue_id, buggy_file)

        if pos_key in positive_index:
            continue

        chash = content_hash(pos_content)
        if chash in seen_hashes:
            positive_index[pos_key] = seen_hashes[chash]
        else:
            pos_filepath = t.get("positive_path") or f"{t.get('path_to_buggy_file', '')}/{buggy_file}"
            repo_vid = filepath_to_vid.get(pos_filepath)

            if repo_vid is not None:
                # Replace repo-SHA version with buggy version
                old_hash = metadata_vectors[repo_vid]["content_hash"]
                if old_hash in seen_hashes and seen_hashes[old_hash] == repo_vid:
                    del seen_hashes[old_hash]
                seen_hashes[chash] = repo_vid
                metadata_vectors[repo_vid] = {"filepath": pos_filepath, "content_hash": chash}
                file_contents[repo_vid] = pos_content
                positive_index[pos_key] = repo_vid
                del filepath_to_vid[pos_filepath]
                replaced_count += 1
            else:
                # File not in cache — add as new entry
                vid = len(file_contents)
                seen_hashes[chash] = vid
                file_contents.append(pos_content)
                metadata_vectors.append({
                    "filepath": pos_filepath,
                    "content_hash": chash,
                })
                positive_index[pos_key] = vid

    # Re-embed all file contents (including replaced ones) with the new model
    embeddings = []
    print(f"  Re-embedding {len(file_contents)} cached files for {repo_name}...")
    for vid, fc in enumerate(tqdm(file_contents, desc=f"  {repo_name} re-embed", leave=False)):
        filepath = metadata_vectors[vid]["filepath"].replace("[positive] ", "")
        filtered_fc = filter_for_embedding(fc, filepath)
        emb = embedder.sliding_window_embed(filtered_fc, path=filepath if embedder.use_path else "")
        embeddings.append(emb)

    # Build issue_vectors mapping (identical logic to build_repo_index)
    issue_vectors = {}
    for t in triplets_for_repo:
        issue_id = str(t["issue_id"])
        buggy_file = t["buggy_file_name"]
        pos_key = (issue_id, buggy_file)
        vid = positive_index.get(pos_key)
        if vid is not None:
            if issue_id not in issue_vectors:
                issue_vectors[issue_id] = []
            if vid not in issue_vectors[issue_id]:
                issue_vectors[issue_id].append(vid)

    if not embeddings:
        return None

    emb_array = np.array(embeddings, dtype=np.float32)
    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    positive_index_serializable = {
        f"{k[0]}|||{k[1]}": v for k, v in positive_index.items()
    }

    metadata_dict = {
        "vectors": metadata_vectors,
        "positive_index": positive_index_serializable,
        "issue_vectors": issue_vectors,
    }

    print(f"  {repo_name}: {index.ntotal} vectors (from cache, "
          f"{len(positive_index)} positive mappings, {replaced_count} replaced)")

    return index, metadata_dict, file_contents


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build per-repo FAISS indices from full repository contents")
    parser.add_argument("--triplet_path", type=str, default="../../data/triplet_dataset.json")
    parser.add_argument("--output_dir", type=str, default="../../data/faiss_index_codebert")
    parser.add_argument("--repos_dir", type=str, default="../../data/repos",
                        help="Directory containing cached repo clones")
    parser.add_argument("--tokenizer_path", type=str, default="../../models/codebert/codebert_tokenizer")
    parser.add_argument("--model_ckpt_path", type=str, default="../../models/codebert/best_codebert_triplet.pt",
                        help="Path to fine-tuned CodeBERT checkpoint.")
    parser.add_argument("--extension", type=str, default=".py",
                        help="File extension to index (e.g. .py, .java)")
    parser.add_argument("--reuse_contents_from", type=str, default=None,
                        help="Path to a previous FAISS output dir whose file_contents.json "
                             "and faiss_metadata.json will be reused (skips git operations, "
                             "only re-embeds with the new model)")
    parser.add_argument("--use_path", action="store_true", default=False,
                        help="Enable path-aware file embeddings")
    args = parser.parse_args()

    print(f"Loading triplets from: {args.triplet_path}")
    triplets = load_triplet_dataset(args.triplet_path)
    print(f"Loaded {len(triplets)} triplets")

    embedder = CodeBERTEmbedder(
        model_ckpt_path=args.model_ckpt_path,
        tokenizer_path=args.tokenizer_path,
        use_path=args.use_path
    )
    print(f"Using CodeBERT on device: {embedder.device}")

    # Group triplets by repo
    from collections import defaultdict
    repo_triplets = defaultdict(list)
    for t in triplets:
        repo_triplets[t["repo_name"]].append(t)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_contents_from:
        # Reuse file contents from a previous run — skip all git operations
        cached_dir = Path(args.reuse_contents_from)
        print(f"Reusing file contents from: {cached_dir}")
        repo_names = sorted(repo_triplets.keys())
        print(f"Found {len(repo_names)} repos in triplets: {repo_names}")
    else:
        # Get HEAD SHA per repo for git-based file reading
        repos_dir_early = Path(args.repos_dir)
        repo_names_all = sorted(repo_triplets.keys())
        repo_shas = {}
        for rname in repo_names_all:
            rpath = repos_dir_early / rname
            if not rpath.exists():
                print(f"  WARNING: Repo not found at {rpath}, skipping.")
                continue
            sha = get_head_sha(rpath)
            if sha:
                repo_shas[rname] = sha
            else:
                print(f"  WARNING: Could not get HEAD SHA for {rname}, skipping.")
        print(f"Found {len(repo_shas)} repos: {sorted(repo_shas.keys())}")
        repo_names = sorted(repo_shas.keys())

    repos_dir = Path(args.repos_dir)
    total_vectors = 0

    for repo_name in repo_names:
        if args.reuse_contents_from:
            print(f"\nProcessing {repo_name} (from cache)...")
            result = rebuild_repo_index_from_cache(
                repo_name, args.reuse_contents_from,
                repo_triplets.get(repo_name, []),
                embedder
            )
        else:
            sha = repo_shas[repo_name]
            print(f"\nProcessing {repo_name} (SHA: {sha[:7]})...")
            repo_path = repos_dir / repo_name
            if not repo_path.exists():
                print(f"  WARNING: Repo not found at {repo_path}, skipping.")
                continue
            result = build_repo_index(
                repo_name, repo_path, sha,
                repo_triplets.get(repo_name, []),
                embedder, args.extension
            )

        if result is None:
            print(f"  WARNING: No index built for {repo_name}")
            continue

        index, metadata_dict, file_contents_list = result
        total_vectors += index.ntotal

        # Save per-repo outputs
        repo_out = output_dir / repo_name
        repo_out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(repo_out / "faiss_index.bin"))
        with open(repo_out / "faiss_metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        with open(repo_out / "file_contents.json", 'w') as f:
            json.dump(file_contents_list, f)

        print(f"  Saved to {repo_out}")

    print(f"\nIndexing complete. Total vectors across all repos: {total_vectors}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
