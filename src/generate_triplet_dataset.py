import os
import json
import random
import git
from pathlib import Path
from tqdm import tqdm

def read_filtered_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

def safe_checkout(repo_path, sha):
    try:
        repo = git.Repo(repo_path)
        repo.git.checkout(sha, force=True)
        return True
    except Exception as e:
        print(f"Checkout failed for {repo_path} to {sha}: {e}")
        return False

def load_file_code(repo_path, rel_path):
    full_path = os.path.join(repo_path, rel_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def get_java_files_at_sha(repo_path, exclude_file):
    java_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                if rel_path != exclude_file:
                    java_files.append(rel_path)
    return java_files

def generate_triplet_dataset(bug_dataset_path, repos_dir, output_path):
    bugs = read_filtered_dataset(bug_dataset_path)
    all_repos_dir = Path(repos_dir)

    triplets = []
    skipped = 0

    for bug in tqdm(bugs):
        try:
            bug_id = bug['issue_id']
            buggy_file_name = bug['buggy_file_name']
            buggy_function_name = bug.get('buggy_function_name', '')
            buggy_line_number = bug.get('buggy_line_number', -1)
            repo_name = bug['repo_name'].split('/')[-1]
            repo_path = all_repos_dir / repo_name
            rel_path = os.path.join(bug['path_to_buggy_file'], buggy_file_name)

            if not safe_checkout(repo_path, bug['before_fix_sha']):
                skipped += 1
                continue

            buggy_code = load_file_code(repo_path, rel_path)
            if buggy_code is None or not buggy_code.strip():
                skipped += 1
                continue

            stack_trace = bug.get('stack_trace', '').strip()
            if not stack_trace:
                skipped += 1
                continue

            java_files = get_java_files_at_sha(repo_path, exclude_file=rel_path)
            random.shuffle(java_files)

            negative_file = None
            for candidate in java_files:
                neg_code = load_file_code(repo_path, candidate)
                if neg_code and neg_code.strip():
                    negative_file = neg_code
                    break

            if not negative_file:
                skipped += 1
                continue

            triplets.append({
                "issue_id": bug_id,
                "repo_name": repo_name,
                "buggy_file_name": buggy_file_name,
                "buggy_function_name": buggy_function_name,
                "buggy_line_number": buggy_line_number,
                "path_to_buggy_file": bug['path_to_buggy_file'],
                "anchor": stack_trace,
                "positive": buggy_code,
                "negative": negative_file
            })
        except Exception as e:
            print(f"Error processing bug {bug.get('issue_id', '')}: {e}")
            skipped += 1

    print(f"Saved {len(triplets)} triplets | Skipped: {skipped}")
    with open(output_path, 'w') as f:
        json.dump(triplets, f, indent=2)

if __name__ == "__main__":
    generate_triplet_dataset(
        bug_dataset_path="../data/bug_localization_dataset_filtered.json",
        repos_dir="../data/repos",
        output_path="../data/triplet_dataset.json"
    )
