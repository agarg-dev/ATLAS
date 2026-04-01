import os
import json
import argparse
import numpy as np


def top_ratio(hit_counter, label_bug_counter):
    return round(hit_counter / label_bug_counter * 100, 1)


def calculate_top_k_hits(probabilities, labels, label_name, k_values):
    """
    Calculates Top-K hits for different values of K.

    A bug is considered a "hit" for a given K if the line with the highest
    predicted probability is within ±K lines of any true buggy line.

    Parameters
    ----------
    probabilities : list[float]
        Model confidence for each line (flattened over all bugs/projects).
    labels : list[int]
        Ground-truth binary labels (1 = buggy line, 0 = clean line).
    label_name : str
        Used to pick dataset-specific `data_split` / `window_split`.
    k_values : list[int]
        A list of integers for the K values to evaluate (e.g., [1, 5, 10]).

    Returns
    -------
    dict[int, int]
        A dictionary where keys are the K values and values are the hit counts.
    """
    # ---------------- dataset-specific parameters ----------------
    if "bugsinpy" in label_name:
        data_split, window_split = 15, 15
    elif "defects4j" in label_name:
        data_split, window_split = 15, 14
    elif "devign" in label_name:
        data_split, window_split = 20, 50
    elif "beetlebox" in label_name:
        data_split, window_split = 15, 15
    else:
        raise ValueError(f"Unknown dataset in label_name: {label_name}")

    hits = {k: 0 for k in k_values}
    # ----------------------------------------------------------------

    split_probs_proj  = np.array_split(probabilities, data_split)
    split_labels_proj = np.array_split(labels,        data_split)

    for prob_project, label_project in zip(split_probs_proj, split_labels_proj):
        split_probs_bug  = np.array_split(prob_project,  window_split)
        split_labels_bug = np.array_split(label_project, window_split)

        for prob_bug, label_bug in zip(split_probs_bug, split_labels_bug):
            if sum(label_bug) == 0:
                # no buggy lines in this window – nothing to evaluate
                continue

            pred_idx       = int(np.argmax(prob_bug))       # highest-confidence line
            buggy_indices  = np.where(np.array(label_bug) == 1)[0]     # ground-truth buggy lines

            # compute min distance from prediction to any buggy line
            min_dist = np.min(np.abs(buggy_indices - pred_idx))

            for k in sorted(k_values):
                if min_dist <= k:
                    hits[k] += 1
    
    return hits


def get_per_bug_scores(probabilities, labels, label_name):
    """
    Splits flattened probabilities into a structured dictionary keyed by bug ID.

    Parameters
    ----------
    probabilities : list[float]
        Model confidence for each line (flattened over all bugs/projects).
    labels : list[int]
        Ground-truth binary labels (1 = buggy line, 0 = clean line).
    label_name : str
        Used to pick dataset-specific `data_split` / `window_split`.

    Returns
    -------
    dict[str, dict]
        A dictionary where keys are bug identifiers (e.g., "project_0_bug_0")
        and values are dictionaries with "scores" and "labels" keys.
    """
    # ---------------- dataset-specific parameters ----------------
    if "bugsinpy" in label_name:
        data_split, window_split = 15, 15
    elif "defects4j" in label_name:
        data_split, window_split = 15, 14
    elif "devign" in label_name:
        data_split, window_split = 20, 50
    elif "beetlebox" in label_name:
        data_split, window_split = 15, 15
    else:
        raise ValueError(f"Unknown dataset in label_name: {label_name}")

    scores_by_bug = {}
    # ----------------------------------------------------------------

    split_probs_proj  = np.array_split(probabilities, data_split)
    split_labels_proj = np.array_split(labels,        data_split)

    for proj_idx, (prob_project, label_project) in enumerate(zip(split_probs_proj, split_labels_proj)):
        split_probs_bug  = np.array_split(prob_project,  window_split)
        split_labels_bug = np.array_split(label_project, window_split)

        for bug_idx, (prob_bug, label_bug) in enumerate(zip(split_probs_bug, split_labels_bug)):
            if sum(label_bug) == 0:
                # no buggy lines in this window, can optionally skip
                continue

            bug_id = f"project_{proj_idx}_bug_{bug_idx}"
            scores_by_bug[bug_id] = {"scores": prob_bug, "labels": label_bug}

    return scores_by_bug


def results(log_path, data_name, codegen_size, k_values=None):
    if k_values is None:
        k_values = [1, 3, 5, 10]
    total_hits = {k: 0 for k in k_values}
    total_bugs = 0
    data_log_path = f"{log_path}/{data_name}"
    print(f"[DEBUG] Starting results function with log_path={log_path}, data_name={data_name}, codegen_size={codegen_size}")
    print(f"[DEBUG] data_log_path: {data_log_path}")
    
    for subdir, _, files in os.walk(data_log_path):
        print(f"[DEBUG] Processing subdirectory: {subdir}")
        print(f"[DEBUG] Found {len(files)} files in subdirectory")
        
        # iterate through all json files within each sub-directory
        for file in files:
            print(f"[DEBUG] Checking file: {file}")
            if not file.endswith(".json"):
                print(f"[DEBUG] Skipping non-JSON file: {file}")
                continue
            
            file_path = os.path.join(subdir, file)
            print(f"[DEBUG] Opening JSON file: {file_path}")
            f = open(file_path)
            print(file)

            # Use os.path.sep for cross-platform path separator handling
            subdir_name = subdir.replace(data_log_path + os.path.sep, "").replace(data_log_path + "/", "")
            print(f"[DEBUG] subdir_name after replacement: {subdir_name}")

            split_dir = subdir_name.split("_")
            print(f"[DEBUG] split_dir: {split_dir}")
            
            # Handle the case where the path parsing fails due to Windows path separators
            if len(split_dir) < 3:
                print(f"[DEBUG] Insufficient split_dir parts, skipping file")
                f.close()
                continue
                
            parsed_data_name = split_dir[0]
            params = split_dir[1]
            dimension = split_dir[2]
            print(f"[DEBUG] Parsed - data_name: {parsed_data_name}, params: {params}, dimension: {dimension}")

            # if params == "256" or params == "512" or params == "1024":
            #     params = "scratch"
            #     print(f"[DEBUG] Changed params to 'scratch'")

            # print(f"[DEBUG] Comparing codegen_size '{codegen_size}' with dimension '{dimension}'")
            # if codegen_size != dimension:
            #     print(f"[DEBUG] Skipping due to codegen_size mismatch")
            #     f.close()
            #     continue

            # if "scratch" in dimension:
            #     dimension = ""
            #     print(f"[DEBUG] Cleared dimension due to 'scratch'")
            
            print(f"[DEBUG] Loading JSON data from file")
            data = json.load(f)
            probabilities = data["prob"]
            labels = data["label"]
            print(f"[DEBUG] Loaded {len(probabilities)} probabilities and {len(labels)} labels")
            f.close()
            
            filtered_prob = []
            filtered_label = []
            for i, prob in enumerate(probabilities):
                if prob != 0:
                    filtered_prob.append(prob)
                    filtered_label.append(labels[i])
            print(f"[DEBUG] Filtered to {len(filtered_prob)} non-zero probabilities")
            
            label_name = f"{parsed_data_name}-{params}".replace("--", "-")
            print(f"[DEBUG] Generated label_name: {label_name}")
            print(parsed_data_name)
            
            print(f"[DEBUG] Calling calculate_top_k_hits function with K={k_values}")
            hits = calculate_top_k_hits(probabilities, labels, label_name, k_values)
            
            for k, count in hits.items():
                total_hits[k] += count
            
            print(f"[DEBUG] Running totals: {total_hits}")

            # New code to get and print per-bug scores
            print(f"[INFO] Getting per-bug scores for {label_name}")
            per_bug_scores = get_per_bug_scores(probabilities, labels, label_name)
            print(f"--- Per-Bug/File Scores for {os.path.join(subdir, file)} ---")
            for bug_id, data in per_bug_scores.items():
                scores = data["scores"]
                bug_labels = data["labels"]
                true_bug_indices = np.where(np.array(bug_labels) == 1)[0]

                print(f"  {bug_id} (True bug lines at: {true_bug_indices}):")
                
                # We only print the top 5 scores for brevity
                top_5_indices = np.argsort(scores)[-5:][::-1]
                top_5_scores = scores[top_5_indices]

                for i in range(len(top_5_indices)):
                    line_idx = top_5_indices[i]
                    is_correct_hit = "<- HIT" if line_idx in true_bug_indices else ""
                    print(f"    - Line {line_idx: <4}: {top_5_scores[i]:.4f} {is_correct_hit}")
            print(f"--- End Per-Bug/File Scores ---")

            if "bugsinpy" in label_name:
                total_bugs = 493
                print(f"[DEBUG] Set total_bugs to {total_bugs} for bugsinpy")
            elif "defects4j-1.2.0" in label_name:
                total_bugs = 226
                print(f"[DEBUG] Set total_bugs to {total_bugs} for defects4j-1.2.0")
            elif "defects4j" in label_name:
                total_bugs = 395
                print(f"[DEBUG] Set total_bugs to {total_bugs} for defects4j")
            elif "devign" in label_name:
                total_bugs = 5260
                print(f"[DEBUG] Set total_bugs to {total_bugs} for devign")
            elif "beetlebox" in label_name:
                total_bugs = 498
                print(f"[DEBUG] Set total_bugs to {total_bugs} for beetlebox")
            else:
                print(f"[DEBUG] No matching dataset found for label_name: {label_name}")
    
    print(f"[DEBUG] Final total_bugs: {total_bugs}")
    print(total_bugs)
    if total_bugs:
        print(f"[DEBUG] Generating final output with total_bugs={total_bugs}")
        
        # Build the output string for all K values
        output_parts = []
        for k in sorted(total_hits.keys()):
            hits_count = total_hits[k]
            ratio = top_ratio(hits_count, total_bugs)
            output_parts.append(f"Top-{k}: {hits_count} ({ratio}%)")
        
        print(f"Results for {total_bugs} total bugs for {data_name}-{codegen_size}:")
        print(" | ".join(output_parts))

    else:
        print(f"[DEBUG] No output generated because total_bugs is 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="Path to data root")
    ap.add_argument("pretrain_type", help="Pretrain size")
    ap.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="K",
        help="K values to evaluate (default: 1 3 5 10).",
    )
    ap.add_argument(
        "--data-name",
        default="beetlebox",
        help="Dataset name (default: beetlebox).",
    )

    args = ap.parse_args()
    log_path = args.log_path
    pretrain_type = args.pretrain_type
    data_name = args.data_name

    # Do not override log_path provided via CLI; uncomment the following lines only if you
    # want to fall back to a default relative path when no CLI argument is supplied.
    # current_path = os.getcwd()
    # log_path = os.path.join(current_path, "logs_path")

    results(log_path, data_name, pretrain_type, k_values=sorted(set(args.k_values)))
