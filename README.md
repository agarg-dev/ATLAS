# Bug Localization using Triplet-Loss Fine-Tuned CodeBERT

This project implements a pipeline for localizing buggy code files from stack traces using a CodeBERT model trained with triplet loss. The pipeline includes preprocessing of bug reports, generation of triplet training data, model training, hard negative mining using FAISS, and construction of a hierarchical retrieval dataset for evaluation.

---

## Project Structure and File Descriptions

### 1. Data Processing and Triplet Generation

- `parse_dataset_json.py`  
    Parses dataset from processed_samples_train_all.json file to bug_localization_dataset.json file by extracting only relevant metadata to be used for further processing.

- `extract_buggy_function_spans.py`  
  Parses the bug localization dataset and uses Git checkout to analyze Java files at specific commits. It identifies buggy functions or global spans and updates the dataset with file-level metadata.

- `filter_incomplete_bug_entries.py`  
  Filters out incomplete or malformed entries from the dataset, ensuring each sample has necessary fields such as buggy line number, file path, and fix information.

- `generate_triplet_dataset.py`  
  Uses the filtered dataset to generate training triplets of the form (stack trace, buggy code, non-buggy code). It checks out the relevant Git commit to extract file contents.

### 2. Model Training

- `train_bert.py`  
  Trains CodeBERT using triplet loss with both in-batch contrastive and explicit margin-based loss. It supports training on normal and hard negative datasets and logs metrics like similarity scores and retrieval accuracy.

### 3. Embedding and Indexing

- `build_faiss_index.py`  
  Embeds buggy (positive) and non-buggy (negative) code files using a sliding window approach and builds a FAISS index for similarity-based retrieval.

- `mine_hard_negatives.py`  
  Uses the FAISS index to find challenging negatives (semantically similar non-buggy files) for each anchor, creating a hard negative triplet dataset for improved model training.

### 4. Evaluation Dataset Construction

- `build_hierarchical_dataset.py`  
  Constructs train/val/test datasets for file-level retrieval. For each stack trace, it retrieves top-K candidate files using FAISS and marks the correct file index for evaluation.

---

## Execution Pipeline

1. Parse initial dataset:
    ```
    parse_dataset_json.py
    ```

2. Extract function information and global spans:
    ```
    python extract_buggy_function_spans.py
    ```

3. Filter the enriched dataset:
    ```
    python filter_incomplete_bug_entries.py
    ```

4. Generate triplets for training:
    ```
    python generate_triplet_dataset.py
    ```

5. Train CodeBERT on triplets:
    ```
    python train_bert.py --triplet_path ../data/triplet_dataset.json
    ```

6. Build FAISS index:
    ```
    python build_faiss_index.py
    ```

7. Mine hard negatives:
    ```
    python mine_hard_negatives.py
    ```

8. Retrain with hard negatives:
    ```
    python train_bert.py --triplet_path ../data/triplet_dataset_hardneg.json
    ```

9. Rebuild FAISS index with new model:
    ```
    python build_faiss_index.py
    ```
10. Construct hierarchical retrieval dataset:
    ```
    python build_hierarchical_dataset.py
    ```

---
