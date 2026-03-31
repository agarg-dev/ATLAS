import time
import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from agents import (RankingFileLevelAgent, RankingFunctionLevelAgent, RankingLineLevelAgent,
                    MLPFileLevelAgent, MLPLineLevelAgent, build_flat_line_pool)
from dataembedder import extract_functions_regex, extract_global_code, get_all_function_candidates

class StandaloneEvaluator:
    """Standalone evaluator for trained HRL agents using issue-based metrics"""
    
    def __init__(self, test_data, embedder, device="cuda"):
        self.test_data = test_data
        self.embedder = embedder
        self.device = device
        
        # Build issue-to-files mapping from test dataset
        self.issue_file_mapping = self._build_issue_file_mapping(test_data)
        
    def _build_issue_file_mapping(self, dataset):
        """Build mapping from issue_id to all associated file indices"""
        issue_mapping = defaultdict(set)
        
        for sample in dataset:
            if 'issue_id' in sample:
                issue_id = sample['issue_id']
                correct_file_idx = sample['correct_file_idx']
                issue_mapping[issue_id].add(correct_file_idx)
        
        return {k: list(v) for k, v in issue_mapping.items()}
    
    def _file_belongs_to_issue(self, file_idx, sample):
        """Check if file belongs to the same issue_id"""
        if file_idx == sample['correct_file_idx']:
            return True
        
        if 'issue_id' in sample:
            current_issue_id = sample['issue_id']
            if current_issue_id in self.issue_file_mapping:
                associated_files = self.issue_file_mapping[current_issue_id]
                if file_idx in associated_files:
                    return True
        
        return False
    
    def _function_contains_issue_bug(self, func_info, sample):
        """Check if function contains any buggy line from the issue"""
        if 'buggy_function_name' in sample and func_info['name'] == sample['buggy_function_name']:
            return True
            
        if 'buggy_line_number' in sample:
            buggy_line = sample['buggy_line_number']
            if func_info['start_line'] <= buggy_line <= func_info['end_line']:
                return True
                
        if 'buggy_functions' in sample:
            for buggy_func in sample['buggy_functions']:
                if func_info['name'] == buggy_func.get('name', ''):
                    return True
                    
        if 'buggy_lines' in sample:
            for buggy_line in sample['buggy_lines']:
                if func_info['start_line'] <= buggy_line <= func_info['end_line']:
                    return True
        
        return False
    
    def _get_issue_buggy_lines_in_function(self, func_info, sample):
        """Get all buggy line indices within the function for this issue"""
        buggy_lines = []
        
        if 'buggy_line_number' in sample:
            buggy_line = sample['buggy_line_number']
            if func_info['start_line'] <= buggy_line <= func_info['end_line']:
                relative_line = buggy_line - func_info['start_line']
                if relative_line >= 0:
                    buggy_lines.append(relative_line)
        
        if 'buggy_lines' in sample:
            for buggy_line in sample['buggy_lines']:
                if func_info['start_line'] <= buggy_line <= func_info['end_line']:
                    relative_line = buggy_line - func_info['start_line']
                    if relative_line >= 0:
                        buggy_lines.append(relative_line)
        
        return buggy_lines
    
    def _get_all_issue_files(self, sample):
        """Get all file indices that belong to this issue"""
        issue_files = [sample['correct_file_idx']]
        
        if 'issue_id' in sample:
            current_issue_id = sample['issue_id']
            if current_issue_id in self.issue_file_mapping:
                issue_files = self.issue_file_mapping[current_issue_id]
        
        return issue_files
    
    def _get_all_issue_functions_in_file(self, function_info, sample):
        """Get all function indices that contain bugs for this issue"""
        issue_functions = []
        
        for idx, func_info in enumerate(function_info):
            if self._function_contains_issue_bug(func_info, sample):
                issue_functions.append(idx)
        
        return issue_functions
    
    def _get_all_issue_lines_in_function(self, function_info, sample):
        """Get all line indices that are buggy for this issue within the function"""
        if function_info.get('name') == 'global':
            return []  # Skip line eval for global code (line numbering unreliable after extraction)
        return self._get_issue_buggy_lines_in_function(function_info, sample)
    
    def compute_mrr_multi_target(self, rankings, correct_indices_list):
        """Compute MRR for multiple correct targets per sample"""
        reciprocal_ranks = []
        
        for i in range(len(rankings)):
            correct_indices = correct_indices_list[i]
            ranking = rankings[i]
            
            best_reciprocal_rank = 0.0
            
            for correct_idx in correct_indices:
                try:
                    if correct_idx in ranking:
                        position = list(ranking).index(correct_idx) + 1
                        reciprocal_rank = 1.0 / position
                        best_reciprocal_rank = max(best_reciprocal_rank, reciprocal_rank)
                except Exception:
                    continue
            
            reciprocal_ranks.append(best_reciprocal_rank)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def compute_hit_at_k(self, rankings, correct_indices_list, k):
        """Compute Hit@K for multiple correct targets per sample"""
        hits = 0
        
        for i in range(len(rankings)):
            correct_indices = correct_indices_list[i]
            ranking = rankings[i]
            
            # Check if any correct index is in top-k
            top_k_predictions = ranking[:k]
            if any(correct_idx in top_k_predictions for correct_idx in correct_indices):
                hits += 1
        
        return hits / len(rankings) if rankings else 0.0
    
    def load_agents(self, file_agent_path, func_agent_path, line_agent_path, encoder_type="lstm"):
        """Load trained agents from given paths.
        encoder_type: 'lstm' (default BiLSTM agents) or 'mlp' (MLP ablation agents)."""
        FileCls = MLPFileLevelAgent if encoder_type == "mlp" else RankingFileLevelAgent
        LineCls = MLPLineLevelAgent if encoder_type == "mlp" else RankingLineLevelAgent

        self.file_agent = FileCls(
            top_k=5, exploration_mode="epsilon_greedy", epsilon=0.0, temperature=1.0
        ).to(self.device)

        self.func_agent = RankingFunctionLevelAgent(
            top_k=15, exploration_mode="epsilon_greedy", epsilon=0.0, temperature=1.0
        ).to(self.device)

        self.line_agent = LineCls(
            top_k=15, exploration_mode="epsilon_greedy", epsilon=0.0, temperature=1.0
        ).to(self.device)

        # Load weights
        self.file_agent.load_state_dict(torch.load(file_agent_path, map_location=self.device, weights_only=True))
        self.func_agent.load_state_dict(torch.load(func_agent_path, map_location=self.device, weights_only=True))
        self.line_agent.load_state_dict(torch.load(line_agent_path, map_location=self.device, weights_only=True))

        # Set to evaluation mode
        self.file_agent.eval()
        self.func_agent.eval()
        self.line_agent.eval()

        print(f"Loaded agents from: {file_agent_path}, {func_agent_path}, {line_agent_path}")

    def load_flat_agent(self, line_agent_path):
        """Load only the line agent for flat hierarchy ablation evaluation."""
        self.file_agent = None
        self.func_agent = None
        self.line_agent = RankingLineLevelAgent(
            top_k=15, exploration_mode="epsilon_greedy", epsilon=0.0, temperature=1.0
        ).to(self.device)
        self.line_agent.load_state_dict(torch.load(line_agent_path, map_location=self.device, weights_only=True))
        self.line_agent.eval()
        print(f"Loaded flat line agent from: {line_agent_path}")
    
    def evaluate_issue_based(self):
        """Original issue-based evaluation - independent evaluation at each level"""
        print("Starting issue-based evaluation...")

        # Storage for all predictions and ground truth
        file_rankings_all = []
        file_correct_indices_all = []

        func_rankings_all = []
        func_correct_indices_all = []

        line_rankings_all = []
        line_correct_indices_all = []

        total_samples = 0
        inference_times_ms = []

        with torch.no_grad():
            for sample_idx, sample in enumerate(self.test_data):
                try:
                    sample_start = time.time()
                    print(f"Processing sample {sample_idx + 1}/{len(self.test_data)}: {sample.get('issue_id', 'N/A')}")
                    bug_emb = torch.tensor(sample['stack_trace_embedding'],
                                        dtype=torch.float, device=self.device).unsqueeze(0)
                    candidate_files = sample['file_contents']
                    candidate_file_paths = sample.get('file_paths')

                    # === FILE LEVEL EVALUATION ===
                    file_emb = self.embedder.get_file_embeddings(candidate_files, file_paths=candidate_file_paths).to(self.device).unsqueeze(0)
                    file_rankings, file_scores, file_confidences = self.file_agent(bug_emb, file_emb, training=False)
                    
                    # Get all files that belong to this issue
                    issue_files = self._get_all_issue_files(sample)
                    
                    file_rankings_all.append(file_rankings[0].cpu().numpy())
                    file_correct_indices_all.append(issue_files)
                    
                    # === FUNCTION LEVEL EVALUATION ===
                    # Evaluate functions in ALL files that belong to this issue
                    for file_idx in issue_files:
                        if file_idx < len(candidate_files):
                            selected_file_content = candidate_files[file_idx]
                            candidate_functions, function_info = get_all_function_candidates(selected_file_content)
                            
                            if candidate_functions:
                                func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                                func_rankings, func_scores, func_confidences = self.func_agent(bug_emb, func_emb, training=False)
                                
                                # Get all functions that contain bugs for this issue
                                issue_functions = self._get_all_issue_functions_in_file(function_info, sample)
                                
                                func_rankings_all.append(func_rankings[0].cpu().numpy())
                                func_correct_indices_all.append(issue_functions)
                                
                                # === LINE LEVEL EVALUATION ===
                                # Evaluate lines in ALL functions that contain bugs for this issue
                                for func_idx in issue_functions:
                                    if func_idx < len(candidate_functions):
                                        selected_function_content = candidate_functions[func_idx]
                                        candidate_lines = selected_function_content.split('\n')
                                        
                                        if candidate_lines:
                                            line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                                            line_rankings, line_scores, line_confidences = self.line_agent(bug_emb, line_emb, training=False)
                                            
                                            # Get all lines that are buggy for this issue within this function
                                            issue_lines = self._get_all_issue_lines_in_function(function_info[func_idx], sample)
                                            
                                            if issue_lines:
                                                line_rankings_all.append(line_rankings[0].cpu().numpy())
                                                line_correct_indices_all.append(issue_lines)
                    
                    total_samples += 1
                    inference_ms = (time.time() - sample_start) * 1000
                    inference_times_ms.append(inference_ms)

                    if (sample_idx + 1) % 100 == 0:
                        print(f"Processed {sample_idx + 1}/{len(self.test_data)} samples")

                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    continue

        # Print inference timing summary
        if inference_times_ms:
            times = np.array(inference_times_ms)
            mean_t = np.mean(times)
            p50 = np.percentile(times, 50)
            p95 = np.percentile(times, 95)
            print(f"\nInference timing (ms/bug): mean={mean_t:.1f}, p50={p50:.1f}, p95={p95:.1f}")

        # Compute all metrics
        metrics = {}
        
        # File metrics
        if file_rankings_all:
            metrics['file_hit_at_1'] = self.compute_hit_at_k(file_rankings_all, file_correct_indices_all, 1)
            metrics['file_hit_at_5'] = self.compute_hit_at_k(file_rankings_all, file_correct_indices_all, 5)
            metrics['file_hit_at_10'] = self.compute_hit_at_k(file_rankings_all, file_correct_indices_all, 10)
            metrics['file_mrr'] = self.compute_mrr_multi_target(file_rankings_all, file_correct_indices_all)
        
        # Function metrics
        if func_rankings_all:
            metrics['func_hit_at_1'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 1)
            metrics['func_hit_at_5'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 5)
            metrics['func_hit_at_10'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 10)
            metrics['func_hit_at_15'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 15)
            metrics['func_mrr'] = self.compute_mrr_multi_target(func_rankings_all, func_correct_indices_all)
        
        # Line metrics
        if line_rankings_all:
            metrics['line_hit_at_1'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 1)
            metrics['line_hit_at_5'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 5)
            metrics['line_hit_at_10'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 10)
            metrics['line_hit_at_15'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 15)
            metrics['line_mrr'] = self.compute_mrr_multi_target(line_rankings_all, line_correct_indices_all)
        
        # Add counts
        metrics['total_samples'] = total_samples
        metrics['file_evaluations'] = len(file_rankings_all)
        metrics['func_evaluations'] = len(func_rankings_all)
        metrics['line_evaluations'] = len(line_rankings_all)
        
        # Add evaluation type
        metrics['evaluation_type'] = "issue_based_full"

        # Add inference timing
        if inference_times_ms:
            times = np.array(inference_times_ms)
            metrics['inference_mean_ms'] = float(np.mean(times))
            metrics['inference_p50_ms'] = float(np.percentile(times, 50))
            metrics['inference_p95_ms'] = float(np.percentile(times, 95))

        return metrics
    
    def evaluate_with_ground_truth_file(self):
        """
        LLMAO baseline comparison: Use ground truth file and evaluate function/line agents
        This eliminates file-level prediction errors for fair comparison
        """
        print("Starting evaluation with ground truth files (LLMAO baseline comparison)...")
        
        # Storage for predictions and ground truth
        func_rankings_all = []
        func_correct_indices_all = []
        
        line_rankings_all = []
        line_correct_indices_all = []
        
        total_samples = 0
        
        with torch.no_grad():
            for sample_idx, sample in enumerate(self.test_data):
                try:
                    print(f"Processing sample {sample_idx + 1}/{len(self.test_data)}: {sample.get('issue_id', 'N/A')}")
                    bug_emb = torch.tensor(sample['stack_trace_embedding'], 
                                        dtype=torch.float, device=self.device).unsqueeze(0)
                    candidate_files = sample['file_contents']
                    
                    # === USE GROUND TRUTH FILE ===
                    correct_file_idx = sample['correct_file_idx']
                    
                    if correct_file_idx < len(candidate_files):
                        selected_file_content = candidate_files[correct_file_idx]
                        
                        candidate_functions, function_info = get_all_function_candidates(selected_file_content)
                        
                        if candidate_functions:
                            # === FUNCTION LEVEL EVALUATION ===
                            func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                            func_rankings, func_scores, func_confidences = self.func_agent(bug_emb, func_emb, training=False)
                            
                            # Get all functions that contain bugs for this issue
                            issue_functions = self._get_all_issue_functions_in_file(function_info, sample)
                            
                            func_rankings_all.append(func_rankings[0].cpu().numpy())
                            func_correct_indices_all.append(issue_functions)
                            
                            # === LINE LEVEL EVALUATION — use top-1 predicted function ===
                            # Matches paper LLMAO protocol: GT file → func agent → top-1
                            # predicted function → line agent (not oracle function context)
                            top1_func_idx = func_rankings[0, 0].item()
                            if top1_func_idx < len(candidate_functions):
                                selected_function_content = candidate_functions[top1_func_idx]
                                candidate_lines = selected_function_content.split('\n')

                                if candidate_lines:
                                    line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                                    line_rankings, line_scores, line_confidences = self.line_agent(bug_emb, line_emb, training=False)

                                    issue_lines = self._get_all_issue_lines_in_function(
                                        function_info[top1_func_idx], sample)
                                    if issue_lines:
                                        line_rankings_all.append(line_rankings[0].cpu().numpy())
                                        line_correct_indices_all.append(issue_lines)
                    
                    total_samples += 1
                    
                    if (sample_idx + 1) % 100 == 0:
                        print(f"Processed {sample_idx + 1}/{len(self.test_data)} samples")
                        
                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    continue
        
        # Compute metrics (only function and line metrics since file is ground truth)
        metrics = {}
        
        # Function metrics
        if func_rankings_all:
            metrics['func_hit_at_1'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 1)
            metrics['func_hit_at_5'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 5)
            metrics['func_hit_at_10'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 10)
            metrics['func_hit_at_15'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 15)
            metrics['func_mrr'] = self.compute_mrr_multi_target(func_rankings_all, func_correct_indices_all)
        
        # Line metrics
        if line_rankings_all:
            metrics['line_hit_at_1'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 1)
            metrics['line_hit_at_5'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 5)
            metrics['line_hit_at_10'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 10)
            metrics['line_hit_at_15'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 15)
            metrics['line_mrr'] = self.compute_mrr_multi_target(line_rankings_all, line_correct_indices_all)
        
        # Add counts
        metrics['total_samples'] = total_samples
        metrics['func_evaluations'] = len(func_rankings_all)
        metrics['line_evaluations'] = len(line_rankings_all)
        
        # Add evaluation type and note
        metrics['evaluation_type'] = "llmao_baseline"
        metrics['note'] = "Evaluation using ground truth files (LLMAO baseline comparison)"
        
        return metrics
    
    def evaluate_e2e_cascade(self):
        """E2E cascade evaluation matching the paper's E2E protocol (Table 3, Table 7).

        File agent predicts a ranking over all candidate files. The func agent always
        receives the top-1 predicted file's functions (regardless of whether that file
        is correct). The line agent always receives the top-1 predicted function's lines.
        Each level's metrics are computed over all samples that could be evaluated at
        that level.
        """
        print("Starting E2E cascade evaluation...")

        file_rankings_all = []
        file_correct_indices_all = []
        func_rankings_all = []
        func_correct_indices_all = []
        line_rankings_all = []
        line_correct_indices_all = []
        total_samples = 0

        with torch.no_grad():
            for sample_idx, sample in enumerate(self.test_data):
                try:
                    bug_emb = torch.tensor(sample['stack_trace_embedding'],
                                           dtype=torch.float, device=self.device).unsqueeze(0)
                    candidate_files = sample['file_contents']
                    candidate_file_paths = sample.get('file_paths')

                    # === FILE LEVEL ===
                    file_emb = self.embedder.get_file_embeddings(candidate_files, file_paths=candidate_file_paths).to(self.device).unsqueeze(0)
                    file_rankings, _, _ = self.file_agent(bug_emb, file_emb, training=False)

                    issue_files = self._get_all_issue_files(sample)
                    file_rankings_all.append(file_rankings[0].cpu().numpy())
                    file_correct_indices_all.append(issue_files)
                    total_samples += 1

                    # === FUNCTION LEVEL — always use top-1 predicted file ===
                    top1_file_idx = file_rankings[0, 0].item()
                    if top1_file_idx < len(candidate_files):
                        selected_file_content = candidate_files[top1_file_idx]
                        candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                        if candidate_functions:
                            func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                            func_rankings, _, _ = self.func_agent(bug_emb, func_emb, training=False)

                            issue_functions = self._get_all_issue_functions_in_file(function_info, sample)
                            func_rankings_all.append(func_rankings[0].cpu().numpy())
                            func_correct_indices_all.append(issue_functions)

                            # === LINE LEVEL — always use top-1 predicted function ===
                            top1_func_idx = func_rankings[0, 0].item()
                            if top1_func_idx < len(candidate_functions):
                                selected_function_content = candidate_functions[top1_func_idx]
                                candidate_lines = selected_function_content.split('\n')

                                if candidate_lines:
                                    line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                                    line_rankings, _, _ = self.line_agent(bug_emb, line_emb, training=False)

                                    issue_lines = self._get_all_issue_lines_in_function(
                                        function_info[top1_func_idx], sample)
                                    if issue_lines:
                                        line_rankings_all.append(line_rankings[0].cpu().numpy())
                                        line_correct_indices_all.append(issue_lines)

                    if (sample_idx + 1) % 100 == 0:
                        print(f"Processed {sample_idx + 1}/{len(self.test_data)} samples")

                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    continue

        metrics = {}

        if file_rankings_all:
            metrics['file_hit_at_1'] = self.compute_hit_at_k(file_rankings_all, file_correct_indices_all, 1)
            metrics['file_hit_at_5'] = self.compute_hit_at_k(file_rankings_all, file_correct_indices_all, 5)
            metrics['file_hit_at_10'] = self.compute_hit_at_k(file_rankings_all, file_correct_indices_all, 10)
            metrics['file_mrr'] = self.compute_mrr_multi_target(file_rankings_all, file_correct_indices_all)

        if func_rankings_all:
            metrics['func_hit_at_1'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 1)
            metrics['func_hit_at_5'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 5)
            metrics['func_hit_at_10'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 10)
            metrics['func_hit_at_15'] = self.compute_hit_at_k(func_rankings_all, func_correct_indices_all, 15)
            metrics['func_mrr'] = self.compute_mrr_multi_target(func_rankings_all, func_correct_indices_all)

        if line_rankings_all:
            metrics['line_hit_at_1'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 1)
            metrics['line_hit_at_5'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 5)
            metrics['line_hit_at_10'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 10)
            metrics['line_hit_at_15'] = self.compute_hit_at_k(line_rankings_all, line_correct_indices_all, 15)
            metrics['line_mrr'] = self.compute_mrr_multi_target(line_rankings_all, line_correct_indices_all)

        metrics['total_samples'] = total_samples
        metrics['file_evaluations'] = len(file_rankings_all)
        metrics['func_evaluations'] = len(func_rankings_all)
        metrics['line_evaluations'] = len(line_rankings_all)
        metrics['evaluation_type'] = "e2e_cascade"
        metrics['note'] = (
            "E2E cascade: file agent predicts → func agent gets top-1 predicted file "
            "→ line agent gets top-1 predicted function"
        )

        return metrics

    def evaluate_multi_line(self):
        """Evaluate C2C's ability to find ALL buggy lines for each issue.

        Groups test samples by issue_id. For each unique issue:
        - Collects all (file, function, line) buggy targets
        - Runs full pipeline once (same stack trace -> same predictions)
        - Reports what fraction of the issue's buggy lines were found.

        Metrics:
        - Multi-file recall: found_files / total_buggy_files per issue
        - Multi-line recall: found_lines / total_buggy_lines per issue
        """
        print("Starting multi-line evaluation...")

        # Group test samples by issue_id
        issue_samples = defaultdict(list)
        for sample in self.test_data:
            issue_id = sample.get('issue_id')
            if issue_id:
                issue_samples[issue_id].append(sample)

        file_recalls = []
        line_recalls = []

        with torch.no_grad():
            for issue_id, samples in issue_samples.items():
                try:
                    # Use first sample's stack trace (all samples in same issue share it)
                    sample0 = samples[0]
                    bug_emb = torch.tensor(sample0['stack_trace_embedding'],
                                          dtype=torch.float, device=self.device).unsqueeze(0)
                    candidate_files = sample0['file_contents']
                    candidate_file_paths = sample0.get('file_paths')

                    # Run file-level prediction
                    file_emb = self.embedder.get_file_embeddings(candidate_files, file_paths=candidate_file_paths).to(self.device).unsqueeze(0)
                    file_rankings, _, _ = self.file_agent(bug_emb, file_emb, training=False)
                    predicted_file_indices = set(file_rankings[0].cpu().numpy().tolist())

                    # Collect all buggy targets for this issue
                    buggy_file_indices = set()
                    buggy_lines = []  # list of (file_idx, line_number)
                    for s in samples:
                        fi = s['correct_file_idx']
                        buggy_file_indices.add(fi)
                        if 'buggy_line_number' in s and s['buggy_line_number'] not in (-1, []):
                            buggy_lines.append((fi, s['buggy_line_number']))

                    # Multi-file recall
                    found_files = buggy_file_indices & predicted_file_indices
                    if buggy_file_indices:
                        file_recalls.append(len(found_files) / len(buggy_file_indices))

                    # Multi-line recall: for each predicted buggy file, run func+line pipeline
                    predicted_lines = set()  # (file_idx, absolute_line_number)
                    for fi in found_files:
                        if fi >= len(candidate_files):
                            continue
                        file_content = candidate_files[fi]
                        candidate_functions, function_info = get_all_function_candidates(file_content)

                        if not candidate_functions:
                            continue

                        func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                        func_rankings, _, _ = self.func_agent(bug_emb, func_emb, training=False)
                        predicted_func_indices = func_rankings[0].cpu().numpy().tolist()

                        for func_idx in predicted_func_indices:
                            if func_idx >= len(candidate_functions):
                                continue
                            func_content = candidate_functions[func_idx]
                            func_lines = func_content.split('\n')
                            if not func_lines:
                                continue

                            line_emb = self.embedder.get_line_embeddings(func_lines).to(self.device).unsqueeze(0)
                            line_rankings, _, _ = self.line_agent(bug_emb, line_emb, training=False)
                            predicted_line_indices = line_rankings[0].cpu().numpy().tolist()

                            finfo = function_info[func_idx]
                            for li in predicted_line_indices:
                                abs_line = finfo['start_line'] + li
                                predicted_lines.add((fi, abs_line))

                    # Compute line recall for this issue
                    if buggy_lines:
                        found_count = sum(1 for (fi, ln) in buggy_lines
                                         if (fi, ln) in predicted_lines)
                        line_recalls.append(found_count / len(buggy_lines))

                except Exception as e:
                    print(f"Error processing issue {issue_id}: {e}")
                    continue

        metrics = {
            'multi_file_recall': float(np.mean(file_recalls)) if file_recalls else 0.0,
            'multi_line_recall': float(np.mean(line_recalls)) if line_recalls else 0.0,
            'num_issues_evaluated': len(issue_samples),
            'num_issues_with_file_recall': len(file_recalls),
            'num_issues_with_line_recall': len(line_recalls),
        }

        print(f"\nMulti-line evaluation results:")
        print(f"  Multi-file recall: {metrics['multi_file_recall']:.4f} ({len(file_recalls)} issues)")
        print(f"  Multi-line recall: {metrics['multi_line_recall']:.4f} ({len(line_recalls)} issues)")

        return metrics

    def evaluate_flat_line(self, max_flat_lines=300):
        """Evaluate the flat hierarchy ablation: line agent over flat pool of all candidate lines."""
        print("Starting flat line evaluation...")
        total_samples = 0
        line_top1_correct = 0
        line_top5_correct = 0
        line_top10_correct = 0
        line_top15_correct = 0
        line_rankings_list = []
        line_correct_indices_list = []

        with torch.no_grad():
            for sample_idx, sample in enumerate(self.test_data):
                try:
                    if sample['correct_file_idx'] < 0:
                        continue
                    pool_lines, _, _, correct_pool_indices = build_flat_line_pool(
                        sample, max_flat_lines=max_flat_lines
                    )
                    if not correct_pool_indices:
                        continue

                    bug_emb = torch.tensor(sample['stack_trace_embedding'],
                                           dtype=torch.float, device=self.device).unsqueeze(0)
                    line_emb = self.embedder.get_line_embeddings(pool_lines).to(self.device).unsqueeze(0)
                    line_rankings, _, _ = self.line_agent(bug_emb, line_emb, training=False)

                    ranking = line_rankings[0].cpu().numpy()
                    line_rankings_list.append(ranking)
                    line_correct_indices_list.append(correct_pool_indices)

                    if ranking[0] in correct_pool_indices:
                        line_top1_correct += 1
                    if any(r in correct_pool_indices for r in ranking[:5]):
                        line_top5_correct += 1
                    if any(r in correct_pool_indices for r in ranking[:10]):
                        line_top10_correct += 1
                    if any(r in correct_pool_indices for r in ranking[:15]):
                        line_top15_correct += 1

                    total_samples += 1
                    if (sample_idx + 1) % 50 == 0:
                        print(f"  Processed {sample_idx + 1}/{len(self.test_data)} samples")
                except Exception as e:
                    print(f"Error in flat eval sample {sample_idx}: {e}")
                    continue

        line_mrr = self.compute_mrr_multi_target(line_rankings_list, line_correct_indices_list)
        metrics = {
            'line_mrr': line_mrr,
            'line_hit_at_1': line_top1_correct / total_samples if total_samples > 0 else 0.0,
            'line_hit_at_5': line_top5_correct / total_samples if total_samples > 0 else 0.0,
            'line_hit_at_10': line_top10_correct / total_samples if total_samples > 0 else 0.0,
            'line_hit_at_15': line_top15_correct / total_samples if total_samples > 0 else 0.0,
            'total_samples': total_samples,
            'evaluation_type': 'flat_line',
        }
        print(f"Flat line evaluation complete: {metrics}")
        return metrics

    def run_both_evaluations(self):
        """
        Run all evaluation methods and return combined results.
        Includes E2E cascade (paper Table 3/7), LLMAO baseline, oracle per-level,
        and multi-line recall.
        """
        print("="*60)
        print("RUNNING ALL EVALUATIONS")
        print("="*60)

        # Run E2E cascade evaluation (paper protocol)
        print("\n1. Running E2E cascade evaluation...")
        e2e_metrics = self.evaluate_e2e_cascade()

        # Run LLMAO baseline evaluation
        print("\n2. Running LLMAO baseline evaluation (ground truth files)...")
        llmao_baseline_metrics = self.evaluate_with_ground_truth_file()

        # Run original issue-based evaluation (oracle per-level)
        print("\n3. Running oracle per-level evaluation...")
        issue_based_metrics = self.evaluate_issue_based()

        # Run multi-line evaluation
        print("\n4. Running multi-line evaluation...")
        multi_line_metrics = self.evaluate_multi_line()

        combined_results = {
            'e2e_cascade_evaluation': e2e_metrics,
            'llmao_baseline_evaluation': llmao_baseline_metrics,
            'issue_based_evaluation': issue_based_metrics,
            'multi_line_evaluation': multi_line_metrics,
        }

        return combined_results


def evaluate_trained_agents(file_agent_path, func_agent_path, line_agent_path,
                          test_data_path, device="cuda", embedder_ckpt_path=None, tokenizer_path=None, use_path=False):
    """
    Main evaluation function that loads agents and evaluates them using original method
    """
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Initialize embedder — must match checkpoint used to build stack_trace_embedding
    from dataembedder import CodeBERTEmbedder
    ckpt_kwargs = {"model_ckpt_path": embedder_ckpt_path} if embedder_ckpt_path else {}
    if tokenizer_path:
        ckpt_kwargs["tokenizer_path"] = tokenizer_path
    embedder = CodeBERTEmbedder(device=device, use_path=use_path, **ckpt_kwargs)
    
    # Initialize evaluator
    evaluator = StandaloneEvaluator(test_data, embedder, device)
    
    # Load agents
    evaluator.load_agents(file_agent_path, func_agent_path, line_agent_path)
    print("Agents loaded successfully")
    
    # Run evaluation
    print("Starting evaluation...")
    metrics = evaluator.evaluate_issue_based()
    
    return metrics


def evaluate_with_ground_truth_file(file_agent_path, func_agent_path, line_agent_path,
                                   test_data_path, device="cuda", embedder_ckpt_path=None, tokenizer_path=None, use_path=False):
    """
    Evaluation function for LLMAO baseline comparison using ground truth files
    """
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Initialize embedder — must match checkpoint used to build stack_trace_embedding
    from dataembedder import CodeBERTEmbedder
    ckpt_kwargs = {"model_ckpt_path": embedder_ckpt_path} if embedder_ckpt_path else {}
    if tokenizer_path:
        ckpt_kwargs["tokenizer_path"] = tokenizer_path
    embedder = CodeBERTEmbedder(device=device, use_path=use_path, **ckpt_kwargs)
    
    # Initialize evaluator
    evaluator = StandaloneEvaluator(test_data, embedder, device)
    
    # Load agents
    evaluator.load_agents(file_agent_path, func_agent_path, line_agent_path)
    print("Agents loaded successfully")
    
    # Run evaluation with ground truth files
    print("Starting evaluation with ground truth files...")
    metrics = evaluator.evaluate_with_ground_truth_file()
    
    return metrics


def evaluate_e2e_cascade(file_agent_path, func_agent_path, line_agent_path,
                         test_data_path, device="cuda", embedder_ckpt_path=None, tokenizer_path=None, use_path=False):
    """
    E2E cascade evaluation matching the paper's E2E protocol.
    """
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    from dataembedder import CodeBERTEmbedder
    ckpt_kwargs = {"model_ckpt_path": embedder_ckpt_path} if embedder_ckpt_path else {}
    if tokenizer_path:
        ckpt_kwargs["tokenizer_path"] = tokenizer_path
    embedder = CodeBERTEmbedder(device=device, use_path=use_path, **ckpt_kwargs)
    evaluator = StandaloneEvaluator(test_data, embedder, device)
    evaluator.load_agents(file_agent_path, func_agent_path, line_agent_path)
    return evaluator.evaluate_e2e_cascade()


def run_both_evaluations(file_agent_path, func_agent_path, line_agent_path,
                        test_data_path, device="cuda", embedder_ckpt_path=None, tokenizer_path=None, use_path=False):
    """
    Run all evaluation methods and return combined results.
    """
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Initialize embedder — must match checkpoint used to build stack_trace_embedding
    from dataembedder import CodeBERTEmbedder
    ckpt_kwargs = {"model_ckpt_path": embedder_ckpt_path} if embedder_ckpt_path else {}
    if tokenizer_path:
        ckpt_kwargs["tokenizer_path"] = tokenizer_path
    embedder = CodeBERTEmbedder(device=device, use_path=use_path, **ckpt_kwargs)
    
    # Initialize evaluator
    evaluator = StandaloneEvaluator(test_data, embedder, device)
    
    # Load agents
    evaluator.load_agents(file_agent_path, func_agent_path, line_agent_path)
    print("Agents loaded successfully")
    
    # Run both evaluations
    combined_results = evaluator.run_both_evaluations()
    
    return combined_results


def print_evaluation_results(metrics, evaluation_type=""):
    """Helper function to print evaluation results"""
    print(f"\n{evaluation_type} EVALUATION RESULTS")
    print("="*60)
    
    if 'file_hit_at_1' in metrics:
        print(f"\nFile Level Metrics:")
        print(f"  Hit@1:  {metrics.get('file_hit_at_1', 0):.4f}")
        print(f"  Hit@5:  {metrics.get('file_hit_at_5', 0):.4f}")
        print(f"  Hit@10: {metrics.get('file_hit_at_10', 0):.4f}")
        print(f"  MRR:     {metrics.get('file_mrr', 0):.4f}")
    
    print(f"\nFunction Level Metrics:")
    print(f"  Hit@1:  {metrics.get('func_hit_at_1', 0):.4f}")
    print(f"  Hit@5:  {metrics.get('func_hit_at_5', 0):.4f}")
    print(f"  Hit@10: {metrics.get('func_hit_at_10', 0):.4f}")
    print(f"  Hit@15: {metrics.get('func_hit_at_15', 0):.4f}")
    print(f"  MRR:     {metrics.get('func_mrr', 0):.4f}")
    
    print(f"\nLine Level Metrics:")
    print(f"  Hit@1:  {metrics.get('line_hit_at_1', 0):.4f}")
    print(f"  Hit@5:  {metrics.get('line_hit_at_5', 0):.4f}")
    print(f"  Hit@10: {metrics.get('line_hit_at_10', 0):.4f}")
    print(f"  Hit@15: {metrics.get('line_hit_at_15', 0):.4f}")
    print(f"  MRR:     {metrics.get('line_mrr', 0):.4f}")
    
    print(f"\nEvaluation Statistics:")
    print(f"  Total samples: {metrics.get('total_samples', 0)}")
    if 'file_evaluations' in metrics:
        print(f"  File evaluations: {metrics.get('file_evaluations', 0)}")
    print(f"  Function evaluations: {metrics.get('func_evaluations', 0)}")
    print(f"  Line evaluations: {metrics.get('line_evaluations', 0)}")
    
    if 'note' in metrics:
        print(f"\nNote: {metrics['note']}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained HRL agents")
    parser.add_argument("--file_agent_path", type=str, required=False, default="file_agent_t1b1_seed42.pt")
    parser.add_argument("--func_agent_path", type=str, required=False, default="function_agent_t1b1_seed42.pt")
    parser.add_argument("--line_agent_path", type=str, required=False, default="line_agent_t1b1_seed42.pt")
    parser.add_argument("--test_data_path", type=str, required=False, default="../../data/swebench/hierarchical_dataset/test_filtered.json")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--evaluation_type", type=str,
                        choices=["issue_based", "llmao_baseline", "e2e", "both", "flat"],
                        default="both", help="Type of evaluation to run")
    parser.add_argument("--encoder_type", type=str, choices=["lstm", "mlp"], default="lstm",
                        help="Encoder type for file/line agents (lstm default, mlp for LSTM ablation)")
    parser.add_argument("--flat_line_agent_path", type=str, default=None,
                        help="Path to flat line agent checkpoint (required when --evaluation_type flat)")
    parser.add_argument("--max_flat_lines", type=int, default=300,
                        help="Max lines in flat pool for flat hierarchy ablation (default 300)")
    parser.add_argument("--embedder_ckpt_path", type=str, default=None,
                        help="Path to CodeBERT checkpoint for on-the-fly embeddings (must match dataset's stack_trace_embedding)")
    parser.add_argument("--tokenizer_path", type=str,
                        default="../../models/codebert/codebert_tokenizer",
                        help="Path to CodeBERT tokenizer directory")
    parser.add_argument("--use_path", action="store_true",
                        help="Prepend file path tokens to code chunks (must match contrastive training)")

    args = parser.parse_args()

    import json as _json
    with open(args.test_data_path, 'r') as f:
        test_data = _json.load(f)
    print(f"Loaded {len(test_data)} test samples")

    from dataembedder import CodeBERTEmbedder
    ckpt_kwargs = {"model_ckpt_path": args.embedder_ckpt_path} if args.embedder_ckpt_path else {}
    if args.tokenizer_path:
        ckpt_kwargs["tokenizer_path"] = args.tokenizer_path
    embedder = CodeBERTEmbedder(device=args.device, use_path=args.use_path, **ckpt_kwargs)

    evaluator = StandaloneEvaluator(test_data, embedder, args.device)

    if args.evaluation_type == "flat":
        if not args.flat_line_agent_path:
            raise ValueError("--flat_line_agent_path is required when --evaluation_type flat")
        evaluator.load_flat_agent(args.flat_line_agent_path)
        metrics = evaluator.evaluate_flat_line(max_flat_lines=args.max_flat_lines)
        print_evaluation_results(metrics, "FLAT LINE")
        output_file = "flat_line_evaluation_results.json"
        with open(output_file, 'w') as f:
            _json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    elif args.evaluation_type == "issue_based":
        evaluator.load_agents(args.file_agent_path, args.func_agent_path,
                              args.line_agent_path, encoder_type=args.encoder_type)
        metrics = evaluator.evaluate_issue_based()
        print_evaluation_results(metrics, "ISSUE-BASED")
        output_file = "issue_based_evaluation_results.json"
        with open(output_file, 'w') as f:
            _json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    elif args.evaluation_type == "e2e":
        evaluator.load_agents(args.file_agent_path, args.func_agent_path,
                              args.line_agent_path, encoder_type=args.encoder_type)
        metrics = evaluator.evaluate_e2e_cascade()
        print_evaluation_results(metrics, "E2E CASCADE")
        output_file = "e2e_cascade_results.json"
        with open(output_file, 'w') as f:
            _json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    elif args.evaluation_type == "llmao_baseline":
        evaluator.load_agents(args.file_agent_path, args.func_agent_path,
                              args.line_agent_path, encoder_type=args.encoder_type)
        metrics = evaluator.evaluate_with_ground_truth_file()
        print_evaluation_results(metrics, "LLMAO BASELINE")
        output_file = "llmao_baseline_results.json"
        with open(output_file, 'w') as f:
            _json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    else:  # both
        evaluator.load_agents(args.file_agent_path, args.func_agent_path,
                              args.line_agent_path, encoder_type=args.encoder_type)
        combined_results = evaluator.run_both_evaluations()

        print_evaluation_results(combined_results['e2e_cascade_evaluation'], "E2E CASCADE")
        print_evaluation_results(combined_results['llmao_baseline_evaluation'], "LLMAO BASELINE")
        print_evaluation_results(combined_results['issue_based_evaluation'], "ORACLE PER-LEVEL")

        print("\n" + "="*60)
        print("E2E vs LLMAO COMPARISON (paper Table 3)")
        print("="*60)
        e2e_metrics = combined_results['e2e_cascade_evaluation']
        llmao_metrics = combined_results['llmao_baseline_evaluation']
        print(f"\nFunction Level:")
        print(f"  E2E   Hit@1: {e2e_metrics.get('func_hit_at_1', 0):.4f}")
        print(f"  LLMAO Hit@1: {llmao_metrics.get('func_hit_at_1', 0):.4f}")
        print(f"\nLine Level:")
        print(f"  E2E   MRR:   {e2e_metrics.get('line_mrr', 0):.4f}")
        print(f"  LLMAO MRR:   {llmao_metrics.get('line_mrr', 0):.4f}")
        print(f"  E2E   Hit@5: {e2e_metrics.get('line_hit_at_5', 0):.4f}")
        print(f"  LLMAO Hit@5: {llmao_metrics.get('line_hit_at_5', 0):.4f}")

        output_file = "combined_evaluation_results.json"
        with open(output_file, 'w') as f:
            _json.dump(combined_results, f, indent=2)
        print(f"\nCombined results saved to: {output_file}")
