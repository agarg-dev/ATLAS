import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataembedder import extract_functions_regex, extract_global_code, get_all_function_candidates
import random
import math
from collections import defaultdict
import wandb
import os


def fix_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _rankings_with_exploration(scores, num_candidates, top_k, exploration_mode, epsilon, temperature, device):
    """Shared exploration logic for all agent types."""
    batch_size = scores.size(0)
    rankings = []
    for b in range(batch_size):
        s = scores[b]
        if exploration_mode == "epsilon_greedy":
            if random.random() < epsilon:
                idx = torch.arange(num_candidates, device=device)
                ranking = idx[torch.randperm(num_candidates, device=device)][:top_k]
            else:
                _, ranking = torch.topk(s, k=min(top_k, num_candidates))
        elif exploration_mode == "boltzmann":
            probs = F.softmax(s / temperature, dim=0)
            ranking = torch.multinomial(probs, min(top_k, num_candidates), replacement=False)
        else:
            _, ranking = torch.topk(s, k=min(top_k, num_candidates))
        rankings.append(ranking)
    return torch.stack(rankings)


def _greedy_rankings(scores, num_candidates, top_k):
    """Shared greedy ranking for all agent types."""
    _, rankings = torch.topk(scores, k=min(top_k, num_candidates), dim=1)
    return rankings


class RankingFileLevelAgent(nn.Module):
    """File-level agent without attention"""
    def __init__(self, bug_emb_dim=768, file_emb_dim=768, lstm_hidden_dim=128,
                 mlp_hidden_dim=128, top_k=5, 
                 exploration_mode="epsilon_greedy", epsilon=0.1, temperature=1.0):
        super().__init__()
        self.bug_emb_dim = bug_emb_dim
        self.file_emb_dim = file_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.top_k = top_k
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon
        self.temperature = temperature
        
        # File processing LSTM
        self.file_lstm = nn.LSTM(file_emb_dim, lstm_hidden_dim, 
                                batch_first=True, bidirectional=True)
        self.lstm_out_dim = lstm_hidden_dim * 2
        
        # Feature fusion (no attention)
        self.feature_fusion = nn.Sequential(
            nn.Linear(bug_emb_dim + self.lstm_out_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )
        
        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, bug_emb, file_emb, training=True):
        batch_size, num_files, _ = file_emb.shape
        device = file_emb.device
        
        # Process files through LSTM
        file_features, _ = self.file_lstm(file_emb)
        
        # Simple feature fusion without attention
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_files, -1)
        combined = torch.cat([bug_expanded, file_features], dim=-1)
        fused_features = self.feature_fusion(combined)
        
        # Compute scores
        scores = self.ranking_head(fused_features).squeeze(-1)
        confidences = self.confidence_head(fused_features).squeeze(-1)
        
        # Device-safe ranking with exploration
        if training:
            rankings = self._get_rankings_with_exploration(scores, num_files, device)
        else:
            rankings = self._get_greedy_rankings(scores, num_files)
        
        return rankings, scores, confidences
    
    def _get_rankings_with_exploration(self, scores, num_candidates, device):
        return _rankings_with_exploration(scores, num_candidates, self.top_k,
                                         self.exploration_mode, self.epsilon, self.temperature, device)

    def _get_greedy_rankings(self, scores, num_candidates):
        return _greedy_rankings(scores, num_candidates, self.top_k)


class MLPFileLevelAgent(nn.Module):
    """File-level agent with MLP encoder instead of BiLSTM (LSTM ablation variant)."""
    def __init__(self, bug_emb_dim=768, file_emb_dim=768, lstm_hidden_dim=128,
                 mlp_hidden_dim=128, top_k=5,
                 exploration_mode="epsilon_greedy", epsilon=0.1, temperature=1.0):
        super().__init__()
        self.bug_emb_dim = bug_emb_dim
        self.file_emb_dim = file_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.top_k = top_k
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon
        self.temperature = temperature

        # MLP encoder replaces BiLSTM; output dim = lstm_hidden_dim * 2 = 256
        self.lstm_out_dim = lstm_hidden_dim * 2
        self.file_encoder = nn.Sequential(
            nn.Linear(file_emb_dim, self.lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim),
            nn.ReLU()
        )

        # Feature fusion identical to RankingFileLevelAgent
        self.feature_fusion = nn.Sequential(
            nn.Linear(bug_emb_dim + self.lstm_out_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )

        self.ranking_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, bug_emb, file_emb, training=True):
        batch_size, num_files, _ = file_emb.shape
        device = file_emb.device

        file_features = self.file_encoder(file_emb)

        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_files, -1)
        combined = torch.cat([bug_expanded, file_features], dim=-1)
        fused_features = self.feature_fusion(combined)

        scores = self.ranking_head(fused_features).squeeze(-1)
        confidences = self.confidence_head(fused_features).squeeze(-1)

        if training:
            rankings = self._get_rankings_with_exploration(scores, num_files, device)
        else:
            rankings = self._get_greedy_rankings(scores, num_files)

        return rankings, scores, confidences

    def _get_rankings_with_exploration(self, scores, num_candidates, device):
        return _rankings_with_exploration(scores, num_candidates, self.top_k,
                                         self.exploration_mode, self.epsilon, self.temperature, device)

    def _get_greedy_rankings(self, scores, num_candidates):
        return _greedy_rankings(scores, num_candidates, self.top_k)


class RankingFunctionLevelAgent(nn.Module):
    """Function-level agent without attention"""
    
    def __init__(self, bug_emb_dim=768, func_emb_dim=768, mlp_hidden_dim=128, 
                 top_k=5, exploration_mode="epsilon_greedy", 
                 epsilon=0.1, temperature=1.0):
        super().__init__()
        self.bug_emb_dim = bug_emb_dim
        self.func_emb_dim = func_emb_dim
        self.top_k = top_k
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon
        self.temperature = temperature
        
        # Function processing
        self.func_processor = nn.Sequential(
            nn.Linear(func_emb_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )
        
        # Feature fusion (no attention)
        self.feature_fusion = nn.Sequential(
            nn.Linear(bug_emb_dim + mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )
        
        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, bug_emb, func_emb, training=True):
        batch_size, num_funcs, _ = func_emb.shape
        device = func_emb.device
        
        # Process functions
        func_features = self.func_processor(func_emb)
        
        # Simple feature fusion without attention
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_funcs, -1)
        combined = torch.cat([bug_expanded, func_features], dim=-1)
        fused_features = self.feature_fusion(combined)
        
        # Compute scores
        scores = self.ranking_head(fused_features).squeeze(-1)
        confidences = self.confidence_head(fused_features).squeeze(-1)
        
        # Device-safe ranking
        if training:
            rankings = self._get_rankings_with_exploration(scores, num_funcs, device)
        else:
            rankings = self._get_greedy_rankings(scores, num_funcs)
        
        return rankings, scores, confidences
    
    def _get_rankings_with_exploration(self, scores, num_candidates, device):
        return _rankings_with_exploration(scores, num_candidates, self.top_k,
                                         self.exploration_mode, self.epsilon, self.temperature, device)

    def _get_greedy_rankings(self, scores, num_candidates):
        return _greedy_rankings(scores, num_candidates, self.top_k)

class RankingLineLevelAgent(nn.Module):
    """Line-level agent without attention"""
    
    def __init__(self, bug_emb_dim=768, line_emb_dim=768, lstm_hidden_dim=128,
                 mlp_hidden_dim=128, top_k=5, max_span_length=5, 
                 exploration_mode="epsilon_greedy", epsilon=0.1, temperature=1.0):
        super().__init__()
        self.bug_emb_dim = bug_emb_dim
        self.line_emb_dim = line_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.top_k = top_k
        self.max_span_length = max_span_length
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon
        self.temperature = temperature
        
        # Line processing LSTM
        self.line_lstm = nn.LSTM(
            line_emb_dim, lstm_hidden_dim, 
            batch_first=True, bidirectional=True
        )
        self.lstm_out_dim = lstm_hidden_dim * 2
        
        # Sinusoidal positional encoding
        self.register_buffer('pos_encoding_cache', torch.zeros(1, 10000, lstm_hidden_dim))
        self._init_positional_encoding()
        
        # Feature fusion (no attention)
        self.feature_fusion = nn.Sequential(
            nn.Linear(bug_emb_dim + self.lstm_out_dim + lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )
        
        # Line ranking head
        self.line_ranking_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _init_positional_encoding(self):
        """Initialize sinusoidal positional encoding"""
        position = torch.arange(10000).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.lstm_hidden_dim, 2).float() * 
                            -(math.log(10000.0) / self.lstm_hidden_dim))
        
        pe = torch.zeros(10000, self.lstm_hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding_cache[0] = pe
    
    def forward(self, bug_emb, line_emb, training=True):
        batch_size, num_lines, _ = line_emb.shape
        device = line_emb.device
        
        # Safe positional encoding
        pos_emb = self.pos_encoding_cache[0, :num_lines].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process lines
        line_features, _ = self.line_lstm(line_emb)
        
        # Feature fusion without attention
        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_lines, -1)
        combined = torch.cat([bug_expanded, line_features, pos_emb], dim=-1)
        fused_features = self.feature_fusion(combined)
        
        # Compute scores
        line_scores = self.line_ranking_head(fused_features).squeeze(-1)
        confidences = self.confidence_head(fused_features).squeeze(-1)
        
        # Device-safe ranking
        if training:
            line_rankings = self._get_rankings_with_exploration(line_scores, num_lines, device)
        else:
            line_rankings = self._get_greedy_rankings(line_scores, num_lines)
        
        return line_rankings, line_scores, confidences
    
    def _get_rankings_with_exploration(self, scores, num_candidates, device):
        return _rankings_with_exploration(scores, num_candidates, self.top_k,
                                         self.exploration_mode, self.epsilon, self.temperature, device)

    def _get_greedy_rankings(self, scores, num_candidates):
        return _greedy_rankings(scores, num_candidates, self.top_k)


class MLPLineLevelAgent(nn.Module):
    """Line-level agent with MLP encoder instead of BiLSTM (LSTM ablation variant).
    Keeps sinusoidal positional encoding — independent of LSTM."""
    def __init__(self, bug_emb_dim=768, line_emb_dim=768, lstm_hidden_dim=128,
                 mlp_hidden_dim=128, top_k=5, max_span_length=5,
                 exploration_mode="epsilon_greedy", epsilon=0.1, temperature=1.0):
        super().__init__()
        self.bug_emb_dim = bug_emb_dim
        self.line_emb_dim = line_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim  # must be set BEFORE _init_positional_encoding()
        self.top_k = top_k
        self.max_span_length = max_span_length
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon
        self.temperature = temperature

        # MLP encoder replaces BiLSTM; output dim = lstm_hidden_dim * 2 = 256
        self.lstm_out_dim = lstm_hidden_dim * 2
        self.line_encoder = nn.Sequential(
            nn.Linear(line_emb_dim, self.lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim),
            nn.ReLU()
        )

        # Sinusoidal positional encoding identical to RankingLineLevelAgent
        self.register_buffer('pos_encoding_cache', torch.zeros(1, 10000, lstm_hidden_dim))
        self._init_positional_encoding()

        # Feature fusion identical to RankingLineLevelAgent: 768+256+128=1152
        self.feature_fusion = nn.Sequential(
            nn.Linear(bug_emb_dim + self.lstm_out_dim + lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )

        self.line_ranking_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def _init_positional_encoding(self):
        """Initialize sinusoidal positional encoding"""
        position = torch.arange(10000).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.lstm_hidden_dim, 2).float() *
                            -(math.log(10000.0) / self.lstm_hidden_dim))
        pe = torch.zeros(10000, self.lstm_hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding_cache[0] = pe

    def forward(self, bug_emb, line_emb, training=True):
        batch_size, num_lines, _ = line_emb.shape
        device = line_emb.device

        pos_emb = self.pos_encoding_cache[0, :num_lines].unsqueeze(0).expand(batch_size, -1, -1)

        line_features = self.line_encoder(line_emb)

        bug_expanded = bug_emb.unsqueeze(1).expand(-1, num_lines, -1)
        combined = torch.cat([bug_expanded, line_features, pos_emb], dim=-1)
        fused_features = self.feature_fusion(combined)

        line_scores = self.line_ranking_head(fused_features).squeeze(-1)
        confidences = self.confidence_head(fused_features).squeeze(-1)

        if training:
            line_rankings = self._get_rankings_with_exploration(line_scores, num_lines, device)
        else:
            line_rankings = self._get_greedy_rankings(line_scores, num_lines)

        return line_rankings, line_scores, confidences

    def _get_rankings_with_exploration(self, scores, num_candidates, device):
        return _rankings_with_exploration(scores, num_candidates, self.top_k,
                                         self.exploration_mode, self.epsilon, self.temperature, device)

    def _get_greedy_rankings(self, scores, num_candidates):
        return _greedy_rankings(scores, num_candidates, self.top_k)


def build_flat_line_pool(sample, max_flat_lines=300):
    """Build a flat pool of lines from all candidate files for the flat hierarchy ablation.
    Buggy file is prepended first to guarantee correct lines are present in the pool.

    Returns:
        lines: list of line text strings
        pool_file_indices: list of file indices (parallel to lines)
        pool_line_numbers: list of 1-indexed line numbers within each file (parallel to lines)
        correct_pool_indices: list of pool positions (0-indexed) that are buggy lines
    """
    correct_file_idx = sample['correct_file_idx']
    file_contents = sample['file_contents']
    buggy_line_set = set(sample.get('buggy_lines', []))
    if sample.get('buggy_line_number') not in (-1, None):
        buggy_line_set.add(sample['buggy_line_number'])

    lines, pool_file_indices, pool_line_numbers = [], [], []

    # Prepend buggy file lines first (guarantees correct lines in pool)
    if 0 <= correct_file_idx < len(file_contents):
        for ln, text in enumerate(file_contents[correct_file_idx].split('\n'), start=1):
            lines.append(text)
            pool_file_indices.append(correct_file_idx)
            pool_line_numbers.append(ln)

    # Fill remaining slots from other files up to max_flat_lines
    for fi, content in enumerate(file_contents):
        if fi == correct_file_idx:
            continue
        if len(lines) >= max_flat_lines:
            break
        for ln, text in enumerate(content.split('\n'), start=1):
            if len(lines) >= max_flat_lines:
                break
            lines.append(text)
            pool_file_indices.append(fi)
            pool_line_numbers.append(ln)

    correct_pool_indices = [
        p for p, (fi, ln) in enumerate(zip(pool_file_indices, pool_line_numbers))
        if fi == correct_file_idx and ln in buggy_line_set
    ]
    return lines, pool_file_indices, pool_line_numbers, correct_pool_indices


class IssueBasedRewardCalculator:
    """Issue-based reward calculator with sparse and intermediate rewards"""
    
    def __init__(self, reward_type="intermediate", position_weights=None, dataset=None,
                 wrong_item_penalty=0.1):
        """
        Args:
            reward_type: "sparse" or "intermediate"
            position_weights: Weights for different ranking positions
            dataset: Training dataset to build issue-to-files mapping
            wrong_item_penalty: Negative reward for wrong items per ranked position in intermediate mode (0 to disable)
        """
        self.reward_type = reward_type
        self.position_weights = position_weights or [1.0, 0.8, 0.6, 0.4, 0.2]
        self.wrong_item_penalty = wrong_item_penalty
        
        # Build issue-to-files mapping from dataset
        self.issue_file_mapping = {}
        if dataset:
            self.issue_file_mapping = self._build_issue_file_mapping(dataset)
    
    def _build_issue_file_mapping(self, dataset):
        """Build mapping from issue_id to all associated file indices across all samples"""
        issue_mapping = defaultdict(set)
        
        for sample in dataset:
            if 'issue_id' in sample:
                issue_id = sample['issue_id']
                correct_file_idx = sample['correct_file_idx']
                issue_mapping[issue_id].add(correct_file_idx)
        
        # Convert sets to lists for easier handling
        return {k: list(v) for k, v in issue_mapping.items()}
    
    def _file_belongs_to_issue(self, file_idx, sample):
        """Check if the selected file belongs to the issue using issue_id"""
        # Primary check: exact correct file
        if file_idx == sample['correct_file_idx']:
            return True
        
        # Issue-based check: if file belongs to same issue_id
        if 'issue_id' in sample:
            current_issue_id = sample['issue_id']
            
            # Check if this file_idx is associated with the same issue_id
            if current_issue_id in self.issue_file_mapping:
                associated_files = self.issue_file_mapping[current_issue_id]
                if file_idx in associated_files:
                    return True
        
        # Fallback: check for explicit issue-related files list (if exists in your data)
        if 'issue_related_files' in sample:
            return file_idx in sample['issue_related_files']
        
        return False
    
    def compute_file_reward(self, predicted_rankings, samples_info, individual=False):
        """Compute file-level rewards based on issue_id matching.
        individual: when True, sparse returns 1.0 per-level (for TWF/T1B1-RL);
                    when False, sparse returns 0.0 (reward only at final step, for VT/PT1)."""
        batch_size = predicted_rankings.size(0)
        device = predicted_rankings.device
        rewards = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            sample = samples_info[b]
            predicted_ranking = predicted_rankings[b]

            # Check if any of the predicted files belong to the same issue_id
            file_hit = False
            best_position = float('inf')

            for pos, file_idx in enumerate(predicted_ranking):
                if file_idx < len(sample['file_contents']):
                    # Check if this file belongs to the target issue_id
                    if self._file_belongs_to_issue(file_idx.item(), sample):
                        file_hit = True
                        best_position = pos
                        break

            if file_hit:
                if self.reward_type == "sparse":
                    if individual:
                        rewards[b] = 1.0  # Per-agent sparse: binary per level
                    else:
                        rewards[b] = 0.0  # Joint sparse: reward only at final step
                else:  # intermediate
                    if best_position < len(self.position_weights):
                        rewards[b] = self.position_weights[best_position]
                    else:
                        rewards[b] = self.position_weights[-1] * (0.8 ** (best_position - len(self.position_weights) + 1))

        return rewards

    def compute_per_position_rewards(self, predicted_ranking, sample, level="file",
                                      function_info=None, selected_func_idx=None):
        """Compute per-position rewards for a single sample's ranking.
        Returns a list of rewards, one per position in the ranking.
        Used by per-position REINFORCE in TWF-RL and T1B1-RL.

        Penalty logic: wrong items are only penalised when they are ranked
        *above* (before) the last correct item, i.e. they displace a correct
        item from a higher rank.  Wrong items already ranked below all correct
        items receive 0 — the ranking is already correct for those positions."""

        # Pass 1: determine correctness for each position
        correctness = []
        for pos, item_idx in enumerate(predicted_ranking):
            item = item_idx.item() if hasattr(item_idx, 'item') else item_idx

            is_correct = False
            if level == "file":
                if item < len(sample['file_contents']):
                    is_correct = self._file_belongs_to_issue(item, sample)
            elif level == "function":
                if function_info and item < len(function_info) and item != -1:
                    is_correct = self._function_contains_issue_bug(function_info[item], sample)
            elif level == "line":
                if function_info and selected_func_idx is not None:
                    sel_func = selected_func_idx.item() if hasattr(selected_func_idx, 'item') else selected_func_idx
                    if sel_func < len(function_info):
                        buggy_lines = self._get_issue_buggy_lines_in_function(
                            function_info[sel_func], sample)
                        is_correct = item in buggy_lines

            correctness.append((item, is_correct))

        # Find last correct position (highest rank index with a correct item).
        # If no correct item exists, set to len(ranking) so every wrong item
        # is considered "before" it and receives the penalty.
        last_correct_pos = len(predicted_ranking)
        for pos, (_, is_correct) in enumerate(correctness):
            if is_correct:
                last_correct_pos = pos

        # Pass 2: assign rewards
        rewards = []
        for pos, (item, is_correct) in enumerate(correctness):
            if is_correct:
                if self.reward_type == "sparse":
                    reward = 1.0
                else:
                    if pos < len(self.position_weights):
                        reward = self.position_weights[pos]
                    else:
                        reward = self.position_weights[-1] * (0.8 ** (pos - len(self.position_weights) + 1))
            else:
                # Penalise only wrong items ranked above the last correct item
                if (pos < last_correct_pos and
                        self.reward_type == "intermediate" and
                        self.wrong_item_penalty > 0):
                    reward = -self.wrong_item_penalty
                else:
                    reward = 0.0

            rewards.append(reward)
        return rewards

    def compute_per_position_rewards_flat(self, predicted_ranking, flat_correct_indices):
        """Compute per-position rewards for flat line pool (hierarchy ablation).
        flat_correct_indices: set of pool positions (integers) that are buggy lines.
        Same penalty logic as compute_per_position_rewards."""
        correctness = []
        for item_idx in predicted_ranking:
            item = item_idx.item() if hasattr(item_idx, 'item') else item_idx
            correctness.append((item, item in flat_correct_indices))

        last_correct_pos = len(predicted_ranking)
        for pos, (_, is_correct) in enumerate(correctness):
            if is_correct:
                last_correct_pos = pos

        rewards = []
        for pos, (item, is_correct) in enumerate(correctness):
            if is_correct:
                if self.reward_type == "sparse":
                    reward = 1.0
                else:
                    if pos < len(self.position_weights):
                        reward = self.position_weights[pos]
                    else:
                        reward = self.position_weights[-1] * (0.8 ** (pos - len(self.position_weights) + 1))
            else:
                if (pos < last_correct_pos and
                        self.reward_type == "intermediate" and
                        self.wrong_item_penalty > 0):
                    reward = -self.wrong_item_penalty
                else:
                    reward = 0.0
            rewards.append(reward)
        return rewards

    def compute_function_reward(self, predicted_rankings, samples_info, selected_file_indices, function_info_batch, individual=False):
        """Compute function-level rewards based on issue_id matching - any file from issue + any function from issue.
        individual: when True, sparse returns 1.0 per-level (for TWF/T1B1-RL);
                    when False, sparse returns 0.0 (reward only at final step, for VT/PT1)."""
        batch_size = predicted_rankings.size(0)
        device = predicted_rankings.device
        rewards = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            sample = samples_info[b]
            predicted_ranking = predicted_rankings[b]
            selected_file_idx = selected_file_indices[b].item()

            # Check if selected file belongs to the issue (not just the exact correct file)
            file_belongs_to_issue = self._file_belongs_to_issue(selected_file_idx, sample)

            if file_belongs_to_issue:
                function_info = function_info_batch[b]

                # Check if any predicted function belongs to the issue
                func_hit = False
                best_position = float('inf')

                for pos, func_idx in enumerate(predicted_ranking):
                    if func_idx < len(function_info) and func_idx.item() != -1:
                        # Check if this function contains any buggy line from the issue
                        func_info = function_info[func_idx.item()]
                        if self._function_contains_issue_bug(func_info, sample):
                            func_hit = True
                            best_position = pos
                            break

                if func_hit:
                    if self.reward_type == "sparse":
                        if individual:
                            rewards[b] = 1.0  # Per-agent sparse: binary per level
                        else:
                            rewards[b] = 0.0  # Joint sparse: reward only at final step
                    else:  # intermediate
                        if best_position < len(self.position_weights):
                            rewards[b] = self.position_weights[best_position]
                        else:
                            rewards[b] = self.position_weights[-1] * (0.8 ** (best_position - len(self.position_weights) + 1))

        return rewards

    def compute_line_reward(self, predicted_rankings, samples_info, selected_file_indices, selected_func_indices, function_info_batch, individual=False):
        """Compute line-level rewards based on issue_id matching - any file from issue + any function from issue + correct line.
        individual: accepted for API consistency; line-level sparse already returns 1.0 when correct."""
        batch_size = predicted_rankings.size(0)
        device = predicted_rankings.device
        rewards = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            sample = samples_info[b]
            predicted_ranking = predicted_rankings[b]
            selected_file_idx = selected_file_indices[b].item()
            
            # Check if selected file belongs to the issue (not just the exact correct file)
            file_belongs_to_issue = self._file_belongs_to_issue(selected_file_idx, sample)
            
            if not file_belongs_to_issue:
                continue
                
            # Check if selected function belongs to the issue
            function_info = function_info_batch[b]
            if (selected_func_indices[b].item() == -1 or 
                selected_func_indices[b].item() >= len(function_info)):
                continue
                
            selected_func_info = function_info[selected_func_indices[b].item()]
            
            # Check if the selected function contains issue bugs
            if not self._function_contains_issue_bug(selected_func_info, sample):
                continue
                
            # Get all buggy lines for this issue within the selected function
            buggy_lines = self._get_issue_buggy_lines_in_function(selected_func_info, sample)
            
            if buggy_lines:
                # Check if any predicted line matches any buggy line from the issue
                line_hit = False
                best_position = float('inf')
                
                for pos, line_idx in enumerate(predicted_ranking):
                    if line_idx.item() != -1 and line_idx.item() in buggy_lines:
                        line_hit = True
                        best_position = pos
                        break
                
                if line_hit:
                    if self.reward_type == "sparse":
                        rewards[b] = 1.0  # Sparse reward only given when all levels are correct
                    else:  # intermediate
                        if best_position < len(self.position_weights):
                            rewards[b] = self.position_weights[best_position]
                        else:
                            rewards[b] = self.position_weights[-1] * (0.8 ** (best_position - len(self.position_weights) + 1))
        
        return rewards
    
    def _function_contains_issue_bug(self, func_info, sample):
        """Check if function contains any buggy line from the issue"""
        # Check if the function name matches any buggy function for this issue
        if 'buggy_function_name' in sample and func_info['name'] == sample['buggy_function_name']:
            return True
            
        # Check if any buggy line number falls within this function's range
        if 'buggy_line_number' in sample:
            buggy_line = sample['buggy_line_number']
            if func_info['start_line'] <= buggy_line <= func_info['end_line']:
                return True
                
        # For issue-based approach, you might have multiple buggy lines/functions per issue
        # Check if any of them match this function
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
        
        # Single buggy line case
        if 'buggy_line_number' in sample:
            buggy_line = sample['buggy_line_number']
            if func_info['start_line'] <= buggy_line <= func_info['end_line']:
                # Convert to function-relative line index
                relative_line = buggy_line - func_info['start_line']
                if relative_line >= 0:
                    buggy_lines.append(relative_line)
        
        # Multiple buggy lines case (for issue-based approach)
        if 'buggy_lines' in sample:
            for buggy_line in sample['buggy_lines']:
                if func_info['start_line'] <= buggy_line <= func_info['end_line']:
                    relative_line = buggy_line - func_info['start_line']
                    if relative_line >= 0:
                        buggy_lines.append(relative_line)
        
        return buggy_lines
    
    def compute_mrr_multi_target(self, rankings, correct_indices_list):
        """Compute MRR for multiple correct targets per sample"""
        reciprocal_ranks = []
        
        for i in range(len(rankings)):
            correct_indices = correct_indices_list[i]
            ranking = rankings[i]
            
            best_reciprocal_rank = 0.0
            
            # Find the best (lowest) position of any correct index
            for correct_idx in correct_indices:
                try:
                    if correct_idx in ranking:
                        position = list(ranking).index(correct_idx) + 1  # 1-indexed
                        reciprocal_rank = 1.0 / position
                        best_reciprocal_rank = max(best_reciprocal_rank, reciprocal_rank)
                except Exception:
                    continue
            
            reciprocal_ranks.append(best_reciprocal_rank)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def compute_mrr(self, rankings, correct_indices):
        """Compute Mean Reciprocal Rank for single targets"""
        reciprocal_ranks = []
        
        for i in range(len(rankings)):
            correct_idx = correct_indices[i]
            ranking = rankings[i]
            
            # Find position of correct index in ranking
            try:
                if correct_idx in ranking:
                    position = list(ranking).index(correct_idx) + 1  # 1-indexed
                    reciprocal_ranks.append(1.0 / position)
                else:
                    reciprocal_ranks.append(0.0)
            except Exception:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


class RankingBasedHRLTrainer:
    """HRL Trainer with issue-based rewards and multiple training strategies"""
    
    def __init__(self, train_data, test_data, embedder, device="cuda",
                 batch_size=16, learning_rate=1e-4, entropy_coef=0.01,
                 file_top_k=5, func_top_k=15, line_top_k=15,
                 exploration_mode="epsilon_greedy", epsilon_start=0.3, epsilon_end=0.05,
                 epsilon_decay=0.995, temperature=1.0, temperature_decay=0.99,
                 reward_type="intermediate", wrong_item_penalty=0.1,
                 use_wandb=True,
                 project_name="ranking-hrl-bug-localization", seed=42,
                 training_strategy="vt",
                 checkpoint_dir="../../outputs/checkpoints/rl",
                 encoder_type="lstm"):
        
        # Fix seed for reproducibility
        fix_seed(seed)
        self.seed = seed
        self.encoder_type = encoder_type
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_oracle_func = 0.0
        self.best_oracle_func_mrr = 0.0   # MRR tie-breaker for best func checkpoint
        self.best_oracle_line = 0.0
        self.best_oracle_line_mrr = 0.0   # MRR tie-breaker for best line checkpoint
        self.best_file_top1 = 0.0  # T1B1/PT2: track best file agent for eval; T1B1 also restores before phase 2
        self.best_file_mrr = 0.0   # Tie-breaker for best file checkpoint selection

        self.device = device
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.reward_type = reward_type
        self.use_wandb = use_wandb
        self.training_strategy = training_strategy
        
        # Initialize Wandb
        if self.use_wandb:
            wandb.init(
                project=project_name,
                config={
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "entropy_coef": entropy_coef,
                    "file_top_k": file_top_k,
                    "func_top_k": func_top_k,
                    "line_top_k": line_top_k,
                    "exploration_mode": exploration_mode,
                    "epsilon_start": epsilon_start,
                    "epsilon_end": epsilon_end,
                    "epsilon_decay": epsilon_decay,
                    "temperature": temperature,
                    "temperature_decay": temperature_decay,
                    "reward_type": reward_type,
                    "train_samples": len(train_data),
                    "test_samples": len(test_data),
                    "seed": seed,
                    "training_strategy": training_strategy
                }
            )
        
        # Initialize embedder and agents (same as before)
        self.embedder = embedder
        
        # Initialize agents (encoder_type selects LSTM vs MLP for file/line agents)
        FileCls = MLPFileLevelAgent if encoder_type == "mlp" else RankingFileLevelAgent
        LineCls = MLPLineLevelAgent if encoder_type == "mlp" else RankingLineLevelAgent

        self.file_agent = FileCls(
            top_k=file_top_k, exploration_mode=exploration_mode,
            epsilon=self.epsilon, temperature=temperature
        ).to(device)

        self.func_agent = RankingFunctionLevelAgent(
            top_k=func_top_k, exploration_mode=exploration_mode,
            epsilon=self.epsilon, temperature=temperature
        ).to(device)

        self.line_agent = LineCls(
            top_k=line_top_k, exploration_mode=exploration_mode,
            epsilon=self.epsilon, temperature=temperature
        ).to(device)
        
        # Shared critic + context encoder (used by VT/PT1 joint training).
        # Joint training computes a combined reward, so one shared baseline suffices.
        self.critic = nn.Sequential(
            nn.Linear(768 + 32, 256),  # Bug embedding + context embedding
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        self.context_encoder = nn.Sequential(
            nn.Linear(3, 32),  # file_idx, func_idx, line_idx
            nn.ReLU(),
            nn.Linear(32, 32)
        ).to(device)

        # Per-level critics (used by TWF/T1B1-RL individual training).
        # Each level has a distinct reward distribution and is trained in a
        # separate phase, so a shared critic suffers catastrophic forgetting.
        # Input is ONLY the bug embedding (state); the selected action is
        # deliberately excluded so the baseline V(s) is unbiased.
        self.file_critic = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1)
        ).to(device)
        self.func_critic = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1)
        ).to(device)
        self.line_critic = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1)
        ).to(device)

        # Initialize optimizers
        self.file_optimizer = torch.optim.Adam(self.file_agent.parameters(), lr=learning_rate)
        self.func_optimizer = torch.optim.Adam(self.func_agent.parameters(), lr=learning_rate)
        self.line_optimizer = torch.optim.Adam(self.line_agent.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.context_encoder.parameters()),
            lr=learning_rate*0.1
        )
        self.file_critic_optimizer = torch.optim.Adam(self.file_critic.parameters(), lr=learning_rate*0.1)
        self.func_critic_optimizer = torch.optim.Adam(self.func_critic.parameters(), lr=learning_rate*0.1)
        self.line_critic_optimizer = torch.optim.Adam(self.line_critic.parameters(), lr=learning_rate*0.1)

        self.initial_lr = learning_rate
        self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.file_optimizer, T_max=50, eta_min=1e-6)
        self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.func_optimizer, T_max=50, eta_min=1e-6)
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=50, eta_min=1e-6)
        
        # Log model to Wandb
        if self.use_wandb:
            wandb.watch([self.file_agent, self.func_agent, self.line_agent, self.critic], log_freq=100)
        
        # Initialize issue-based reward calculator
        self.reward_calculator = IssueBasedRewardCalculator(
            reward_type=reward_type, dataset=train_data,
            wrong_item_penalty=wrong_item_penalty
        )
        
        # Initialize data
        self.train_data = train_data
        self.test_data = test_data
        
        # Reward weights
        self.file_reward_weight = 1.0
        self.func_reward_weight = 2.0
        self.line_reward_weight = 3.0
        
        print(f"Initialized RankingBasedHRLTrainer with {len(train_data)} training samples")
        print(f"Reward type: {reward_type}, Training strategy: {training_strategy}")

    def create_batches(self, data, batch_size):
        """Group samples by file-count then split into fixed-size batches."""
        file_count_groups = defaultdict(list)
        for sample in data:
            file_count_groups[len(sample['file_contents'])].append(sample)

        batches = []
        for num_files, samples in file_count_groups.items():
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                if len(batch) > 0:
                    batches.append(batch)
        
        return batches

    def update_exploration(self):
        """Update exploration parameters"""
        if self.exploration_mode == "epsilon_greedy":
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        elif self.exploration_mode == "boltzmann":
            self.temperature = max(0.1, self.temperature * self.temperature_decay)
        
        self.file_agent.epsilon = self.epsilon
        self.func_agent.epsilon = self.epsilon
        self.line_agent.epsilon = self.epsilon
        
        self.file_agent.temperature = self.temperature
        self.func_agent.temperature = self.temperature
        self.line_agent.temperature = self.temperature

    def _reset_exploration(self):
        """Reset epsilon/temperature to starting values for a new phase.
        Called between sequential phases (TWF/T1B1-RL) so each new agent
        starts with full exploration that decays over its own training."""
        if self.exploration_mode == "epsilon_greedy":
            self.epsilon = 0.3  # matches epsilon_start default
        elif self.exploration_mode == "boltzmann":
            self.temperature = 1.0  # matches temperature default
        self.file_agent.epsilon = self.epsilon
        self.func_agent.epsilon = self.epsilon
        self.line_agent.epsilon = self.epsilon
        self.file_agent.temperature = self.temperature
        self.func_agent.temperature = self.temperature
        self.line_agent.temperature = self.temperature
        print(f"  Exploration reset: epsilon={self.epsilon:.3f}, temperature={self.temperature:.3f}")

    def pad_batch_data(self, batch):
        """Pad file lists to max_files across the batch."""
        max_files = max(len(sample['file_contents']) for sample in batch)
        padded_file_contents = []
        padded_file_paths = []
        correct_file_indices = []
        bug_embeddings = []
        samples_info = []

        for sample in batch:
            files = list(sample['file_contents'])  # copy to avoid mutating original
            paths = list(sample.get('file_paths', []))
            while len(files) < max_files:
                files.append("")  # Empty string for padding
            while len(paths) < max_files:
                paths.append("")

            padded_file_contents.append(files)
            padded_file_paths.append(paths)
            correct_file_indices.append(sample['correct_file_idx'])
            bug_embeddings.append(torch.tensor(sample['stack_trace_embedding'], dtype=torch.float))
            samples_info.append(sample)

        return {
            'file_contents': padded_file_contents,
            'file_paths': padded_file_paths,
            'correct_file_indices': torch.tensor(correct_file_indices, device=self.device),
            'bug_embeddings': torch.stack(bug_embeddings).to(self.device),
            'samples_info': samples_info,
            'batch_size': len(batch)
        }

    def pretrain_file_agent(self, epochs=10):
        """PT1: Pretrain file agent independently"""
        print("Pretraining File Agent...")
        self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.file_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            batches = self.create_batches(self.train_data, self.batch_size)
            random.shuffle(batches)
            
            for batch in batches:
                batch_data = self.pad_batch_data(batch)
                loss = self._pretrain_file_batch(batch_data)
                
                if loss.requires_grad:
                    self.file_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.file_agent.parameters(), max_norm=1.0)
                    self.file_optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.file_scheduler.step()
                print(f"File Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")

                if self.use_wandb:
                    wandb.log({
                        "pretrain/file_loss": avg_loss,
                        "pretrain/file_epoch": epoch + 1
                    })

                self.save_agent()

    def train_batch(self, batch_data):
        """Train on a single batch and return loss and reward"""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        file_contents_batch = batch_data['file_contents']
        file_paths_batch = batch_data.get('file_paths', [None] * batch_size)
        samples_info = batch_data['samples_info']

        # Process file embeddings
        all_file_embeddings = []
        for file_contents, file_paths in zip(file_contents_batch, file_paths_batch):
            file_emb = self.embedder.get_file_embeddings(file_contents, file_paths=file_paths).to(self.device)
            all_file_embeddings.append(file_emb)
        
        # Stack and pad file embeddings
        max_files = max(emb.size(0) for emb in all_file_embeddings)
        padded_file_embeddings = []
        
        for file_emb in all_file_embeddings:
            if file_emb.size(0) < max_files:
                pad_size = max_files - file_emb.size(0)
                padding = torch.zeros(pad_size, file_emb.size(1), device=self.device)
                file_emb = torch.cat([file_emb, padding], dim=0)
            padded_file_embeddings.append(file_emb)
        
        file_embeddings = torch.stack(padded_file_embeddings)
        
        # File-level predictions
        file_rankings, file_scores, file_confidences = self.file_agent(
            bug_embeddings, file_embeddings, training=True
        )
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_reward = 0.0
        valid_samples = 0
        
        # Process each sample in the batch
        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            file_ranking = file_rankings[i]
            
            # Select top file
            selected_file_idx = file_ranking[0].item()
            
            if selected_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][selected_file_idx]
                
                # Extract functions
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    # Function-level predictions
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                    func_rankings, func_scores, func_confidences = self.func_agent(
                        bug_emb, func_emb, training=True
                    )
                    
                    # Select top function
                    selected_func_idx = func_rankings[0, 0].item()
                    
                    if selected_func_idx < len(candidate_functions):
                        selected_function_content = candidate_functions[selected_func_idx]
                        candidate_lines = selected_function_content.split('\n')
                        
                        if candidate_lines:
                            # Line-level predictions
                            line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                            line_rankings, line_scores, line_confidences = self.line_agent(
                                bug_emb, line_emb, training=True
                            )
                            
                            # Compute rewards
                            file_reward = self.reward_calculator.compute_file_reward(
                                file_rankings[i:i+1], [sample]
                            )
                            
                            func_reward = self.reward_calculator.compute_function_reward(
                                func_rankings, [sample], 
                                torch.tensor([selected_file_idx], device=self.device),
                                [function_info]
                            )
                            
                            line_reward = self.reward_calculator.compute_line_reward(
                                line_rankings, [sample],
                                torch.tensor([selected_file_idx], device=self.device),
                                torch.tensor([selected_func_idx], device=self.device),
                                [function_info]
                            )
                            
                            # Compute total reward
                            sample_reward = (
                                self.file_reward_weight * file_reward.item() +
                                self.func_reward_weight * func_reward.item() +
                                self.line_reward_weight * line_reward.item()
                            )
                            
                            # Compute critic values
                            context = torch.tensor([selected_file_idx, selected_func_idx, 0], 
                                                device=self.device, dtype=torch.float).unsqueeze(0)
                            context_emb = self.context_encoder(context)
                            critic_input = torch.cat([bug_emb, context_emb], dim=-1)
                            value = self.critic(critic_input)
                            
                            # Compute advantage
                            advantage = sample_reward - value.item()
                            
                            # Policy losses (simplified)
                            file_log_prob = F.log_softmax(file_scores[i:i+1], dim=-1)[0, file_ranking[0]]
                            func_log_prob = F.log_softmax(func_scores[0:1], dim=-1)[0, func_rankings[0, 0]]
                            line_log_prob = F.log_softmax(line_scores[0:1], dim=-1)[0, line_rankings[0, 0]]
                            
                            policy_loss = -(file_log_prob + func_log_prob + line_log_prob) * advantage
                            
                            # Value loss
                            value_loss = F.mse_loss(value.squeeze(), torch.tensor(sample_reward, device=self.device))
                            
                            # Entropy losses
                            file_entropy = -torch.sum(F.softmax(file_scores[i:i+1], dim=-1) * F.log_softmax(file_scores[i:i+1], dim=-1))
                            func_entropy = -torch.sum(F.softmax(func_scores[0:1], dim=-1) * F.log_softmax(func_scores[0:1], dim=-1))
                            line_entropy = -torch.sum(F.softmax(line_scores[0:1], dim=-1) * F.log_softmax(line_scores[0:1], dim=-1))
                            
                            entropy_bonus = self.entropy_coef * (file_entropy + func_entropy + line_entropy)
                            
                            # Total loss
                            sample_loss = policy_loss + value_loss - entropy_bonus
                            total_loss = total_loss + sample_loss
                            
                            total_reward += sample_reward
                            valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples, total_reward / valid_samples
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

    def evaluate(self, quick_eval=False):
        """Issue-based evaluation with hierarchical dependencies"""
        self.file_agent.eval()
        self.func_agent.eval()
        self.line_agent.eval()
        
        total_samples = 0
        
        # File level metrics
        file_top1_correct = 0
        file_top5_correct = 0
        file_rankings_list = []
        file_correct_indices = []
        
        # Function level metrics (issue-based, depends on correct file)
        func_top1_correct = 0
        func_top5_correct = 0
        func_top10_correct = 0
        func_top15_correct = 0
        func_rankings_list = []
        func_correct_indices_list = []
        
        # Line level metrics (issue-based, depends on correct file AND function)
        line_top1_correct = 0
        line_top5_correct = 0
        line_top10_correct = 0
        line_top15_correct = 0
        line_rankings_list = []
        line_correct_indices_list = []
        
        # Hierarchical metrics
        hierarchical_top1_correct = 0
        hierarchical_top5_correct = 0
        
        # Issue-based metrics
        issue_file_coverage = 0
        issue_func_coverage = 0
        issue_line_coverage = 0
        
        # Use subset for quick evaluation
        eval_data = self.test_data[:50] if quick_eval else self.test_data
        
        with torch.no_grad():
            for sample in eval_data:
                try:
                    bug_emb = torch.tensor(sample['stack_trace_embedding'], 
                                        dtype=torch.float, device=self.device).unsqueeze(0)
                    candidate_files = sample['file_contents']
                    candidate_file_paths = sample.get('file_paths')

                    # File level evaluation
                    file_emb = self.embedder.get_file_embeddings(candidate_files, file_paths=candidate_file_paths).to(self.device).unsqueeze(0)
                    file_rankings, file_scores, file_confidences = self.file_agent(bug_emb, file_emb, training=False)
                    
                    correct_file_idx = sample['correct_file_idx']
                    
                    # Store for MRR calculation
                    file_rankings_list.append(file_rankings[0].cpu().numpy())
                    file_correct_indices.append(correct_file_idx)
                    
                    # Check file metrics
                    file_top1_hit = len(file_rankings[0]) > 0 and file_rankings[0, 0].item() == correct_file_idx
                    file_top5_hit = correct_file_idx in file_rankings[0][:5]
                    
                    if file_top1_hit:
                        file_top1_correct += 1
                    if file_top5_hit:
                        file_top5_correct += 1
                        issue_file_coverage += 1
                    
                    # Function evaluation - if ANY file from issue was selected in top 5
                    func_top1_hit = False
                    func_top5_hit = False
                    func_top10_hit = False
                    func_top15_hit = False

                    if file_top5_hit and len(file_rankings[0]) > 0:  # Any file from issue in top 5
                        # Try the top file selection (not necessarily the exact correct file)
                        selected_file_idx = file_rankings[0, 0].item()
                        
                        # Check if this file belongs to the issue
                        file_belongs_to_issue = self.reward_calculator._file_belongs_to_issue(selected_file_idx, sample)
                        
                        if file_belongs_to_issue and selected_file_idx < len(candidate_files):
                            selected_file_content = candidate_files[selected_file_idx]
                            candidate_functions, function_info = get_all_function_candidates(selected_file_content)
                            
                            if candidate_functions:
                                func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                                func_rankings, func_scores, func_confidences = self.func_agent(bug_emb, func_emb, training=False)
                                
                                # Get all correct function indices for this issue
                                correct_func_indices = self._get_all_correct_function_indices_issue_based(function_info, sample)
                                
                                if correct_func_indices:
                                    # Store for MRR calculation
                                    func_rankings_list.append(func_rankings[0].cpu().numpy())
                                    func_correct_indices_list.append(correct_func_indices)
                                    
                                    # Check function metrics (issue-based)
                                    func_top1_hit = len(func_rankings[0]) > 0 and func_rankings[0, 0].item() in correct_func_indices
                                    func_top5_hit = any(idx in correct_func_indices for idx in func_rankings[0][:5])
                                    func_top10_hit = any(idx in correct_func_indices for idx in func_rankings[0][:10])
                                    func_top15_hit = any(idx in correct_func_indices for idx in func_rankings[0][:15])
                                    
                                    if func_top1_hit:
                                        func_top1_correct += 1
                                    if func_top5_hit:
                                        func_top5_correct += 1
                                        issue_func_coverage += 1
                                    if func_top10_hit:
                                        func_top10_correct += 1
                                    if func_top15_hit:
                                        func_top15_correct += 1
                                    
                                    # Line evaluation - if ANY function from issue was selected in top 1
                                    line_top1_hit = False
                                    line_top5_hit = False
                                    line_top10_hit = False
                                    line_top15_hit = False
                                    
                                    if func_top1_hit and len(func_rankings[0]) > 0:  # Any function from issue in top 1
                                        selected_func_idx = func_rankings[0, 0].item()
                                        
                                        if (selected_func_idx in correct_func_indices and 
                                            selected_func_idx < len(candidate_functions)):
                                            
                                            selected_function_content = candidate_functions[selected_func_idx]
                                            candidate_lines = selected_function_content.split('\n')
                                            
                                            if candidate_lines:
                                                line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                                                line_rankings, line_scores, line_confidences = self.line_agent(bug_emb, line_emb, training=False)
                                                
                                                # Get all correct line indices for this issue within the function
                                                correct_line_indices = self._calculate_correct_line_indices_issue_based(sample, function_info[selected_func_idx])
                                                
                                                if correct_line_indices:
                                                    # Store for MRR calculation
                                                    line_rankings_list.append(line_rankings[0].cpu().numpy())
                                                    line_correct_indices_list.append(correct_line_indices)
                                                    
                                                    # Check line metrics (issue-based)
                                                    line_top1_hit = len(line_rankings[0]) > 0 and line_rankings[0, 0].item() in correct_line_indices
                                                    line_top5_hit = any(idx in correct_line_indices for idx in line_rankings[0][:5])
                                                    line_top10_hit = any(idx in correct_line_indices for idx in line_rankings[0][:10])
                                                    line_top15_hit = any(idx in correct_line_indices for idx in line_rankings[0][:15])
                                                    
                                                    if line_top1_hit:
                                                        line_top1_correct += 1
                                                    if line_top5_hit:
                                                        line_top5_correct += 1
                                                        issue_line_coverage += 1
                                                    if line_top10_hit:
                                                        line_top10_correct += 1
                                                    if line_top15_hit:
                                                        line_top15_correct += 1
                    
                    # Hierarchical accuracy (all levels must be correct)
                    if file_top5_hit and func_top1_hit and line_top1_hit:
                        hierarchical_top1_correct += 1
                    
                    if file_top5_hit and func_top5_hit and line_top5_hit:
                        hierarchical_top5_correct += 1
                    
                    total_samples += 1
                except Exception as e:
                    print(f"Error in evaluation sample: {e}")
                    continue
        
        self.file_agent.train()
        self.func_agent.train()
        self.line_agent.train()
        
        # Compute MRR metrics (issue-based)
        file_mrr = self.reward_calculator.compute_mrr(file_rankings_list, file_correct_indices)
        func_mrr = self.reward_calculator.compute_mrr_multi_target(func_rankings_list, func_correct_indices_list)
        line_mrr = self.reward_calculator.compute_mrr_multi_target(line_rankings_list, line_correct_indices_list)
        
        if total_samples > 0:
            metrics = {
                # File level metrics
                'file_accuracy_top_1': file_top1_correct / total_samples,
                'file_accuracy_top_5': file_top5_correct / total_samples,
                'file_mrr': file_mrr,
                
                # Function level metrics (hierarchical - depends on correct file)
                'func_accuracy_top_1': func_top1_correct / total_samples,
                'func_accuracy_top_5': func_top5_correct / total_samples,
                'func_accuracy_top_10': func_top10_correct / total_samples,
                'func_accuracy_top_15': func_top15_correct / total_samples,
                'func_mrr': func_mrr,
                
                # Line level metrics (hierarchical - depends on correct file AND function)
                'line_accuracy_top_1': line_top1_correct / total_samples,
                'line_accuracy_top_5': line_top5_correct / total_samples,
                'line_accuracy_top_10': line_top10_correct / total_samples,
                'line_accuracy_top_15': line_top15_correct / total_samples,
                'line_mrr': line_mrr,
                
                # Hierarchical metrics (all levels correct)
                'hierarchical_accuracy_top_1': hierarchical_top1_correct / total_samples,
                'hierarchical_accuracy_top_5': hierarchical_top5_correct / total_samples,
                
                # Issue coverage metrics
                'issue_file_coverage': issue_file_coverage / total_samples,
                'issue_func_coverage': issue_func_coverage / total_samples,
                'issue_line_coverage': issue_line_coverage / total_samples,
                
                # Sample count
                'total_samples': total_samples,
            }
            
            oracle = self.evaluate_oracle()
            metrics.update(oracle)
            # Best func checkpoint: top1 → MRR → latest epoch
            func_top1 = oracle.get('oracle_func_top1', 0)
            func_mrr = oracle.get('oracle_func_mrr', 0)
            eps = 1e-9
            func_improved = (
                func_top1 > self.best_oracle_func + eps
                or (abs(func_top1 - self.best_oracle_func) <= eps
                    and func_mrr > self.best_oracle_func_mrr + eps)
                or (abs(func_top1 - self.best_oracle_func) <= eps
                    and abs(func_mrr - self.best_oracle_func_mrr) <= eps)
            )
            if func_improved:
                self.best_oracle_func = func_top1
                self.best_oracle_func_mrr = func_mrr
                if self.training_strategy in ('pt2', 't1b1', 'twf', 't1b1rl'):
                    self.save_agent(tag='best_oracle_func', agents=['func'])
                else:
                    self.save_agent(tag='best_oracle_func')
            # Best line checkpoint: top1 → MRR → latest epoch
            line_top1 = oracle.get('oracle_line_top1', 0)
            line_mrr = oracle.get('oracle_line_mrr', 0)
            line_improved = (
                line_top1 > self.best_oracle_line + eps
                or (abs(line_top1 - self.best_oracle_line) <= eps
                    and line_mrr > self.best_oracle_line_mrr + eps)
                or (abs(line_top1 - self.best_oracle_line) <= eps
                    and abs(line_mrr - self.best_oracle_line_mrr) <= eps)
            )
            if line_improved:
                self.best_oracle_line = line_top1
                self.best_oracle_line_mrr = line_mrr
                if self.training_strategy in ('pt2', 't1b1', 'twf', 't1b1rl'):
                    self.save_agent(tag='best_oracle_line', agents=['line'])
                else:
                    self.save_agent(tag='best_oracle_line')
            # T1B1/PT2/TWF: track best file agent for evaluation (T1B1 also uses it in phase 2 training)
            if self.training_strategy in ('t1b1', 'pt2', 'twf', 't1b1rl'):
                file_top1 = metrics.get('file_accuracy_top_1', 0)
                file_mrr = metrics.get('file_mrr', 0)
                eps = 1e-9
                improved_top1 = file_top1 > self.best_file_top1 + eps
                tie_top1_improved_mrr = (
                    abs(file_top1 - self.best_file_top1) <= eps and
                    file_mrr > self.best_file_mrr + eps
                )
                # If both top-1 and MRR tie, keep the latest checkpoint.
                tie_top1_tie_mrr_use_latest = (
                    abs(file_top1 - self.best_file_top1) <= eps and
                    abs(file_mrr - self.best_file_mrr) <= eps
                )
                if improved_top1 or tie_top1_improved_mrr or tie_top1_tie_mrr_use_latest:
                    self.best_file_top1 = file_top1
                    self.best_file_mrr = file_mrr
                    self.save_agent(tag='best_file', agents=['file'])
            return metrics
        else:
            return {'error': 'No valid samples for evaluation'}

    def evaluate_oracle(self):
        """Evaluate func/line agents using GT file and GT function (independent of cascade).
        Reveals whether agents are learning even when E2E metrics are near zero."""
        self.file_agent.eval(); self.func_agent.eval(); self.line_agent.eval()
        oracle_func_top1 = oracle_func_top5 = oracle_line_top1 = oracle_line_top5 = 0
        oracle_func_rr_sum = oracle_line_rr_sum = 0.0  # reciprocal ranks for MRR
        func_total = line_total = 0
        ng_func_top1 = ng_func_top5 = ng_func_total = 0

        with torch.no_grad():
            for sample in self.test_data:
                try:
                    correct_file_idx = sample['correct_file_idx']
                    if correct_file_idx < 0 or correct_file_idx >= len(sample['file_contents']):
                        continue
                    candidate_functions, function_info = get_all_function_candidates(
                        sample['file_contents'][correct_file_idx])
                    if not candidate_functions:
                        continue

                    bug_emb = torch.tensor(sample['stack_trace_embedding'],
                                           dtype=torch.float, device=self.device).unsqueeze(0)
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                    func_rankings, _, _ = self.func_agent(bug_emb, func_emb, training=False)
                    correct_func_indices = self._get_all_correct_function_indices_issue_based(
                        function_info, sample)
                    if not correct_func_indices:
                        continue

                    func_total += 1
                    if func_rankings[0, 0].item() in correct_func_indices:
                        oracle_func_top1 += 1
                    if any(i in correct_func_indices for i in func_rankings[0][:5].tolist()):
                        oracle_func_top5 += 1
                    # Reciprocal rank: 1/(position of first correct)
                    for pos, idx in enumerate(func_rankings[0].tolist()):
                        if idx in correct_func_indices:
                            oracle_func_rr_sum += 1.0 / (pos + 1)
                            break

                    # Non-global breakdown (exclude samples where buggy function is 'global')
                    is_global_sample = sample.get('buggy_function_name') == 'global'
                    if not is_global_sample:
                        ng_func_total += 1
                        if func_rankings[0, 0].item() in correct_func_indices:
                            ng_func_top1 += 1
                        if any(i in correct_func_indices for i in func_rankings[0][:5].tolist()):
                            ng_func_top5 += 1

                    # Oracle line: use GT function directly, independent of func agent prediction
                    gt_func_idx = next(
                        (i for i, fi in enumerate(function_info)
                         if self.reward_calculator._function_contains_issue_bug(fi, sample)), None)
                    if gt_func_idx is not None and gt_func_idx < len(candidate_functions):
                        correct_line_indices = self._calculate_correct_line_indices_issue_based(
                            sample, function_info[gt_func_idx])
                        if correct_line_indices:
                            candidate_lines = candidate_functions[gt_func_idx].split('\n')
                            line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                            line_rankings, _, _ = self.line_agent(bug_emb, line_emb, training=False)
                            line_total += 1
                            if line_rankings[0, 0].item() in correct_line_indices:
                                oracle_line_top1 += 1
                            if any(i in correct_line_indices for i in line_rankings[0][:5].tolist()):
                                oracle_line_top5 += 1
                            for pos, idx in enumerate(line_rankings[0].tolist()):
                                if idx in correct_line_indices:
                                    oracle_line_rr_sum += 1.0 / (pos + 1)
                                    break
                except Exception:
                    continue

        self.file_agent.train(); self.func_agent.train(); self.line_agent.train()
        return {
            'oracle_func_top1': oracle_func_top1 / func_total if func_total else 0.0,
            'oracle_func_top5': oracle_func_top5 / func_total if func_total else 0.0,
            'oracle_func_mrr': oracle_func_rr_sum / func_total if func_total else 0.0,
            'oracle_line_top1': oracle_line_top1 / line_total if line_total else 0.0,
            'oracle_line_top5': oracle_line_top5 / line_total if line_total else 0.0,
            'oracle_line_mrr': oracle_line_rr_sum / line_total if line_total else 0.0,
            'oracle_func_n': func_total,
            'oracle_line_n': line_total,
            'oracle_func_top1_nonglobal': ng_func_top1 / ng_func_total if ng_func_total else 0.0,
            'oracle_func_top5_nonglobal': ng_func_top5 / ng_func_total if ng_func_total else 0.0,
            'oracle_func_n_nonglobal': ng_func_total,
        }

    def _get_all_correct_function_indices_issue_based(self, function_info, sample):
        """Get all correct function indices based on issue"""
        correct_indices = []
        
        for idx, func_info in enumerate(function_info):
            if self.reward_calculator._function_contains_issue_bug(func_info, sample):
                correct_indices.append(idx)
        
        return correct_indices

    def _find_correct_function_idx_issue_based(self, function_info, sample):
        """Find the correct function index based on issue"""
        for idx, func_info in enumerate(function_info):
            if self.reward_calculator._function_contains_issue_bug(func_info, sample):
                return idx
        return 0  # Default to first function

    def _calculate_correct_line_indices_issue_based(self, sample, function_info):
        """Calculate correct line indices within a function based on issue"""
        if function_info.get('name') == 'global':
            return []  # Skip line eval for global code (line numbering unreliable after extraction)
        return self.reward_calculator._get_issue_buggy_lines_in_function(function_info, sample)
    def pretrain_function_agent(self, epochs=10):
        """PT1: Pretrain function agent independently"""
        print("Pretraining Function Agent...")
        self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.func_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            batches = self.create_batches(self.train_data, self.batch_size)
            random.shuffle(batches)
            
            for batch in batches:
                batch_data = self.pad_batch_data(batch)
                loss = self._pretrain_function_batch(batch_data)
                
                if loss.requires_grad:
                    self.func_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.func_agent.parameters(), max_norm=1.0)
                    self.func_optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.func_scheduler.step()
                print(f"Function Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")
                self.save_agent()
                if self.use_wandb:
                    wandb.log({
                        "pretrain/func_loss": avg_loss,
                        "pretrain/func_epoch": epoch + 1
                    })

    def pretrain_line_agent(self, epochs=10):
        """PT1: Pretrain line agent independently"""
        print("Pretraining Line Agent...")
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            batches = self.create_batches(self.train_data, self.batch_size)
            random.shuffle(batches)
            
            for batch in batches:
                batch_data = self.pad_batch_data(batch)
                loss = self._pretrain_line_batch(batch_data)
                
                if loss.requires_grad:
                    self.line_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.line_agent.parameters(), max_norm=1.0)
                    self.line_optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.line_scheduler.step()
                print(f"Line Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")
                self.save_agent()
                if self.use_wandb:
                    wandb.log({
                        "pretrain/line_loss": avg_loss,
                        "pretrain/line_epoch": epoch + 1
                    })

    def _pretrain_file_batch(self, batch_data):
        """Pretrain file agent using ground truth"""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        file_contents_batch = batch_data['file_contents']
        file_paths_batch = batch_data.get('file_paths', [None] * batch_size)
        samples_info = batch_data['samples_info']

        # Process file embeddings
        all_file_embeddings = []
        for file_contents, file_paths in zip(file_contents_batch, file_paths_batch):
            file_emb = self.embedder.get_file_embeddings(file_contents, file_paths=file_paths).to(self.device)
            all_file_embeddings.append(file_emb)
        
        # Stack and pad
        max_files = max(emb.size(0) for emb in all_file_embeddings)
        padded_file_embeddings = []
        
        for file_emb in all_file_embeddings:
            if file_emb.size(0) < max_files:
                pad_size = max_files - file_emb.size(0)
                padding = torch.zeros(pad_size, file_emb.size(1), device=self.device)
                file_emb = torch.cat([file_emb, padding], dim=0)
            padded_file_embeddings.append(file_emb)
        
        file_embeddings = torch.stack(padded_file_embeddings)
        
        # Get predictions
        file_rankings, file_scores, file_confidences = self.file_agent(bug_embeddings, file_embeddings, training=True)
        
        # Compute supervised loss using ground truth
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i in range(batch_size):
            correct_file_idx = samples_info[i]['correct_file_idx']
            if correct_file_idx < file_scores.size(1):
                # Cross-entropy loss with ground truth
                target = torch.tensor([correct_file_idx], device=self.device)
                loss = F.cross_entropy(file_scores[i:i+1], target)
                total_loss = total_loss + loss
        
        return total_loss / batch_size

    def _pretrain_function_batch(self, batch_data):
        """Pretrain function agent using ground truth"""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        samples_info = batch_data['samples_info']
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0
        
        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            
            # Use ground truth file
            correct_file_idx = sample['correct_file_idx']
            if correct_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][correct_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                    func_rankings, func_scores, func_confidences = self.func_agent(bug_emb, func_emb, training=True)

                    # Get ground truth function indices
                    correct_func_indices = self._get_all_correct_function_indices_issue_based(function_info, sample)

                    if correct_func_indices and len(correct_func_indices) > 0:
                        # Use first correct function as target
                        target = torch.tensor([correct_func_indices[0]], device=self.device)
                        if correct_func_indices[0] < func_scores.size(1):
                            loss = F.cross_entropy(func_scores, target)
                            total_loss = total_loss + loss
                            valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _pretrain_line_batch(self, batch_data):
        """Pretrain line agent using ground truth"""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        samples_info = batch_data['samples_info']
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0
        
        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            
            # Use ground truth file and function
            correct_file_idx = sample['correct_file_idx']
            if correct_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][correct_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    # Get ground truth function
                    correct_func_idx = self._find_correct_function_idx_issue_based(function_info, sample)
                    
                    if correct_func_idx >= 0 and correct_func_idx < len(candidate_functions):
                        selected_function_content = candidate_functions[correct_func_idx]
                        candidate_lines = selected_function_content.split('\n')
                        
                        if candidate_lines:
                            line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                            line_rankings, line_scores, line_confidences = self.line_agent(bug_emb, line_emb, training=True)
                            
                            # Get ground truth line indices
                            correct_line_indices = self._calculate_correct_line_indices_issue_based(sample, function_info[correct_func_idx])
                            
                            if correct_line_indices and len(correct_line_indices) > 0:
                                # Use first correct line as target
                                target = torch.tensor([correct_line_indices[0]], device=self.device)
                                if correct_line_indices[0] < line_scores.size(1):
                                    loss = F.cross_entropy(line_scores, target)
                                    total_loss = total_loss + loss
                                    valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_sequential_ground_truth(self, epochs=20, eval_interval=5):
        """PT2: Train each agent sequentially using ground truth from upper levels"""
        print("Training with Sequential Ground Truth Strategy (PT2)...")
        
        # Train file agent first
        print("Phase 1: Training File Agent")
        self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.file_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._train_file_agent_sequential(epoch)
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 1 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                wandb.log({
                            **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                            "train/step": epoch
                        })
                print(f"File Agent Epoch {epoch+1} Metrics: {metrics}")
                self.save_agent()

        # Train function agent with ground truth files
        print("Phase 2: Training Function Agent with Ground Truth Files")
        self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.func_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._train_function_agent_sequential(epoch)
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 2 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                wandb.log({
                            **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                            "train/step": epoch
                        })
                print(f"Function Agent Epoch {epoch+1} Metrics: {metrics}")
                self.save_agent()

        # Train line agent with ground truth files and functions
        print("Phase 3: Training Line Agent with Ground Truth Files and Functions")
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._train_line_agent_sequential(epoch)
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 3 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                wandb.log({
                            **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                            "train/step": epoch
                        })
                print(f"Line Agent Epoch {epoch+1} Metrics: {metrics}")
            self.save_agent()

    def train_top_down_with_freezing(self, epochs=20, eval_interval=5):
        """T1B1: Train agents one by one, freezing upper agents after training"""
        print("Training with Top-Down Freezing Strategy (T1B1)...")
        
        # Phase 1: Train file agent only
        print("Phase 1: Training File Agent Only")
        self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.file_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._train_file_agent_sequential(epoch)
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 1 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                print(f"File Only Epoch {epoch+1} Metrics: {metrics}")
                wandb.log({
                            **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                            "train/step": epoch
                        })

        # Restore best file agent before freezing (final epoch may have drifted from peak)
        strategy_suffix = f"_{self.training_strategy}" if self.training_strategy else ""
        reward_suffix = f"_{self.reward_calculator.reward_type}" if hasattr(self, 'reward_calculator') else ""
        seed_suffix = f"_seed{self.seed}" if self.seed is not None else ""
        best_file_path = os.path.join(self.checkpoint_dir, f"file_agent{strategy_suffix}{reward_suffix}{seed_suffix}_best_file.pt")
        if os.path.exists(best_file_path):
            self.file_agent.load_state_dict(torch.load(best_file_path, map_location=self.device, weights_only=True))
            print(f"Restored best file agent from {best_file_path} (file_top1={self.best_file_top1:.4f})")

        # Freeze file agent
        for param in self.file_agent.parameters():
            param.requires_grad = False
        print("File agent frozen")

        # Phase 2: Train function agent with frozen file agent
        print("Phase 2: Training Function Agent with Frozen File Agent")
        self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.func_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._train_function_agent_with_frozen_file(epoch)
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 2 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                print(f"Function with Frozen File Epoch {epoch+1} Metrics: {metrics}")
                wandb.log({
                            **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                            "train/step": epoch
                        })

        # Restore best func agent before freezing (saved during Phase 2 as best_oracle_func)
        best_func_path = os.path.join(self.checkpoint_dir, f"function_agent{strategy_suffix}{reward_suffix}{seed_suffix}_best_oracle_func.pt")
        if os.path.exists(best_func_path):
            self.func_agent.load_state_dict(torch.load(best_func_path, map_location=self.device, weights_only=True))
            print(f"Restored best func agent from {best_func_path} (oracle_func={self.best_oracle_func:.4f})")

        # Freeze function agent
        for param in self.func_agent.parameters():
            param.requires_grad = False
        print("Function agent frozen")

        # Phase 3: Train line agent with frozen file and function agents
        print("Phase 3: Training Line Agent with Frozen File and Function Agents")
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._train_line_agent_with_frozen_upper(epoch)
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 3 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                print(f"Line with Frozen Upper Epoch {epoch+1} Metrics: {metrics}")
                wandb.log({
                            **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                            "train/step": epoch
                        })
        self.save_agent()

    def _train_agent_sequential(self, epoch, agent, optimizer, scheduler, batch_method, label):
        """Generic sequential training epoch for any agent (CE loss)."""
        epoch_loss = 0
        num_batches = 0

        batches = self.create_batches(self.train_data, self.batch_size)
        random.shuffle(batches)

        for batch in batches:
            batch_data = self.pad_batch_data(batch)
            loss = batch_method(batch_data)

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        self.update_exploration()
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            scheduler.step()
            print(f"{label} Sequential Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def _train_file_agent_sequential(self, epoch):
        self._train_agent_sequential(epoch, self.file_agent, self.file_optimizer,
                                     self.file_scheduler, self._pretrain_file_batch, "File")

    def _train_function_agent_sequential(self, epoch):
        self._train_agent_sequential(epoch, self.func_agent, self.func_optimizer,
                                     self.func_scheduler, self._pretrain_function_batch, "Function")

    def _train_line_agent_sequential(self, epoch):
        self._train_agent_sequential(epoch, self.line_agent, self.line_optimizer,
                                     self.line_scheduler, self._pretrain_line_batch, "Line")

    def _train_function_agent_with_frozen_file(self, epoch):
        """Train function agent with frozen file agent in T1B1 strategy"""
        epoch_loss = 0
        num_batches = 0
        
        batches = self.create_batches(self.train_data, self.batch_size)
        random.shuffle(batches)
        
        for batch in batches:
            batch_data = self.pad_batch_data(batch)
            loss = self._train_function_with_file_predictions(batch_data)
            
            if loss.requires_grad:
                self.func_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.func_agent.parameters(), max_norm=1.0)
                self.func_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        self.update_exploration()
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            self.func_scheduler.step()
            print(f"Function with Frozen File Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def _train_line_agent_with_frozen_upper(self, epoch):
        """Train line agent with frozen file and function agents in T1B1 strategy"""
        epoch_loss = 0
        num_batches = 0
        
        batches = self.create_batches(self.train_data, self.batch_size)
        random.shuffle(batches)
        
        for batch in batches:
            batch_data = self.pad_batch_data(batch)
            loss = self._train_line_with_upper_predictions(batch_data)
            
            if loss.requires_grad:
                self.line_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.line_agent.parameters(), max_norm=1.0)
                self.line_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        self.update_exploration()
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            self.line_scheduler.step()
            print(f"Line with Frozen Upper Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def _train_function_with_file_predictions(self, batch_data):
        """Train function agent using file agent predictions"""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        file_contents_batch = batch_data['file_contents']
        file_paths_batch = batch_data.get('file_paths', [None] * batch_size)
        samples_info = batch_data['samples_info']

        # Get file predictions (frozen)
        all_file_embeddings = []
        for file_contents, file_paths in zip(file_contents_batch, file_paths_batch):
            file_emb = self.embedder.get_file_embeddings(file_contents, file_paths=file_paths).to(self.device)
            all_file_embeddings.append(file_emb)
        
        max_files = max(emb.size(0) for emb in all_file_embeddings)
        padded_file_embeddings = []
        
        for file_emb in all_file_embeddings:
            if file_emb.size(0) < max_files:
                pad_size = max_files - file_emb.size(0)
                padding = torch.zeros(pad_size, file_emb.size(1), device=self.device)
                file_emb = torch.cat([file_emb, padding], dim=0)
            padded_file_embeddings.append(file_emb)
        
        file_embeddings = torch.stack(padded_file_embeddings)
        
        with torch.no_grad():
            file_rankings, _, _ = self.file_agent(bug_embeddings, file_embeddings, training=False)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0
        
        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            
            # Use predicted file (top prediction)
            selected_file_idx = file_rankings[i, 0].item()

            if selected_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][selected_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                    func_rankings, func_scores, func_confidences = self.func_agent(bug_emb, func_emb, training=True)

                    # Get ground truth function indices
                    correct_func_indices = self._get_all_correct_function_indices_issue_based(function_info, sample)

                    if correct_func_indices and len(correct_func_indices) > 0:
                        target = torch.tensor([correct_func_indices[0]], device=self.device)
                        if correct_func_indices[0] < func_scores.size(1):
                            loss = F.cross_entropy(func_scores, target)
                            total_loss = total_loss + loss
                            valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _train_line_with_upper_predictions(self, batch_data):
        """Train line agent using file and function agent predictions"""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        file_contents_batch = batch_data['file_contents']
        file_paths_batch = batch_data.get('file_paths', [None] * batch_size)
        samples_info = batch_data['samples_info']

        # Get file predictions (frozen)
        all_file_embeddings = []
        for file_contents, file_paths in zip(file_contents_batch, file_paths_batch):
            file_emb = self.embedder.get_file_embeddings(file_contents, file_paths=file_paths).to(self.device)
            all_file_embeddings.append(file_emb)
        
        max_files = max(emb.size(0) for emb in all_file_embeddings)
        padded_file_embeddings = []
        
        for file_emb in all_file_embeddings:
            if file_emb.size(0) < max_files:
                pad_size = max_files - file_emb.size(0)
                padding = torch.zeros(pad_size, file_emb.size(1), device=self.device)
                file_emb = torch.cat([file_emb, padding], dim=0)
            padded_file_embeddings.append(file_emb)
        
        file_embeddings = torch.stack(padded_file_embeddings)
        
        with torch.no_grad():
            file_rankings, _, _ = self.file_agent(bug_embeddings, file_embeddings, training=False)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0
        
        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            
            # Use predicted file
            selected_file_idx = file_rankings[i, 0].item()

            if selected_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][selected_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                    
                    # Get function predictions (frozen)
                    with torch.no_grad():
                        func_rankings, _, _ = self.func_agent(bug_emb, func_emb, training=False)
                    
                    # Use predicted function
                    selected_func_idx = func_rankings[0, 0].item()
                    
                    if selected_func_idx < len(candidate_functions):
                        selected_function_content = candidate_functions[selected_func_idx]
                        candidate_lines = selected_function_content.split('\n')
                        
                        if candidate_lines:
                            line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                            line_rankings, line_scores, line_confidences = self.line_agent(bug_emb, line_emb, training=True)
                            
                            # Get ground truth line indices
                            correct_line_indices = self._calculate_correct_line_indices_issue_based(sample, function_info[selected_func_idx])
                            
                            if correct_line_indices and len(correct_line_indices) > 0:
                                target = torch.tensor([correct_line_indices[0]], device=self.device)
                                if correct_line_indices[0] < line_scores.size(1):
                                    loss = F.cross_entropy(line_scores, target)
                                    total_loss = total_loss + loss
                                    valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)


    # --- Per-position REINFORCE helper ---

    def _per_position_reinforce_loss(self, bug_emb, rankings_i, scores_i,
                                     per_pos_rewards, num_candidates, critic):
        """Compute per-position REINFORCE loss for a single sample.

        Args:
            bug_emb: (1, 768) bug embedding for this sample
            rankings_i: (top_k,) ranking indices for this sample
            scores_i: (1, num_candidates) scores for this sample
            per_pos_rewards: list of per-position rewards
            num_candidates: number of real (non-padding) candidates
            critic: critic network to compute baseline

        Returns:
            sample_loss tensor, or None if no valid positions
        """
        value = critic(bug_emb)
        baseline = value.item()

        log_probs = F.log_softmax(scores_i, dim=-1)[0]
        policy_loss = torch.tensor(0.0, device=self.device)
        valid_positions = 0
        reward_sum = 0.0

        for p, reward_p in enumerate(per_pos_rewards):
            item = rankings_i[p].item()
            if item < num_candidates:
                advantage_p = reward_p - baseline
                policy_loss = policy_loss + (-log_probs[item] * advantage_p)
                reward_sum += reward_p
                valid_positions += 1

        if valid_positions > 0:
            mean_reward = reward_sum / valid_positions
            value_loss = F.mse_loss(value.squeeze(),
                                    torch.tensor(mean_reward, device=self.device))

            probs = F.softmax(scores_i, dim=-1)
            entropy = -torch.sum(probs * F.log_softmax(scores_i, dim=-1))
            entropy_bonus = self.entropy_coef * entropy

            return policy_loss + value_loss - entropy_bonus
        return None

    # --- TWF-RL batch methods ---

    def _twf_rl_file_batch(self, batch_data):
        """TWF-RL: Train file agent with REINFORCE using individual file reward."""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        file_contents_batch = batch_data['file_contents']
        file_paths_batch = batch_data.get('file_paths', [None] * batch_size)
        samples_info = batch_data['samples_info']

        # Embed files (same as _pretrain_file_batch)
        all_file_embeddings = []
        for file_contents, file_paths in zip(file_contents_batch, file_paths_batch):
            file_emb = self.embedder.get_file_embeddings(file_contents, file_paths=file_paths).to(self.device)
            all_file_embeddings.append(file_emb)

        max_files = max(emb.size(0) for emb in all_file_embeddings)
        padded_file_embeddings = []
        for file_emb in all_file_embeddings:
            if file_emb.size(0) < max_files:
                pad_size = max_files - file_emb.size(0)
                padding = torch.zeros(pad_size, file_emb.size(1), device=self.device)
                file_emb = torch.cat([file_emb, padding], dim=0)
            padded_file_embeddings.append(file_emb)
        file_embeddings = torch.stack(padded_file_embeddings)

        # Forward pass
        file_rankings, file_scores, file_confidences = self.file_agent(
            bug_embeddings, file_embeddings, training=True
        )

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0

        for i in range(batch_size):
            sample = samples_info[i]
            per_pos_rewards = self.reward_calculator.compute_per_position_rewards(
                file_rankings[i], sample, level="file"
            )
            loss = self._per_position_reinforce_loss(
                bug_embeddings[i:i+1], file_rankings[i], file_scores[i:i+1],
                per_pos_rewards, len(sample['file_contents']), self.file_critic
            )
            if loss is not None:
                total_loss = total_loss + loss
                valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _twf_rl_function_batch(self, batch_data):
        """TWF-RL: Train function agent with REINFORCE using individual func reward.
        Teacher forcing: uses ground-truth file."""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        samples_info = batch_data['samples_info']

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0

        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            correct_file_idx = sample['correct_file_idx']

            if correct_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][correct_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                    func_rankings, func_scores, func_confidences = self.func_agent(
                        bug_emb, func_emb, training=True
                    )

                    per_pos_rewards = self.reward_calculator.compute_per_position_rewards(
                        func_rankings[0], sample, level="function",
                        function_info=function_info
                    )
                    loss = self._per_position_reinforce_loss(
                        bug_emb, func_rankings[0], func_scores[0:1],
                        per_pos_rewards, len(candidate_functions), self.func_critic
                    )
                    if loss is not None:
                        total_loss = total_loss + loss
                        valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _twf_rl_line_batch(self, batch_data):
        """TWF-RL: Train line agent with REINFORCE using individual line reward.
        Teacher forcing: uses ground-truth file and ground-truth function."""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        samples_info = batch_data['samples_info']

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0

        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            correct_file_idx = sample['correct_file_idx']

            if correct_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][correct_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    correct_func_idx = self._find_correct_function_idx_issue_based(function_info, sample)

                    if correct_func_idx >= 0 and correct_func_idx < len(candidate_functions):
                        selected_function_content = candidate_functions[correct_func_idx]
                        candidate_lines = selected_function_content.split('\n')

                        if candidate_lines:
                            line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                            line_rankings, line_scores, line_confidences = self.line_agent(
                                bug_emb, line_emb, training=True
                            )

                            per_pos_rewards = self.reward_calculator.compute_per_position_rewards(
                                line_rankings[0], sample, level="line",
                                function_info=function_info,
                                selected_func_idx=correct_func_idx
                            )
                            loss = self._per_position_reinforce_loss(
                                bug_emb, line_rankings[0], line_scores[0:1],
                                per_pos_rewards, len(candidate_lines), self.line_critic
                            )
                            if loss is not None:
                                total_loss = total_loss + loss
                                valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _twf_rl_train_epoch(self, epoch, batch_method, agent, optimizer, scheduler,
                            phase_name, log_prefix, level_critic=None,
                            level_critic_optimizer=None):
        """Generic epoch loop for a single TWF-RL phase.
        level_critic / level_critic_optimizer: the per-level critic for this
        phase.  Only this critic is zeroed / stepped / clipped, avoiding
        Adam state pollution on the other two unused critics."""
        epoch_loss = 0
        num_batches = 0

        batches = self.create_batches(self.train_data, self.batch_size)
        random.shuffle(batches)

        for batch in batches:
            batch_data = self.pad_batch_data(batch)
            loss = batch_method(batch_data)

            if loss.requires_grad:
                optimizer.zero_grad()
                if level_critic_optimizer:
                    level_critic_optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                if level_critic is not None:
                    torch.nn.utils.clip_grad_norm_(level_critic.parameters(), max_norm=1.0)

                optimizer.step()
                if level_critic_optimizer:
                    level_critic_optimizer.step()

                # Per-batch epsilon decay (match VT/PT1 behaviour)
                self.update_exploration()

                epoch_loss += loss.item()
                num_batches += 1
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            scheduler.step()
            print(f"{phase_name} Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            if self.use_wandb:
                wandb.log({f"{log_prefix}/loss": avg_loss, f"{log_prefix}/epoch": epoch + 1})

    def train_twf_rl(self, epochs=20, eval_interval=5):
        """TWF-RL: Teacher-forced RL with individual REINFORCE reward signals.
        Same sequential structure as PT2 but uses REINFORCE+A2C instead of CE loss.
        Each agent is trained with its own reward signal from reward_calculator."""
        print("Training with TWF-RL Strategy (Teacher-Forced REINFORCE)...")

        # Phase 1: File agent with individual file reward
        print("Phase 1: Training File Agent (REINFORCE, file reward)")
        self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.file_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._twf_rl_train_epoch(
                epoch, self._twf_rl_file_batch, self.file_agent,
                self.file_optimizer, self.file_scheduler,
                "TWF-RL File", "twf_rl/file",
                level_critic=self.file_critic,
                level_critic_optimizer=self.file_critic_optimizer
            )
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 1 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                if self.use_wandb:
                    wandb.log({
                        **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/step": epoch
                    })
                print(f"File Agent Epoch {epoch+1} Metrics: {metrics}")
                self.save_agent()

        # Phase 2: Function agent with individual func reward (GT file)
        print("Phase 2: Training Function Agent (REINFORCE, func reward, GT file)")
        self._reset_exploration()  # new agent needs fresh exploration schedule
        self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.func_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._twf_rl_train_epoch(
                epoch, self._twf_rl_function_batch, self.func_agent,
                self.func_optimizer, self.func_scheduler,
                "TWF-RL Func", "twf_rl/func",
                level_critic=self.func_critic,
                level_critic_optimizer=self.func_critic_optimizer
            )
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 2 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                if self.use_wandb:
                    wandb.log({
                        **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/step": epoch
                    })
                print(f"Function Agent Epoch {epoch+1} Metrics: {metrics}")
                self.save_agent()

        # Phase 3: Line agent with individual line reward (GT file + GT func)
        print("Phase 3: Training Line Agent (REINFORCE, line reward, GT file + GT func)")
        self._reset_exploration()  # new agent needs fresh exploration schedule
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._twf_rl_train_epoch(
                epoch, self._twf_rl_line_batch, self.line_agent,
                self.line_optimizer, self.line_scheduler,
                "TWF-RL Line", "twf_rl/line",
                level_critic=self.line_critic,
                level_critic_optimizer=self.line_critic_optimizer
            )
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 3 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                if self.use_wandb:
                    wandb.log({
                        **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/step": epoch
                    })
                print(f"Line Agent Epoch {epoch+1} Metrics: {metrics}")
            self.save_agent()

    # --- T1B1-RL batch methods ---

    def _t1b1_rl_function_batch(self, batch_data):
        """T1B1-RL: Train function agent with REINFORCE using frozen file agent's predictions."""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        file_contents_batch = batch_data['file_contents']
        file_paths_batch = batch_data.get('file_paths', [None] * batch_size)
        samples_info = batch_data['samples_info']

        # Embed files
        all_file_embeddings = []
        for file_contents, file_paths in zip(file_contents_batch, file_paths_batch):
            file_emb = self.embedder.get_file_embeddings(file_contents, file_paths=file_paths).to(self.device)
            all_file_embeddings.append(file_emb)

        max_files = max(emb.size(0) for emb in all_file_embeddings)
        padded_file_embeddings = []
        for file_emb in all_file_embeddings:
            if file_emb.size(0) < max_files:
                pad_size = max_files - file_emb.size(0)
                padding = torch.zeros(pad_size, file_emb.size(1), device=self.device)
                file_emb = torch.cat([file_emb, padding], dim=0)
            padded_file_embeddings.append(file_emb)
        file_embeddings = torch.stack(padded_file_embeddings)

        # Run frozen file agent
        with torch.no_grad():
            file_rankings, _, _ = self.file_agent(bug_embeddings, file_embeddings, training=False)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0

        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            selected_file_idx = file_rankings[i, 0].item()

            if selected_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][selected_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)
                    func_rankings, func_scores, func_confidences = self.func_agent(
                        bug_emb, func_emb, training=True
                    )

                    per_pos_rewards = self.reward_calculator.compute_per_position_rewards(
                        func_rankings[0], sample, level="function",
                        function_info=function_info
                    )
                    loss = self._per_position_reinforce_loss(
                        bug_emb, func_rankings[0], func_scores[0:1],
                        per_pos_rewards, len(candidate_functions), self.func_critic
                    )
                    if loss is not None:
                        total_loss = total_loss + loss
                        valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _t1b1_rl_line_batch(self, batch_data):
        """T1B1-RL: Train line agent with REINFORCE using frozen file+func agents' predictions."""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        file_contents_batch = batch_data['file_contents']
        file_paths_batch = batch_data.get('file_paths', [None] * batch_size)
        samples_info = batch_data['samples_info']

        # Embed files
        all_file_embeddings = []
        for file_contents, file_paths in zip(file_contents_batch, file_paths_batch):
            file_emb = self.embedder.get_file_embeddings(file_contents, file_paths=file_paths).to(self.device)
            all_file_embeddings.append(file_emb)

        max_files = max(emb.size(0) for emb in all_file_embeddings)
        padded_file_embeddings = []
        for file_emb in all_file_embeddings:
            if file_emb.size(0) < max_files:
                pad_size = max_files - file_emb.size(0)
                padding = torch.zeros(pad_size, file_emb.size(1), device=self.device)
                file_emb = torch.cat([file_emb, padding], dim=0)
            padded_file_embeddings.append(file_emb)
        file_embeddings = torch.stack(padded_file_embeddings)

        # Run frozen file agent
        with torch.no_grad():
            file_rankings, _, _ = self.file_agent(bug_embeddings, file_embeddings, training=False)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0

        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]
            selected_file_idx = file_rankings[i, 0].item()

            if selected_file_idx < len(sample['file_contents']):
                selected_file_content = sample['file_contents'][selected_file_idx]
                candidate_functions, function_info = get_all_function_candidates(selected_file_content)

                if candidate_functions:
                    func_emb = self.embedder.get_function_embeddings(candidate_functions).to(self.device).unsqueeze(0)

                    # Run frozen func agent
                    with torch.no_grad():
                        func_rankings, _, _ = self.func_agent(bug_emb, func_emb, training=False)

                    selected_func_idx = func_rankings[0, 0].item()

                    if selected_func_idx < len(candidate_functions):
                        selected_function_content = candidate_functions[selected_func_idx]
                        candidate_lines = selected_function_content.split('\n')

                        if candidate_lines:
                            line_emb = self.embedder.get_line_embeddings(candidate_lines).to(self.device).unsqueeze(0)
                            line_rankings, line_scores, line_confidences = self.line_agent(
                                bug_emb, line_emb, training=True
                            )

                            per_pos_rewards = self.reward_calculator.compute_per_position_rewards(
                                line_rankings[0], sample, level="line",
                                function_info=function_info,
                                selected_func_idx=selected_func_idx
                            )
                            loss = self._per_position_reinforce_loss(
                                bug_emb, line_rankings[0], line_scores[0:1],
                                per_pos_rewards, len(candidate_lines), self.line_critic
                            )
                            if loss is not None:
                                total_loss = total_loss + loss
                                valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_t1b1_rl(self, epochs=20, eval_interval=5):
        """T1B1-RL: Top-down REINFORCE with freezing.
        Same sequential freeze structure as CE T1B1 but uses REINFORCE+A2C instead of CE loss.
        Each agent is trained with its own reward signal from reward_calculator."""
        print("Training with T1B1-RL Strategy (Top-Down REINFORCE with Freezing)...")

        # Phase 1: File agent with individual file reward (same as TWF phase 1)
        print("Phase 1: Training File Agent (REINFORCE, file reward)")
        self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.file_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._twf_rl_train_epoch(
                epoch, self._twf_rl_file_batch, self.file_agent,
                self.file_optimizer, self.file_scheduler,
                "T1B1-RL File", "t1b1_rl/file",
                level_critic=self.file_critic,
                level_critic_optimizer=self.file_critic_optimizer
            )
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 1 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                if self.use_wandb:
                    wandb.log({
                        **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/step": epoch
                    })
                print(f"File Agent Epoch {epoch+1} Metrics: {metrics}")
                self.save_agent()

        # Restore best file agent before freezing
        strategy_suffix = f"_{self.training_strategy}" if self.training_strategy else ""
        reward_suffix = f"_{self.reward_calculator.reward_type}" if hasattr(self, 'reward_calculator') else ""
        seed_suffix = f"_seed{self.seed}" if self.seed is not None else ""
        best_file_path = os.path.join(self.checkpoint_dir, f"file_agent{strategy_suffix}{reward_suffix}{seed_suffix}_best_file.pt")
        if os.path.exists(best_file_path):
            self.file_agent.load_state_dict(torch.load(best_file_path, map_location=self.device, weights_only=True))
            print(f"Restored best file agent from {best_file_path} (file_top1={self.best_file_top1:.4f})")

        # Freeze file agent
        for param in self.file_agent.parameters():
            param.requires_grad = False
        print("File agent frozen")

        # Phase 2: Function agent with frozen file agent (REINFORCE)
        print("Phase 2: Training Function Agent (REINFORCE, frozen file agent)")
        self._reset_exploration()  # new agent needs fresh exploration schedule
        self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.func_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._twf_rl_train_epoch(
                epoch, self._t1b1_rl_function_batch, self.func_agent,
                self.func_optimizer, self.func_scheduler,
                "T1B1-RL Func", "t1b1_rl/func",
                level_critic=self.func_critic,
                level_critic_optimizer=self.func_critic_optimizer
            )
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 2 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                if self.use_wandb:
                    wandb.log({
                        **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/step": epoch
                    })
                print(f"Function Agent Epoch {epoch+1} Metrics: {metrics}")
                self.save_agent()

        # Restore best func agent before freezing
        best_func_path = os.path.join(self.checkpoint_dir, f"function_agent{strategy_suffix}{reward_suffix}{seed_suffix}_best_oracle_func.pt")
        if os.path.exists(best_func_path):
            self.func_agent.load_state_dict(torch.load(best_func_path, map_location=self.device, weights_only=True))
            print(f"Restored best func agent from {best_func_path} (oracle_func={self.best_oracle_func:.4f})")

        # Freeze function agent
        for param in self.func_agent.parameters():
            param.requires_grad = False
        print("Function agent frozen")

        # Phase 3: Line agent with frozen file and function agents (REINFORCE)
        print("Phase 3: Training Line Agent (REINFORCE, frozen file + func agents)")
        self._reset_exploration()  # new agent needs fresh exploration schedule
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._twf_rl_train_epoch(
                epoch, self._t1b1_rl_line_batch, self.line_agent,
                self.line_optimizer, self.line_scheduler,
                "T1B1-RL Line", "t1b1_rl/line",
                level_critic=self.line_critic,
                level_critic_optimizer=self.line_critic_optimizer
            )
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Phase 3 Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                if self.use_wandb:
                    wandb.log({
                        **{f"quick_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/step": epoch
                    })
                print(f"Line Agent Epoch {epoch+1} Metrics: {metrics}")
            self.save_agent()

    # --- Flat hierarchy ablation methods ---

    def _flat_line_batch(self, batch_data):
        """Flat hierarchy ablation: train line agent on flat pool of all candidate file lines."""
        batch_size = batch_data['batch_size']
        bug_embeddings = batch_data['bug_embeddings']
        samples_info = batch_data['samples_info']

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0

        for i in range(batch_size):
            sample = samples_info[i]
            bug_emb = bug_embeddings[i:i+1]

            if sample['correct_file_idx'] < 0:
                continue

            pool_lines, _, _, correct_pool_indices = build_flat_line_pool(sample)
            if not correct_pool_indices:
                continue

            line_emb = self.embedder.get_line_embeddings(pool_lines).to(self.device).unsqueeze(0)
            line_rankings, line_scores, _ = self.line_agent(bug_emb, line_emb, training=True)

            per_pos_rewards = self.reward_calculator.compute_per_position_rewards_flat(
                line_rankings[0], set(correct_pool_indices)
            )
            loss = self._per_position_reinforce_loss(
                bug_emb, line_rankings[0], line_scores[0:1],
                per_pos_rewards, len(pool_lines), self.line_critic
            )
            if loss is not None:
                total_loss = total_loss + loss
                valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def evaluate_flat(self):
        """In-training evaluation for flat hierarchy ablation."""
        self.line_agent.eval()
        total_samples = 0
        line_top1_correct = 0
        line_top5_correct = 0
        line_top15_correct = 0
        line_rankings_list = []
        line_correct_indices_list = []

        with torch.no_grad():
            for sample in self.test_data:
                try:
                    if sample['correct_file_idx'] < 0:
                        continue
                    pool_lines, _, _, correct_pool_indices = build_flat_line_pool(sample)
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
                    if any(r in correct_pool_indices for r in ranking[:15]):
                        line_top15_correct += 1

                    total_samples += 1
                except Exception:
                    continue

        line_mrr = self.reward_calculator.compute_mrr_multi_target(
            line_rankings_list, line_correct_indices_list
        )
        self.line_agent.train()
        return {
            'line_mrr': line_mrr,
            'line_hit_at_1': line_top1_correct / total_samples if total_samples > 0 else 0.0,
            'line_hit_at_5': line_top5_correct / total_samples if total_samples > 0 else 0.0,
            'line_hit_at_15': line_top15_correct / total_samples if total_samples > 0 else 0.0,
            'total_samples': total_samples,
        }

    def train_flat(self, epochs=40, eval_interval=1):
        """Flat hierarchy ablation: train only the line agent on a flat pool of all lines."""
        print("Training with Flat Strategy (single line agent, no hierarchy)...")
        self._reset_exploration()
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=epochs, eta_min=1e-6)
        for epoch in range(epochs):
            epoch_start = time.time()
            self._twf_rl_train_epoch(
                epoch, self._flat_line_batch, self.line_agent,
                self.line_optimizer, self.line_scheduler,
                "Flat Line", "flat/line",
                level_critic=self.line_critic,
                level_critic_optimizer=self.line_critic_optimizer
            )
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  Flat Epoch {epoch+1} wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")
            if self.use_wandb:
                wandb.log({"train/epoch_wall_time_sec": epoch_time, "train/peak_gpu_mem_gb": peak_mem})
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate_flat()
                if self.use_wandb:
                    wandb.log({
                        **{f"flat_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/step": epoch
                    })
                print(f"Flat Line Agent Epoch {epoch+1} Metrics: {metrics}")
                line_mrr = metrics.get('line_mrr', 0.0)
                line_top1 = metrics.get('line_hit_at_1', 0.0)
                eps = 1e-9
                line_improved = (
                    line_top1 > self.best_oracle_line + eps
                    or (abs(line_top1 - self.best_oracle_line) <= eps
                        and line_mrr > self.best_oracle_line_mrr + eps)
                    or (abs(line_top1 - self.best_oracle_line) <= eps
                        and abs(line_mrr - self.best_oracle_line_mrr) <= eps)
                )
                if line_improved:
                    self.best_oracle_line = line_top1
                    self.best_oracle_line_mrr = line_mrr
                    self.save_agent(tag='best_oracle_line', agents=['line'])
            self.save_agent(agents=['line'])

    def save_agent(self, tag=None, agents=None):
        """Save the trained agents with training strategy and seed identifiers.
        tag: optional suffix like 'best_oracle_func' for best-checkpoint files.
        agents: list subset of ['file','func','line']. Default None = save all."""
        strategy_suffix = f"_{self.training_strategy}" if self.training_strategy else ""
        seed_suffix = f"_seed{self.seed}" if self.seed is not None else ""
        # Include reward type in checkpoint filename to prevent collision
        # when running different reward types concurrently with the same strategy+seed.
        reward_suffix = ""
        if hasattr(self, 'reward_calculator'):
            reward_suffix = f"_{self.reward_calculator.reward_type}"
        # Include encoder type so MLP ablation checkpoints don't collide with LSTM checkpoints
        encoder_suffix = "_mlp" if getattr(self, 'encoder_type', 'lstm') == 'mlp' else ""
        tag_suffix = f"_{tag}" if tag else ""
        save_all = agents is None

        if save_all or 'file' in agents:
            path = os.path.join(self.checkpoint_dir, f"file_agent{strategy_suffix}{reward_suffix}{encoder_suffix}{seed_suffix}{tag_suffix}.pt")
            torch.save(self.file_agent.state_dict(), path)
        if save_all or 'func' in agents:
            path = os.path.join(self.checkpoint_dir, f"function_agent{strategy_suffix}{reward_suffix}{encoder_suffix}{seed_suffix}{tag_suffix}.pt")
            torch.save(self.func_agent.state_dict(), path)
        if save_all or 'line' in agents:
            path = os.path.join(self.checkpoint_dir, f"line_agent{strategy_suffix}{reward_suffix}{encoder_suffix}{seed_suffix}{tag_suffix}.pt")
            torch.save(self.line_agent.state_dict(), path)

        print(f"Agents saved: tag={tag}, agents={agents or 'all'}")        

    # Main training method that dispatches to appropriate strategy
    def train(self, epochs=20, eval_interval=5, pretrain_epochs=10):
        """Main training method that selects strategy based on training_strategy parameter"""
        
        if self.training_strategy == "vt":
            # Vanilla training (current method)
            print("Using Vanilla Training Strategy (VT)")
            self._train_vanilla(epochs, eval_interval)
            
        elif self.training_strategy == "pt1":
            # Pretraining then joint training
            print("Using Pretraining Strategy (PT1)")
            self.pretrain_file_agent(pretrain_epochs)
            self.pretrain_function_agent(pretrain_epochs)
            self.pretrain_line_agent(pretrain_epochs)
            # Reset schedulers before joint training so the cosine restarts
            # from the initial LR (pretrain and joint are separate phases).
            self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.file_optimizer, T_max=epochs, eta_min=1e-6)
            self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.func_optimizer, T_max=epochs, eta_min=1e-6)
            self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.line_optimizer, T_max=epochs, eta_min=1e-6)
            print("Pretraining completed. Starting joint training...")
            self._train_vanilla(epochs, eval_interval)
            
        elif self.training_strategy == "pt2":
            # Sequential ground truth training
            print("Using Sequential Ground Truth Strategy (PT2)")
            self.train_sequential_ground_truth(epochs, eval_interval)
            
        elif self.training_strategy == "t1b1":
            # Top-down with freezing
            print("Using Top-Down with Freezing Strategy (T1B1)")
            self.train_top_down_with_freezing(epochs, eval_interval)

        elif self.training_strategy == "twf":
            # Teacher-forced RL with individual REINFORCE reward signals
            print("Using TWF-RL Strategy (Teacher-Forced REINFORCE)")
            self.train_twf_rl(epochs, eval_interval)

        elif self.training_strategy == "t1b1rl":
            # Top-down RL with freezing (RL counterpart of CE t1b1)
            print("Using T1B1-RL Strategy (Top-Down REINFORCE with Freezing)")
            self.train_t1b1_rl(epochs, eval_interval)

        elif self.training_strategy == "flat":
            # Flat hierarchy ablation: single line agent over all candidate lines
            print("Using Flat Strategy (hierarchy ablation: single line agent)")
            self.train_flat(epochs, eval_interval)

        else:
            raise ValueError(f"Unknown training strategy: {self.training_strategy}")

    def _train_vanilla(self, epochs=20, eval_interval=5):
        """Original vanilla training method"""
        print("Starting Ranking-based HRL Training with Batching...")
        self.file_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.file_optimizer, T_max=epochs, eta_min=1e-6)
        self.func_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.func_optimizer, T_max=epochs, eta_min=1e-6)
        self.line_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.line_optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0
            epoch_reward = 0
            num_batches = 0
            
            # Create batches
            batches = self.create_batches(self.train_data, self.batch_size)
            random.shuffle(batches)
            
            for batch_idx, batch in enumerate(batches):
                # Pad batch data
                batch_data = self.pad_batch_data(batch)
                
                # Train on batch
                loss, reward = self.train_batch(batch_data)
                
                if loss.requires_grad:
                    # Backward pass
                    self.file_optimizer.zero_grad()
                    self.func_optimizer.zero_grad()
                    self.line_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    
                    loss.backward()
                    
                    # Gradient clipping
                    file_grad_norm = torch.nn.utils.clip_grad_norm_(self.file_agent.parameters(), max_norm=1.0)
                    func_grad_norm = torch.nn.utils.clip_grad_norm_(self.func_agent.parameters(), max_norm=1.0)
                    line_grad_norm = torch.nn.utils.clip_grad_norm_(self.line_agent.parameters(), max_norm=1.0)
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                    context_grad_norm = torch.nn.utils.clip_grad_norm_(self.context_encoder.parameters(), max_norm=1.0)
                    
                    self.file_optimizer.step()
                    self.func_optimizer.step()
                    self.line_optimizer.step()
                    self.critic_optimizer.step()

                    # Per-batch epsilon decay (VT/PT1 only)
                    self.update_exploration()

                    epoch_loss += loss.item()
                    epoch_reward += reward
                    num_batches += 1

                    # Log training metrics to Wandb
                    if self.use_wandb and (batch_idx + 1) % 10 == 0:
                        wandb.log({
                            "train/batch_loss": loss.item(),
                            "train/batch_reward": reward,
                            "train/epoch": epoch + 1,
                            "train/batch": batch_idx + 1,
                            "train/epsilon": self.epsilon,
                            "train/temperature": self.temperature,
                            "gradients/file_grad_norm": file_grad_norm.item(),
                            "gradients/func_grad_norm": func_grad_norm.item(),
                            "gradients/line_grad_norm": line_grad_norm.item(),
                            "gradients/critic_grad_norm": critic_grad_norm.item(),
                            "gradients/context_grad_norm": context_grad_norm.item(),
                        })

                # Print progress and quick eval
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(batches)}, "
                          f"Avg Loss: {epoch_loss/max(num_batches, 1):.4f}, "
                          f"Avg Reward: {epoch_reward/max(num_batches, 1):.4f}")
                    
                    # Quick evaluation on small subset
                    quick_metrics = self.evaluate(quick_eval=True)
                    print(f"Quick Eval Metrics: {quick_metrics}")
                    if self.use_wandb:
                        wandb.log({
                            **{f"quick_eval/{k}": v for k, v in quick_metrics.items() if isinstance(v, (int, float))},
                            "train/step": epoch * len(batches) + batch_idx + 1
                        })
                        print(quick_metrics)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_reward = epoch_reward / num_batches
                self.file_scheduler.step()
                self.func_scheduler.step()
                self.line_scheduler.step()
                print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Reward = {avg_reward:.4f}, "
                      f"Epsilon = {self.epsilon:.4f}, Temperature = {self.temperature:.4f}")
                print(f"  Epoch wall time: {epoch_time:.0f}s | Peak GPU: {peak_mem:.1f} GB")

                # Log epoch summary
                if self.use_wandb:
                    wandb.log({
                        "train/epoch_loss": avg_loss,
                        "train/epoch_reward": avg_reward,
                        "train/epoch_num": epoch + 1,
                        "exploration/epsilon": self.epsilon,
                        "exploration/temperature": self.temperature,
                        "train/epoch_wall_time_sec": epoch_time,
                        "train/peak_gpu_mem_gb": peak_mem,
                    })
            
            # Full evaluation
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate(quick_eval=False)
                print(f"Full Evaluation Metrics: {metrics}")
                
                # Log full evaluation metrics
                if self.use_wandb:
                    wandb.log({
                        **{f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "train/epoch": epoch + 1
                    })
                self.save_agent()  # Save after each eval interval