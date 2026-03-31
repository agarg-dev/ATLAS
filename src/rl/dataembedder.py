import ast
import torch
from transformers import RobertaTokenizer, RobertaModel
import re
import torch.nn.functional as F
import numpy as np

def _extract_functions_python(file_content):
    """Extract Python functions/methods using the ast module."""
    try:
        tree = ast.parse(file_content)
        lines = file_content.split('\n')
        functions = []
        function_info = []
        nodes = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        nodes.sort(key=lambda n: n.lineno)  # ast.walk order is arbitrary
        for node in nodes:
            start_line = node.lineno
            end_line = node.end_lineno
            func_content = '\n'.join(lines[start_line - 1:end_line])
            functions.append(func_content)
            function_info.append({'name': node.name, 'start_line': start_line, 'end_line': end_line})
        return functions, function_info
    except SyntaxError:
        return [], []
    except Exception as e:
        print(f"Error extracting Python functions: {e}")
        return [], []

def extract_functions_regex(file_content):
    # Try Python AST first; fall back to Java brace-matching regex
    python_funcs, python_info = _extract_functions_python(file_content)
    if python_funcs:
        return python_funcs, python_info
    try:
        method_pattern = r'((?:public|private|protected|static|final|abstract|synchronized|\s)\s+[\w\<\>\[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{)'
        functions = []
        function_info = []
        for match in re.finditer(method_pattern, file_content):
            method_start = match.start()
            method_name = match.group(2)
            start_line = file_content[:method_start].count('\n') + 1
            open_braces = 0
            close_pos = match.end()
            for i in range(match.end(), len(file_content)):
                if file_content[i] == '{':
                    open_braces += 1
                elif file_content[i] == '}':
                    if open_braces == 0:
                        close_pos = i + 1
                        break
                    open_braces -= 1
            end_line = file_content[:close_pos].count('\n') + 1
            function_content = file_content[match.start():close_pos]
            functions.append(function_content)
            function_info.append({'name': method_name, 'start_line': start_line, 'end_line': end_line})
        return functions, function_info
    except Exception as e:
        print(f"Error extracting functions: {e}")
        return [], []

def extract_global_code(file_content, function_info):
    try:
        lines = file_content.split('\n')
        function_line_ranges = [(info['start_line'], info['end_line']) for info in function_info]
        global_lines = []
        for i, line in enumerate(lines, 1):
            if not any(start <= i <= end for start, end in function_line_ranges):
                global_lines.append(line)
        return '\n'.join(global_lines)
    except Exception as e:
        print(f"Error extracting global code: {e}")
        return ""


def get_all_function_candidates(file_content):
    """Extract named functions + global code as candidates.

    Always includes global code (if non-empty) as the last candidate,
    even when named functions exist. This ensures module-level and
    class-level buggy code has a valid candidate to match against.
    """
    functions, function_info = extract_functions_regex(file_content)
    global_code = extract_global_code(file_content, function_info)
    if global_code.strip():
        n_global_lines = len(global_code.split('\n'))
        functions = functions + [global_code]
        function_info = function_info + [
            {'name': 'global', 'start_line': 1, 'end_line': n_global_lines}
        ]
    return functions, function_info


class _StripDocstrings(ast.NodeTransformer):
    """AST transformer that removes docstrings from modules, classes, and functions.
    Identical to the copy in train_bert.py / build_faiss_index.py."""
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


def _filter_python_for_embedding(source):
    """Remove docstrings and # comments from Python source via AST round-trip.
    Falls back to raw source on SyntaxError (e.g. Python 2 syntax).
    Identical to filter_python_for_embedding() in train_bert.py / build_faiss_index.py."""
    try:
        tree = ast.parse(source)
        tree = _StripDocstrings().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return source


def _filter_java_for_embedding(text):
    """Strip block comments, line comments, imports, package decls, and blank lines from Java source.
    Identical to filter_java_for_embedding() in train_bert.py / build_faiss_index.py."""
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


def filter_code_for_embedding(text):
    """Strip comments, imports, and docstrings. Auto-detects Python vs Java by trying ast.parse."""
    if not text:
        return text
    # Auto-detect: if it parses as Python, use AST-based filtering
    try:
        ast.parse(text)
        return _filter_python_for_embedding(text)
    except (SyntaxError, ValueError):
        return _filter_java_for_embedding(text)


class CodeBERTEmbedder:
    def __init__(self,
                 tokenizer_path="../../models/codebert/codebert_tokenizer",
                 model_ckpt_path="../../models/codebert/best_codebert_triplet.pt",
                 device="cuda",
                 use_path=False):
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
        self.use_path = use_path
    
    def _sliding_window_embed(self, text, path="", max_tokens=512, max_chunks=50):
        """Optimized sliding window embedding with batching.
        When path is provided, prepends '# path' tokens to every chunk
        (matching train_bert.py / build_faiss_index.py behavior)."""
        if not text:
            return torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)

        path_ids = self.tokenizer.encode(f"# {path}", add_special_tokens=False) if path else []
        content_budget = max(1, max_tokens - 2 - len(path_ids))

        input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        chunks = [input_ids[i:i + content_budget] for i in range(0, len(input_ids), content_budget)]
        chunks = chunks[:max_chunks]

        if not chunks:
            return torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)

        # Prepare batched input
        batch_input_ids = []
        batch_attention_mask = []

        for chunk in chunks:
            ids = [self.tokenizer.cls_token_id] + path_ids + chunk + [self.tokenizer.sep_token_id]
            mask = [1] * len(ids)
            pad_len = max_tokens - len(ids)
            ids += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
        
        # Convert to tensors
        batch_input_ids = torch.tensor(batch_input_ids, device=self.device)
        batch_attention_mask = torch.tensor(batch_attention_mask, device=self.device)
        
        # Single forward pass for all chunks
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type if self.device.type == "cuda" else "cpu"):
                output = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                cls_embeddings = output.last_hidden_state[:, 0, :]  # Shape: (num_chunks, hidden_size)
        
        # Mean pooling across chunks
        mean_pooled = cls_embeddings.mean(dim=0)
        return F.normalize(mean_pooled, dim=0)
    
    def _batch_sliding_window_embed(self, texts, paths=None, max_tokens=512, max_chunks=50, batch_size=32):
        """Batch multiple texts together for even faster processing."""
        if not texts:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32, device=self.device)

        all_embeddings = []

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_paths = paths[i:i + batch_size] if paths else None
            batch_embeddings = self._process_text_batch(batch_texts, batch_paths, max_tokens, max_chunks)
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)
    
    def _process_text_batch(self, texts, paths=None, max_tokens=512, max_chunks=50):
        """Process a batch of texts with sliding window.
        When paths are provided, prepends '# path' tokens to every chunk."""
        all_chunks = []
        chunk_text_indices = []

        for text_idx, text in enumerate(texts):
            path = paths[text_idx] if paths else ""
            path_ids = self.tokenizer.encode(f"# {path}", add_special_tokens=False) if path else []
            content_budget = max(1, max_tokens - 2 - len(path_ids))

            if not text:
                # Handle empty text
                all_chunks.append([self.tokenizer.cls_token_id] + path_ids + [self.tokenizer.sep_token_id])
                chunk_text_indices.append(text_idx)
                continue

            input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            chunks = [input_ids[i:i + content_budget] for i in range(0, len(input_ids), content_budget)]
            chunks = chunks[:max_chunks]

            if not chunks:
                # Handle case where tokenization failed
                all_chunks.append([self.tokenizer.cls_token_id] + path_ids + [self.tokenizer.sep_token_id])
                chunk_text_indices.append(text_idx)
                continue

            for chunk in chunks:
                ids = [self.tokenizer.cls_token_id] + path_ids + chunk + [self.tokenizer.sep_token_id]
                all_chunks.append(ids)
                chunk_text_indices.append(text_idx)
        
        if not all_chunks:
            return torch.zeros((len(texts), self.embedding_dim), dtype=torch.float32, device=self.device)
        
        # Pad all chunks to same length
        max_len = min(max_tokens, max(len(chunk) for chunk in all_chunks))
        batch_input_ids = []
        batch_attention_mask = []
        
        for chunk in all_chunks:
            mask = [1] * len(chunk)
            pad_len = max_len - len(chunk)
            chunk += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len
            batch_input_ids.append(chunk)
            batch_attention_mask.append(mask)
        
        # Convert to tensors
        batch_input_ids = torch.tensor(batch_input_ids, device=self.device)
        batch_attention_mask = torch.tensor(batch_attention_mask, device=self.device)
        
        # Single forward pass for all chunks
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type if self.device.type == "cuda" else "cpu"):
                output = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                cls_embeddings = output.last_hidden_state[:, 0, :]
        
        # Group embeddings by original text and mean pool
        text_embeddings = []
        for text_idx in range(len(texts)):
            chunk_indices = [i for i, idx in enumerate(chunk_text_indices) if idx == text_idx]
            if chunk_indices:
                text_chunks = cls_embeddings[chunk_indices]
                mean_pooled = text_chunks.mean(dim=0)
                text_embeddings.append(F.normalize(mean_pooled, dim=0))
            else:
                text_embeddings.append(torch.zeros(self.embedding_dim, device=self.device))
        
        return torch.stack(text_embeddings)
    
    def embed_stack_trace(self, stack_trace):
        return self._sliding_window_embed(stack_trace)
    
    def get_bug_embedding(self, bug_report: str) -> torch.Tensor:
        return self._sliding_window_embed(bug_report)

    def get_file_embeddings(self, candidate_files: list, file_paths: list = None) -> torch.Tensor:
        filtered = [filter_code_for_embedding(f) for f in candidate_files]
        paths = file_paths if (self.use_path and file_paths) else None
        return self._batch_sliding_window_embed(filtered, paths=paths, batch_size=128)

    def get_function_embeddings(self, candidate_functions: list) -> torch.Tensor:
        filtered = [filter_code_for_embedding(f) for f in candidate_functions]
        return self._batch_sliding_window_embed(filtered, batch_size=128)

    def get_line_embeddings(self, candidate_lines: list) -> torch.Tensor:
        return self._batch_sliding_window_embed(candidate_lines, batch_size=128)