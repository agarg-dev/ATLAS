import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset
import torch.utils.checkpoint


def _parse_torch_mm():
    """Return (major, minor) from ``torch.__version__`` for coarse comparisons."""
    ver = torch.__version__.split("+")[0].split(".")
    try:
        return int(ver[0]), int(ver[1])
    except (ValueError, IndexError):
        return 0, 0


def cuda_kernels_available() -> bool:
    """True if a simple CUDA kernel runs on device 0 (GPU too new for the wheel ⇒ False)."""
    return _cuda_runtime_works()


def _cuda_runtime_works() -> bool:
    """
    ``torch.cuda.is_available()`` can be true even when this PyTorch build has no kernels for the
    GPU (e.g. Blackwell sm_120 with an older wheel). Probe with a tiny matmul.
    """
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(8, 8, device="cuda", dtype=torch.float16)
        torch.mm(t, t)
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def _use_cuda_for_codegen() -> bool:
    force = os.environ.get("LLMAO_DEVICE", "").strip().lower()
    if force in ("cpu", "force_cpu"):
        return False
    return _cuda_runtime_works()


def _require_torch_for_bin_weights():
    """
    Hugging Face ``transformers`` (recent versions) refuses ``torch.load`` on ``.bin`` checkpoints
    unless PyTorch is at least 2.6 (CVE-2025-32434). Salesforce CodeGen repos only publish
    ``pytorch_model.bin``, not ``model.safetensors``.
    """
    major, minor = _parse_torch_mm()
    if (major, minor) < (2, 6):
        raise RuntimeError(
            "Your PyTorch is too old for this transformers release when loading CodeGen "
            f"({torch.__version__}; need >= 2.6). The hub weights are pytorch_model.bin only.\n\n"
            "Upgrade PyTorch, e.g. (pick one index that matches your setup):\n"
            "  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n\n"
            "See https://pytorch.org/get-started/locally/\n"
            "If you use an NVIDIA RTX 50-series GPU (sm_120), you may need CUDA 12.8+ builds or a "
            "recent nightly—see CUSTOM_DATA.md (Troubleshooting)."
        )


class VoltronDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        for i, code_file_path in enumerate(os.listdir(data_root)):
            if i > 1:
                break
            with open(os.path.join(data_root, code_file_path), 'r') as code_file:
                for code_block in code_file.read().splitlines():
                    encoded = [int(x) for x in code_block.split('\t')]
                    self.samples.append(encoded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CodeGenPass():
    """Loads Salesforce CodeGen; sets ``codegen_device`` to ``\"cuda:0\"`` or ``\"cpu\"`` after ``setup_model``."""

    def __init__(self):
        self.codegen_device = "cuda:0"

    def collate_batch(self, batch):
        # Padds batch of variable length
        tensor_batch = [torch.tensor(x) for x in batch]
        max_len = max([x.squeeze().numel() for x in tensor_batch])
        padded_batch = [torch.nn.functional.pad(x, pad=(
            0, max_len - x.numel()), mode='constant', value=0) for x in tensor_batch]
        padded_batch = torch.stack(padded_batch)
        return padded_batch

    def setup_model(self, type):
        print('Loading codegen model ...')
        starcoder = "bigcode/starcoder"
        codegen = f"Salesforce/codegen-{type}-multi"
        codegen_token = "Salesforce/codegen-350M-mono"

        _require_torch_for_bin_weights()

        use_cuda = _use_cuda_for_codegen()
        self.codegen_device = "cuda:0" if use_cuda else "cpu"
        if not use_cuda and torch.cuda.is_available():
            print(
                "LLMAO: CUDA is visible but this PyTorch build cannot run kernels on your GPU "
                "(common with RTX 50-series / sm_120 until you install a matching wheel). "
                "Using CPU for CodeGen — expect very slow extraction. "
                "Set LLMAO_DEVICE=cpu to hide this message, or upgrade PyTorch per pytorch.org."
            )

        if use_cuda:
            dtype, device_map = torch.bfloat16, "balanced"
        else:
            dtype, device_map = torch.float32, "cpu"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                codegen,
                output_hidden_states=True,
                dtype=dtype,
                device_map=device_map,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                codegen,
                output_hidden_states=True,
                torch_dtype=dtype,
                device_map=device_map,
            )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            codegen_token, fp16=True)
        print('Finished loading')
        return model, tokenizer

    def get_hidden_state(self, decoded_program=None, model=None, tokenizer=None, device=None):
        nl_replacement = '\n'
        if not isinstance(decoded_program, str):
            decoded_program = " ".join(decoded_program)
            if len(decoded_program) > 2048:
                decoded_program = decoded_program[:2048]

        decoded_program = decoded_program.replace(
            '#TAB#', '\t').replace('#NL#', nl_replacement)
        input_ids = tokenizer(
            decoded_program, return_tensors='pt').input_ids.to(device)

        # nl_ids = tokenizer(
        #     '\n', return_tensors='pt').input_ids.to(device)
        # print('nl id: ', nl_ids)
        nl_indices = torch.where(input_ids == 198)

        try:
            outputs = model(input_ids=input_ids)
        except:
            return
        hidden_states = outputs[2]
        attention_hidden_states = hidden_states[1:]
        final_attention_states = attention_hidden_states[-1]
        nl_final_attention_states = final_attention_states[torch.arange(
            final_attention_states.size(0)), nl_indices[1]]
        # project fix number (dense to 1024)
        return nl_final_attention_states, len(nl_indices[1])


if __name__ == '__main__':
    codegen_trainer = CodeGenPass()
    nl_final_attention_states = codegen_trainer.get_hidden_state_local()
    print('\n\n\n'+'done\n\n\n')
