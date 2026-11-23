"""Start the same pinned BF16/Lightning server as legacy, without run sidecars."""
import json
import os
from pathlib import Path
import sys

from .config import PROJECT_ROOT


def build_command(config, snapshot, lora_path):
    model, server = config['model'], config['server']
    command = [
        str(Path(sys.executable).with_name('vllm')), 'serve', str(snapshot), '--omni',
        '--served-model-name', model['id'], '--host', server['host'],
        '--port', str(server['port']), '--dtype', model['dtype'],
        '--tensor-parallel-size', '1', '--max-num-seqs', str(server['max_num_seqs']),
        '--diffusion-attention-backend', server['attention_backend'],
        '--lora-backend', model['lora']['backend'], '--lora-path', str(lora_path),
    ]
    for key, flag in [('step_execution', '--step-execution'),
                      ('vae_slicing', '--vae-use-slicing'), ('enforce_eager', '--enforce-eager')]:
        if server[key]:
            command.append(flag)
    return command


def serve(config, *, dry_run=False):
    if dry_run:
        print(json.dumps(build_command(config, '<pinned_base_snapshot>', '<pinned_lora_file>'), indent=2))
        return 0
    executable = Path(sys.executable).with_name('vllm')
    if not executable.is_file():
        raise RuntimeError('vllm is missing. From the project root, run '
                           'python -m pip install -r requirements-stage3-qwen.txt in the Qwen environment')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('The image server requires an NVIDIA CUDA GPU')
    os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / '.cache/huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    from huggingface_hub import hf_hub_download, snapshot_download
    model, lora = config['model'], config['model']['lora']
    snapshot = snapshot_download(
        model['id'], revision=model['revision'], max_workers=8,
        allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*',
                        'text_encoder/*', 'transformer/*', 'vae/*'])
    lora_path = hf_hub_download(lora['repo_id'], filename=lora['filename'], revision=lora['revision'])
    command = build_command(config, snapshot, lora_path)
    print('Starting image server:', ' '.join(command), flush=True)
    os.chdir(PROJECT_ROOT)
    os.execvpe(str(executable), command, dict(os.environ))
