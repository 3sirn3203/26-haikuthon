"""Settings shared by the server and client. Relative I/O paths use project root."""
import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate key: {key!r}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def validate_config(config):
    try:
        model, server = config['model'], config['server']
        generation, client = config['generation'], config['client']
        for section, keys in [
            (config, ('input_dir', 'output_dir')),
            (model, ('id', 'revision', 'dtype')),
            (model['lora'], ('repo_id', 'revision', 'filename', 'backend')),
            (server, ('host', 'attention_backend')),
        ]:
            if any(not isinstance(section[key], str) or not section[key].strip() for key in keys):
                raise ValueError('Paths and model/server identifiers must be nonempty strings')
        for name, value in {
            'width': generation['width'], 'height': generation['height'],
            'num_inference_steps': generation['num_inference_steps'],
            'port': server['port'], 'max_num_seqs': server['max_num_seqs'],
            'concurrency': client['concurrency'], 'max_attempts': client['max_attempts'],
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if server['port'] > 65535 or client['concurrency'] > server['max_num_seqs']:
            raise ValueError('Invalid port or concurrency exceeds server.max_num_seqs')
        if any(generation[key] % 16 for key in ('width', 'height')):
            raise ValueError('Image dimensions must be multiples of 16')
        for key in ('step_execution', 'vae_slicing', 'enforce_eager'):
            if type(server[key]) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(generation['base_seed']) is not int or not 0 <= generation['base_seed'] < 2**32:
            raise ValueError('base_seed must be a uint32 integer')
        for value in (generation['true_cfg_scale'], client['timeout_seconds']):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('CFG and timeout must be positive finite numbers')
        if not isinstance(generation['negative_prompt'], str):
            raise ValueError('negative_prompt must be a string')
        if model['lora']['backend'] != 'distill' or model['lora']['scale'] != 1.0:
            raise ValueError('The Lightning LoRA requires distill backend and scale 1.0')
        prompt_files = config.get('prompt_files', {'train': 'train_prompts.jsonl', 'test': 'test_prompts.jsonl'})
        if not isinstance(prompt_files, dict) or set(prompt_files) != {'train', 'test'} or any(
                not isinstance(value, str) or not value.strip() for value in prompt_files.values()):
            raise ValueError('prompt_files must contain nonempty train and test filenames')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Invalid or missing configuration field: {exc}') from exc
    return config


def load_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader))


def project_path(value):
    return PROJECT_ROOT / value
