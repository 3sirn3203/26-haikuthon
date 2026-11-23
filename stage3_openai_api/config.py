"""Validated provisional settings. Relative paths use the repository root."""
import math
from pathlib import Path

import yaml

from stage3_image_generation.config import UniqueLoader, project_path


DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')
MODELS = {
    'gpt-image-2.5-flare', 'gpt-image-2.5-sunburst',
    'gpt-image-2.5-flare-2026-09-08', 'gpt-image-2.5-sunburst-2026-09-08',
}


def validate_config(config):
    try:
        required = {'input_dir', 'prompt_files', 'output_dir', 'default_split', 'model',
                    'generation', 'save_size', 'batch', 'client'}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(f'Expected config keys: {sorted(required)}')
        for key in ('input_dir', 'output_dir'):
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f'{key} must be a nonempty path')
        if config['default_split'] not in ('train', 'test'):
            raise ValueError('default_split must be train or test')
        if config['model'] not in MODELS:
            raise ValueError('model must be a GPT Image 2.5 Flare/Sunburst model')
        files = config['prompt_files']
        if not isinstance(files, dict) or set(files) != {'train', 'test'} or any(
                not isinstance(v, str) or not v.strip() for v in files.values()):
            raise ValueError('prompt_files must specify train and test paths')
        g, b, c = config['generation'], config['batch'], config['client']
        if set(g) != {'quality', 'width', 'height', 'background', 'output_format',
                      'output_compression', 'moderation'}:
            raise ValueError('Unexpected or missing generation setting')
        if g['quality'] not in ('low', 'medium', 'high', 'xhigh', 'max', 'auto'):
            raise ValueError('Invalid quality')
        w, h = g['width'], g['height']
        if any(type(v) is not int or v <= 0 or v % 16 for v in (w, h)):
            raise ValueError('API width/height must be positive multiples of 16')
        if not 655_360 <= w*h <= 8_294_400 or max(w, h) > 3840 or max(w, h) > 3*min(w, h):
            raise ValueError('API size exceeds pixel, edge, or aspect ratio limits (768x768 is too small)')
        if g['background'] not in ('opaque', 'transparent', 'auto'):
            raise ValueError('Invalid background')
        if g['output_format'] not in ('png', 'jpeg', 'webp'):
            raise ValueError('output_format must be png, jpeg or webp')
        if g['moderation'] not in ('auto', 'low'):
            raise ValueError('moderation must be auto or low')
        compression = g['output_compression']
        if compression is not None:
            if type(compression) is not int or not 0 <= compression <= 100:
                raise ValueError('output_compression must be null or an integer from 0 to 100')
            if g['output_format'] == 'png':
                raise ValueError('PNG does not support output_compression; use null')
        if g['background'] == 'transparent' and g['output_format'] == 'jpeg':
            raise ValueError('Transparent backgrounds require PNG or WebP')
        size = config['save_size']
        if size is not None:
            if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
                raise ValueError('save_size must be null or [width, height]')
            if size[0]*h != size[1]*w or size[0] > w or size[1] > h:
                raise ValueError('save_size must preserve aspect ratio and cannot upscale')
        if set(b) != {'max_requests', 'completion_window', 'poll_interval_seconds', 'delete_input_after_collect'}:
            raise ValueError('Unexpected or missing batch setting')
        if type(b['max_requests']) is not int or not 1 <= b['max_requests'] <= 50_000:
            raise ValueError('batch.max_requests must be between 1 and 50000')
        if b['completion_window'] != '24h':
            raise ValueError('Only the 24h Batch completion window is supported')
        if type(b['delete_input_after_collect']) is not bool:
            raise ValueError('delete_input_after_collect must be a boolean')
        poll = b['poll_interval_seconds']
        if type(poll) not in (int, float) or not math.isfinite(poll) or not 1 <= poll <= 60:
            raise ValueError('poll_interval_seconds must be between 1 and 60')
        if set(c) != {'timeout_seconds'}:
            raise ValueError('client only accepts timeout_seconds')
        timeout = c['timeout_seconds']
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout_seconds must be a positive finite number')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Invalid configuration: {exc}') from exc
    return config


def load_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader))


def render_settings(config):
    """Compact settings retained on the remote Batch, never in a local file."""
    return {key: config[key] for key in ('model', 'generation', 'save_size')}
