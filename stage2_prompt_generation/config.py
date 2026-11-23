"""Stage2 Batch defaults and project-relative paths; no API calls on import."""

import math
from pathlib import Path
from stage1_components.files import component_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')


def project_path(value):
    return PROJECT_ROOT / value


def validate_sampling(reasoning_effort, temperature):
    if reasoning_effort not in ('none', 'low', 'medium', 'high', 'xhigh', 'max'):
        raise ValueError('Invalid reasoning_effort')
    if temperature is not None:
        if (type(temperature) not in (int, float) or not math.isfinite(temperature)
                or not 0 <= temperature <= 2):
            raise ValueError('temperature must be a finite number between 0 and 2, or null')
        if reasoning_effort != 'none':
            raise ValueError('temperature requires reasoning_effort=none; use temperature=null for reasoning')


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Stage2 config must be a mapping')
    required = {'input_dir', 'output_dir', 'model',
                'reasoning_effort', 'max_output_tokens', 'style_seed'}
    optional = {'temperature', 'component_id', 'output_id'}
    if not required <= set(config) or set(config) - required - optional:
        raise ValueError(f'Stage2 config requires {sorted(required)}; optional keys: {sorted(optional)}')
    config = {'temperature': None, **config}
    component_names(config.get('component_id'))
    component_names(config.get('output_id'))
    for key in ('input_dir', 'output_dir', 'model'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    validate_sampling(config['reasoning_effort'], config['temperature'])
    if type(config['max_output_tokens']) is not int or config['max_output_tokens'] < 1:
        raise ValueError('max_output_tokens must be a positive integer')
    if type(config['style_seed']) is not int:
        raise ValueError('style_seed must be an integer')
    return config


def load_config(path=DEFAULT_CONFIG):
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f'Duplicate configuration key: {key!r}')
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    try:
        config = yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid YAML configuration {path}: {exc}') from exc
    return validate_config(config)


def default_config():
    return load_config()


def input_path(config, split):
    names = dict(zip(('train', 'test'), component_names(config.get('component_id'))))
    return project_path(config['input_dir']) / names[split]


def output_path(config, split):
    names = component_names(config.get('output_id', config.get('component_id')))
    name = dict(zip(('train', 'test'), names))[split].replace('_components', '_prompts', 1)
    return project_path(config['output_dir']) / name
