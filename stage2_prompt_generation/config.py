"""Configuration shared by component loading, the server and generation."""

import math
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')


def generation_settings(settings):
    # Older Stage2 configurations left these fields implicit.
    settings = {'temperature': 0.9, **settings}
    temperature = settings['temperature']
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('generation.temperature must be a finite nonnegative number')
    return settings


def project_path(value):
    """Resolve source paths relative to the project, not the current directory."""
    return PROJECT_ROOT / value


def server_address(config):
    parsed = urlsplit(config['server_url'])
    if (parsed.scheme != 'http' or not parsed.hostname or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('server_url must be an HTTP origin, for example http://127.0.0.1:18001')
    port = 80 if parsed.port is None else parsed.port
    if not 1 <= port <= 65535:
        raise ValueError('Invalid server port')
    return parsed.hostname, port


def validate_config(config):
    try:
        config = {**config, 'generation': generation_settings(config['generation'])}
        for section, keys in [
            (config, ['input_dir', 'components_path', 'server_url']),
            (config['model'], ['id', 'revision', 'dtype']),
        ]:
            if any(not isinstance(section[key], str) or not section[key].strip() for key in keys):
                raise ValueError('Paths, server URL and model fields must be nonempty strings')
        server_address(config)
        if type(config['seed']) is not int or config['seed'] < 0:
            raise ValueError('seed must be a nonnegative integer')
        if type(config['model']['enable_thinking']) is not bool:
            raise ValueError('model.enable_thinking must be a boolean')
        for phase in ['generation', 'critic']:
            settings = config[phase]
            thinking = settings.get('enable_thinking', config['model']['enable_thinking'])
            if type(thinking) is not bool:
                raise ValueError(f'{phase}.enable_thinking must be a boolean')
            effort = settings.get('reasoning_effort')
            if effort is not None and (not thinking or effort not in ['low', 'medium', 'high', 'xhigh']):
                raise ValueError(f'{phase}.reasoning_effort requires thinking and a supported effort')
            for key, lower, upper in [('min_p', 0, 1), ('presence_penalty', -2, 2)]:
                value = settings.get(key, 0.0)
                if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                    raise ValueError(f'{phase}.{key} must be a finite number in [{lower}, {upper}]')
        for section, keys in [
            (config['server'], ['tensor_parallel_size', 'max_model_len', 'max_num_seqs', 'max_num_batched_tokens']),
            (config['generation'], ['batch_size', 'max_new_tokens', 'top_k']),
            (config['critic'], ['batch_size', 'max_new_tokens']),
        ]:
            if any(type(section[key]) is not int or section[key] < 1 for key in keys):
                raise ValueError('Batch sizes, token limits and server capacities must be positive integers')
        for value in [
            config['server']['startup_timeout_seconds'],
            config['server']['shutdown_timeout_seconds'],
            config['generation']['repetition_penalty'],
        ]:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Timeouts and repetition_penalty must be positive finite numbers')
        for value in [config['server']['gpu_memory_utilization'], config['generation']['top_p']]:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('gpu_memory_utilization and top_p must be in (0, 1]')
        if config['critic']['do_sample'] is not False:
            raise ValueError('critic.do_sample must be false')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'Invalid or missing configuration field: {exc}') from exc
    return config


def load_config(path=DEFAULT_CONFIG):
    # Delay the external dependency until an explicit configuration load. Merely
    # importing stage2 never starts a server or installs/loads a model.
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
