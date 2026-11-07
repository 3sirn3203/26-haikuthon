"""Stage2 generation APIs; importing them never runs inference or writes output."""

from .client import VLLMClient
from .config import default_config, load_config
from .inputs import component_to_case, load_component_splits
from .pipeline import generate_prompt, generate_split, generate_train_test
from .server import managed_server

__all__ = [
    'VLLMClient',
    'default_config',
    'load_config',
    'component_to_case',
    'load_component_splits',
    'generate_prompt',
    'generate_split',
    'generate_train_test',
    'managed_server',
]
