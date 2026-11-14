"""Component filenames shared by Stage1 and downstream input selection."""
from datetime import datetime, timedelta, timezone
import re


KST = timezone(timedelta(hours=9), name='KST')
ID_PATTERN = r'[\w.-]+'
OUTPUT_PATTERN = re.compile(rf'(?:train|test)_components(?:_{ID_PATTERN})?\.jsonl')


def valid_prompt_id(value):
    """Four-digit legacy IDs plus canonical decimal IDs from 10000 upward."""
    return isinstance(value, str) and re.fullmatch(r'(?:[0-9]{4}|[1-9][0-9]{4,})', value) is not None


def default_output_id():
    """Capture the invocation time in KST, independently of the host timezone."""
    return datetime.now(KST).strftime('%Y%m%d%H%M')


def component_names(identifier=None):
    """None selects legacy filenames for older downstream configurations."""
    if identifier is not None and (not isinstance(identifier, str) or re.fullmatch(ID_PATTERN, identifier) is None):
        raise ValueError('Component file id must be a nonempty string containing only letters, numbers, _, -, or .')
    suffix = '' if identifier is None else f'_{identifier}'
    return tuple(f'{split}_components{suffix}.jsonl' for split in ('train', 'test'))


def check_output_directory(directory):
    """Allow multiple component runs together while rejecting unrelated entries."""
    unexpected = sorted(path.name for path in directory.iterdir()
                        if not path.is_file() or OUTPUT_PATTERN.fullmatch(path.name) is None)
    if unexpected:
        raise ValueError(f'Output directory contains unexpected files: {unexpected}')
