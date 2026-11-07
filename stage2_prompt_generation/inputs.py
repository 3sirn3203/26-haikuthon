"""Read Stage1 outputs and adapt them to scene dictionaries for the LLM."""

from copy import deepcopy
import hashlib
import json
import random
import re

from .config import project_path
from .semantics import PRESENTATIONS, PRESENTATION_ORDERS, PRESENTATION_STRUCTURES


INPUT_NAMES = {'train': 'train_components.jsonl', 'test': 'test_components.jsonl'}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def component_to_case(component, split, catalog, seed):
    """Keep prompt_id intact; change only internal IDs/references and presentation."""
    if split not in INPUT_NAMES:
        raise ValueError('split must be train or test')
    if not isinstance(component, dict):
        raise ValueError('Each component row must be an object')
    identifier = component.get('prompt_id')
    if not isinstance(identifier, str) or re.fullmatch(r'[0-9]{4}', identifier) is None:
        raise ValueError('prompt_id must be a four-digit string')
    catalog.validate(component, 4 if split == 'train' else 6)
    scene = {
        name: deepcopy(component[name])
        for name in ['entities', 'relations', 'background', 'rendering']
    }
    for index, entity in enumerate(scene['entities']):
        entity['id'] = f'e{index + 1}'
    for relation in scene['relations']:
        for role in ['subject', 'object']:
            relation[role] = scene['entities'][relation[role]]['id']
    # This stable per-item default is used for independent scenes. Family members
    # receive a balanced, non-repeating allocation after the full split is read.
    digest = hashlib.sha256(f'{seed}:{split}:{identifier}'.encode()).digest()
    scene['presentation_id'] = int.from_bytes(digest[:8], 'big') % len(PRESENTATIONS)
    return identifier, scene


def assign_family_presentations(components, cases, catalog, seed, split):
    """Distribute the 5 x 4 guides without changing facts, IDs, or row order."""
    from stage1_components.validation import partition

    by_id = dict(cases)
    _, families = partition(components, catalog)
    for family in families:
        identifiers = sorted(row['prompt_id'] for row in family)
        if len(identifiers) > len(PRESENTATIONS):
            raise ValueError('A family has more members than available presentation guides')
        digest = hashlib.sha256(f'{seed}:{split}:family:{",".join(identifiers)}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest, 'big'))
        orders = list(range(len(PRESENTATION_ORDERS)))
        structures = list(range(len(PRESENTATION_STRUCTURES)))
        rng.shuffle(orders)
        rng.shuffle(structures)
        rng.shuffle(identifiers)
        # With eight members, every order appears once or twice, every structure
        # appears twice, and all eight combined guides are different.
        schedule = [
            order * len(structures) + structures[(index + cycle) % len(structures)]
            for cycle in range(len(structures)) for index, order in enumerate(orders)
        ]
        for identifier, presentation in zip(identifiers, schedule):
            by_id[identifier]['presentation_id'] = presentation


def load_component_splits(config, splits=('train', 'test')):
    """Validate every requested input before starting a server or opening outputs.

    Return prepared cases and the effective vocabulary.
    No stage1 sampling or dataset generation is performed here.
    """
    from stage1_components.model import Catalog

    catalog = Catalog(project_path(config['components_path']))
    vocab = dict(catalog.data)
    vocab['entities'] = catalog.entities
    prepared = {}
    for split in splits:
        if split not in INPUT_NAMES:
            raise ValueError('split must be train or test')
        path = project_path(config['input_dir']) / INPUT_NAMES[split]
        cases = []
        components = []
        seen = set()
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    if not line.strip():
                        raise ValueError('Blank JSONL line')
                    component = json.loads(line, object_pairs_hook=_unique_pairs)
                    identifier, scene = component_to_case(component, split, catalog, config['seed'])
                    if identifier in seen:
                        raise ValueError(f'Duplicate prompt_id: {identifier}')
                    seen.add(identifier)
                    cases.append((identifier, scene))
                    components.append(component)
                except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
        if not cases:
            raise ValueError(f'Empty component input: {path}')
        assign_family_presentations(components, cases, catalog, config['seed'], split)
        prepared[split] = cases
    return prepared, vocab
