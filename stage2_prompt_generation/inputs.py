"""Adapt Stage1 components to the scene used by the prompt assembler."""

from copy import deepcopy
from stage1_components.files import valid_prompt_id

def component_to_scene(component, split, catalog):
    """Validate Stage1 data and adapt entity references for prompt assembly."""
    from stage1_components.model import normalize_scene
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if not isinstance(component, dict):
        raise ValueError('Each component row must be an object')
    component = normalize_scene(component)
    identifier = component.get('prompt_id')
    if not valid_prompt_id(identifier):
        raise ValueError('prompt_id must be a canonical numeric string with at least four digits')
    catalog.validate(component, 4 if split == 'train' else 6)
    scene = {
        name: deepcopy(component[name])
        for name in ['entities', 'relations', 'background', 'rendering']
    }
    if 'is_query' in component and split != 'test':
        raise ValueError('is_query applies only to test candidates')
    for index, entity in enumerate(scene['entities']):
        entity['id'] = f'e{index + 1}'
    for relation in scene['relations']:
        for role in ['subject', 'object']:
            relation[role] = scene['entities'][relation[role]]['id']
    return scene
