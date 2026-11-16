"""Edit the instructions and prompt assembly here; no API calls or file writes."""

import hashlib
import json
from itertools import combinations
import random

from stage1_components.files import valid_prompt_id
from stage1_components.model import Catalog
from stage2_prompt_generation.inputs import component_to_scene
from stage2_prompt_generation.semantics import english_scene


STYLE_SEED = 0
STYLE_AXES = (
    (
        'Begin with an object or object group.',
        'Begin with a spatial relationship or interaction.',
        'Begin with the setting.',
        'Begin with the visual medium, viewpoint, or lighting.',
    ),
    (
        'Describe each relationship in a separate sentence.',
        'Join the relationship statements with "and" in one sentence.',
        'Put one relationship in a relative clause.',
        'Put one relationship in a phrase modifying an object noun.',
    ),
    (
        'Include rendering conditions in sentences describing objects or the setting.',
        'Keep rendering conditions in sentences separate from object and relationship descriptions.',
    ),
    (
        'Place descriptive attributes before object nouns.',
        'Describe attributes after introducing the object nouns.',
    ),
)

# A single relationship needs a second supplied fact for B2, or a main clause
# for B3/B4. Select the applicable instruction in Python, not in the LLM prompt.
SINGLE_RELATION_SYNTAX = (
    STYLE_AXES[1][0],
    'Join the relationship and setting statements with "and" in one sentence.',
    'Put the relationship in a relative clause and describe the setting in the main clause.',
    'Put the relationship in a phrase modifying an object noun and describe the setting in the main clause.',
)


SYSTEM_PROMPT = """Write an image-generation prompt in natural English from the supplied scene.
Return only the finished prompt as plain text, without JSON, headings, quotation
marks, explanations, or reasoning. Use at most 180 words.

Preserve every supplied object, exact count, attribute and its binding,
relationship, background, visual medium, viewpoint, and lighting.
Make all counts explicit, including single objects. Use the entity IDs only to
resolve references; never include IDs in the finished prompt.

Describe only supplied facts. Do not invent objects, attributes, visual details,
relationships, or absence constraints from the setting or common knowledge.

Use neutral present-tense descriptions. Do not add emotion, metaphor, opinions,
or a narrator. Do not force synonyms that change object categories, attribute
values, or relationship meanings. Use pronouns only when their referents are clear.

Follow the writing preferences below while preserving every scene fact.
Keep counts, attribute bindings, and relationship references unambiguous;
repeat object names or relax a style instruction only when needed for clarity.
Choose all unspecified wording and structure freely. Write connected, complete
sentences without fixed openings or stock closings. Do not mention the writing
preferences. Stop when all scene facts are covered.
"""


def style_profile_id(profile):
    """Readable identifier for a validated A(4)-B(4)-C(2)-D(2) profile."""
    if (not isinstance(profile, (tuple, list)) or len(profile) != len(STYLE_AXES)
            or any(type(value) is not int or not 0 <= value < len(choices)
                   for value, choices in zip(profile, STYLE_AXES))):
        raise ValueError('A style profile must contain four valid zero-based axis indices')
    return '-'.join(f'{axis}{value + 1}' for axis, value in zip('ABCD', profile))


def _style_rng(seed, split, key):
    value = json.dumps(['style-v1', seed, split, key], separators=(',', ':'))
    return random.Random(int.from_bytes(hashlib.sha256(value.encode()).digest(), 'big'))


def assign_style_profiles(components, split='test', *, seed=STYLE_SEED):
    """Balance profiles within groups without inspecting scene facts or query flags.

    Pass the complete input collection before selecting IDs or a limit. Input
    order does not affect allocation. Every aligned block of four balances all
    axes, and every 64 positions covers the full Cartesian product exactly once.
    Groups in a cluster share a schedule; unclustered groups have their own.
    Legacy rows without group IDs receive a deterministic individual assignment.
    """
    if split not in ('train', 'test') or type(seed) is not int:
        raise ValueError('Style allocation requires a train/test split and an integer seed')
    buckets = {}
    seen = set()
    group_clusters = {}
    for component in components:
        identifier = component.get('prompt_id')
        if not valid_prompt_id(identifier) or identifier in seen:
            raise ValueError(f'Invalid or duplicate style allocation prompt_id: {identifier!r}')
        seen.add(identifier)
        group, cluster = component.get('group_id'), component.get('cluster_id')
        for field, value in (('group_id', group), ('cluster_id', cluster)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'Invalid {field} for style allocation: {value!r}')
        if group is not None:
            if group in group_clusters and group_clusters[group] != cluster:
                raise ValueError(f'Group {group} belongs to multiple clusters')
            group_clusters[group] = cluster
        group_key = ('group', group) if group is not None else ('item', identifier)
        bucket = ('cluster', cluster) if cluster is not None else group_key
        buckets.setdefault(bucket, {}).setdefault(group_key, []).append(identifier)

    assignments = {}
    for bucket, groups in buckets.items():
        rng = _style_rng(seed, split, bucket)
        axes = [list(range(len(choices))) for choices in STYLE_AXES]
        for values in axes:
            rng.shuffle(values)
        schedule = []
        for index in range(64):
            # A and B each visit 0..3 per block; C and D each visit 0..1 twice.
            # The higher bits select the remaining combinations bijectively.
            indices = (index % 4, (index % 4 + index // 4) % 4,
                       (index // 2 + index // 16) % 2, (index + index // 32) % 2)
            schedule.append(tuple(values[i] for values, i in zip(axes, indices)))
        group_order = sorted(groups)
        rng.shuffle(group_order)
        cursor = 0
        for group_key in group_order:
            identifiers = sorted(groups[group_key])
            rng.shuffle(identifiers)
            for identifier in identifiers:
                assignments[identifier] = schedule[cursor % len(schedule)]
                cursor += 1
            # Also support partial/custom groups without misaligning later ones.
            cursor += -cursor % 4
    return assignments


def _entity_references(entities):
    """Use category names, adding minimal distinguishing attributes for repeats."""
    references = {}
    for entity in entities:
        category = entity['category']
        others = [other for other in entities
                  if other['category'] == category and other['id'] != entity['id']]
        if not others:
            references[entity['id']] = category
            continue
        attributes = entity['attributes']
        # Prefer color for equally short descriptions; keep selection deterministic.
        fields = sorted(attributes, key=lambda field: (field != 'color', field))
        selected = next((keys for size in range(1, len(fields) + 1)
                         for keys in combinations(fields, size)
                         if all(any(key in other['attributes']
                                    and attributes[key] != other['attributes'][key]
                                    for key in keys) for other in others)), None)
        if selected is None:
            raise ValueError(f'Cannot distinguish same-category entities: {category}')
        # Directional phrases read naturally after the noun (e.g. "cat facing left").
        before = [attributes[key] for key in selected if key not in ('facing', 'orientation')]
        after = [attributes[key] for key in selected if key in ('facing', 'orientation')]
        references[entity['id']] = ' '.join([*before, category, *after])
    return references


def build_prompt(component, split='test', *, catalog=None, style_profile=None,
                 style_seed=STYLE_SEED):
    """Validate one Stage1 component and assemble Responses API prompt fields.

    Reuse Stage2's reference conversion and English labels, then append the
    selected writing preferences. Adapt B for one relationship and omit D when
    no descriptive attributes are supplied. For collection-stable assignment, supply the
    profile returned by assign_style_profiles for the complete input collection.
    The standalone fallback assigns using this component alone. The original
    component is preserved for output storage.
    """
    catalog = Catalog() if catalog is None else catalog
    scene = component_to_scene(component, split, catalog)
    vocab = {**catalog.data, 'entities': catalog.entities}
    payload = {'scene': english_scene(scene, vocab)}
    references = _entity_references(payload['scene']['entities'])
    payload['scene']['relationships'] = [
        f"{references[relationship['subject']]} {relationship['meaning']} "
        f"{references[relationship['object']]}"
        for relationship in payload['scene']['relationships']
    ]
    if style_profile is None:
        style_profile = assign_style_profiles([component], split, seed=style_seed)[component['prompt_id']]
    style_profile_id(style_profile)  # Validate before indexing the instruction bank.
    selected = [choices[index] for choices, index in zip(STYLE_AXES, style_profile)]
    if len(scene['relations']) == 1:
        selected[1] = SINGLE_RELATION_SYNTAX[style_profile[1]]
    if not any(entity['attributes'] for entity in scene['entities']):
        selected.pop()  # D cannot change attribute placement when there are none.
    preferences = '\n'.join(f'- {instruction}' for instruction in selected)
    return {
        'instructions': SYSTEM_PROMPT + '\nWriting preferences:\n' + preferences,
        'input': json.dumps(payload, ensure_ascii=False, indent=2),
    }
