"""Shared vocabulary, semantic normalization and scene validation."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path
import json

import yaml

ROOT = Path(__file__).resolve().parent


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for kn, vn in node.value:
        key = loader.construct_object(kn, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate YAML key: {key}')
        result[key] = loader.construct_object(vn, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path):
    return yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family(predicate):
    return {'left_of': 'horizontal', 'right_of': 'horizontal', 'above': 'vertical', 'below': 'vertical'}.get(predicate, predicate)


def edge(a, p, b):
    if p in ('right_of', 'below'):
        a, b, p = b, a, {'right_of': 'left_of', 'below': 'above'}[p]
    if p == 'next_to' and b < a:
        a, b = b, a
    return a, p, b


def relation(a, p, b):
    r = {'subject': a, 'predicate': p, 'object': b, 'scope': 'all_pairs'}
    if p in ('left_of', 'right_of', 'above', 'below'):
        r['reference_frame'] = 'viewer'
    return r


def normalized_edges(scene):
    edges = {edge(r['subject'], r['predicate'], r['object']) for r in scene['relations']}
    for p in ('left_of', 'above', 'on', 'inside'):
        links = {(a, b) for a, q, b in edges if q == p}
        while True:
            expanded = links | {(a, d) for a, b in links for c, d in links if b == c}
            if expanded == links:
                break
            links = expanded
        require(not any(a == b for a, b in links), f'Cyclic {p} relation')
        if p in ('left_of', 'above'):
            edges.update((a, p, b) for a, b in links)
    return edges


def total(scene):
    return sum(e['count'] for e in scene['entities'])


def inventory(scene):
    return frozenset(e['category'] for e in scene['entities'])


def label(e, axis=None):
    return (e['category'], e['count'] if axis in (None, 'count') else 0,
            tuple(sorted(e['attributes'].items())) if axis in (None, 'binding') else ())


def canonical(scene, axis=None):
    """Canonical labeled graph, invariant to entity IDs, order and edge wording.

    Projection labels must be canonicalized again *after* masking: otherwise
    attribute-dependent node ordering would leak into the relation projection.
    """
    labels = [label(e, axis) for e in scene['entities']]
    if axis in ('category', 'count', 'binding'):
        return tuple(sorted(labels))
    groups = defaultdict(list)
    for i, v in enumerate(labels):
        groups[v].append(i)
    ordered = sorted(groups)
    candidates = product(*(permutations(groups[v]) for v in ordered))
    edges = normalized_edges(scene)
    best = None
    for pieces in candidates:
        order = [i for piece in pieces for i in piece]
        renaming = {old: new for new, old in enumerate(order)}
        result = (tuple(labels[i] for i in order), tuple(sorted(edge(renaming[a], p, renaming[b]) for a, p, b in edges)))
        if best is None or result < best:
            best = result
    return best


@lru_cache(maxsize=1)
def default_catalog():
    return Catalog()


def context_key(scene, catalog=None):
    catalog = catalog or default_catalog()
    # A foil may replace one object by another from its domain. This key is
    # reconstructible from visible components and survives entity reordering.
    return (tuple(sorted(catalog.entities[e['category']]['domain'] for e in scene['entities'])), total(scene),
            json.dumps(scene['background'], sort_keys=True), json.dumps(scene['rendering'], sort_keys=True))


def changed_axes(first, second):
    """Find a least-edit alignment without using array positions as semantics."""
    aa, bb = first['entities'], second['entities']
    if len(aa) != len(bb):
        return {'category', 'relation'}
    ea = normalized_edges(first)
    eb = normalized_edges(second)
    best = None
    for order in permutations(range(len(bb))):
        axes, cost = set(), 0
        for i, j in enumerate(order):
            for key, axis in [('category', 'category'), ('count', 'count'), ('attributes', 'binding')]:
                if aa[i][key] != bb[j][key]:
                    axes.add(axis)
                    if key == 'attributes':
                        cost += len(set(aa[i][key].items()) ^ set(bb[j][key].items()))
                    else:
                        cost += 2
        renaming = {old: new for new, old in enumerate(order)}
        mapped = {edge(renaming[a], p, renaming[b]) for a, p, b in eb}
        if ea != mapped:
            axes.add('relation')
            cost += len(ea ^ mapped)
        candidate = (cost, len(axes), tuple(sorted(axes)))
        if best is None or candidate < best:
            best = candidate
    return set(best[2])


class Catalog:
    def __init__(self, path=ROOT / 'components.yaml'):
        self.data = read_yaml(path)
        self.entities = {k: e for k, e in self.data['entities'].items() if e.get('review_tier') != 'deferred'}
        self.hierarchical = {term for c, e in self.entities.items() if e.get('broader_category')
                             for term in self.ancestors(c)}
        self.rules = self.data['constraints']
        self.active = self.data['active']
        self._attributes = {}
        for c, e in self.entities.items():
            values = dict(self.data['attribute_profiles'][e['attribute_profile']]['attributes'])
            values.update(e.get('attribute_value_overrides', {}))
            self._attributes[c] = {a: tuple(v for v in vv if v not in self.active['disabled_values'].get(a, []))
                                   for a, vv in values.items() if a in self.active['attributes']}
        self.aliases = {alias.lower(): c for c, e in self.entities.items()
                        for alias in [c, e['label']['en']] + e.get('aliases', {}).get('en', [])}

    def attributes(self, category, style=None):
        values = self._attributes[category]
        return {a: vv for a, vv in values.items() if vv and
                not (a == 'color' and style in self.data['style_groups']['monochrome']) and
                not (a == 'material' and style is not None and style not in self.data['style_groups']['material_readable_core'])}

    @lru_cache(maxsize=None)
    def ancestors(self, category):
        result = [category]
        while self.entities[category].get('broader_category'):
            category = self.entities[category]['broader_category']
            require(category not in result, 'Category hierarchy cycle')
            result.append(category)
        return tuple(result)

    @lru_cache(maxsize=None)
    def contexts(self, category):
        e = self.entities[category]
        if e.get('background_candidates'):
            result = set(e['background_candidates'])
        elif e['kind'] == 'animal':
            result = {place for place, cats in self.rules['animal_backgrounds'].items() if category in cats}
        elif e['kind'] == 'human':
            result = set(self.data['backgrounds'])
        elif 'vehicle' in e['capabilities']:
            result = set(self.rules['vehicle_backgrounds'])
        else:
            result = set(self.data['context_profiles'][e['domain']]['background_candidates'])
        for extension in self.rules['additional_contexts']:
            if category in extension['categories']:
                result.update(extension['locations'])
        if e['kind'] == 'animal':
            result &= {place for place, cats in self.rules['animal_backgrounds'].items() if category in cats}
        return frozenset(result)

    def compatible(self, categories):
        unique = set(categories)
        return not any(set(self.ancestors(c)[1:]) & unique for c in unique) and bool(set.intersection(*(set(self.contexts(c)) for c in unique)))

    def accepts_role(self, category, selector):
        e = self.entities[category]
        return (('any_entity' in selector and selector['any_entity']) or
                ('categories' in selector and category in selector['categories']) or
                ('capabilities_any' in selector and bool(set(e['capabilities']) & set(selector['capabilities_any']))))

    def legal_relation(self, a, predicate, b):
        if predicate not in self.active['relations']:
            return False
        r = self.data['relations'][predicate]
        if not all(self.accepts_role(c, r['role_selectors'][role]) for c, role in [(a, 'subject'), (b, 'object')]):
            return False
        if predicate == 'inside':
            contents = self.rules['mug_contents'] if b == 'mug' else self.rules['small_contents']
            return b in self.rules['inside_targets'] and a in contents
        if predicate == 'on':
            caps = self.entities[a]['capabilities']
            soft = self.rules['unsuitable_soft_supports']
            if b in soft['supports'] and self.entities[a]['domain'] in soft['subjects_in_domains']:
                return False
            if 'human' in caps: return b in self.rules['human_supports']
            if 'animal' in caps: return a in self.rules['animal_support_subjects'] and b in self.rules['animal_supports']
            if 'large_subject' in caps: return False
            if b in self.rules['small_supports']: return a in self.rules['small_support_subjects']
            if b == 'skateboard': return False
            return bool(set(caps) & {'portable', 'plant'})
        return True

    def validate(self, scene, maximum=6):
        require(set(scene) <= {'prompt_id', 'entities', 'relations', 'background', 'rendering'}, 'Unexpected scene metadata')
        require({'entities', 'relations', 'background', 'rendering'} <= scene.keys(), 'Missing component fields')
        entities = scene['entities']
        require(isinstance(entities, list) and 2 <= len(entities) <= 4, 'Expected 2–4 entity groups')
        require(2 <= total(scene) <= maximum, 'Scene total count out of range')
        cats = []
        render = scene['rendering']
        require(set(render) == {'style', 'viewpoint', 'framing', 'lighting'}, 'Rendering fields')
        require(render['style'] in self.data['styles'], 'Unknown style')
        require(render['viewpoint'] in self.active['viewpoints'], 'Inactive viewpoint')
        require(render['framing'] == 'all_target_objects_fully_visible', 'Framing contract')
        require(render['lighting'] in self.active['lighting'], 'Inactive lighting')
        for e in entities:
            require(set(e) == {'category', 'count', 'attributes'}, 'Entity fields')
            c = e['category']; require(c in self.entities, f'Inactive category {c}')
            require(type(e['count']) is int and e['count'] > 0, 'Count must be a positive integer')
            if 'large_subject' in self.entities[c]['capabilities']:
                require(e['count'] <= self.rules['large_subject_max_count'], 'Large subject count')
            allowed = self.attributes(c, render['style'])
            require(isinstance(e['attributes'], dict) and len(e['attributes']) <= self.rules['max_attributes_per_entity'], 'Attribute count')
            for a, v in e['attributes'].items():
                require(v in allowed.get(a, ()), f'Illegal binding {c}/{a}/{v}')
            cats.append(c)
        require(self.compatible(cats), 'Incompatible categories/context')
        for i, a in enumerate(entities):
            for b in entities[i+1:]:
                if a['category'] == b['category']:
                    require(any(k in b['attributes'] and v != b['attributes'][k] for k, v in a['attributes'].items()), 'Ambiguous same-category groups')
        require(set(scene['background']) == {'location'}, 'Background fields')
        location = scene['background']['location']
        require(all(location in self.contexts(c) for c in cats), 'Background incompatibility')
        require(1 <= len(scene['relations']) <= 4, 'Relation count')
        for r in scene['relations']:
            require(set(r) <= {'subject', 'predicate', 'object', 'scope', 'reference_frame'}, 'Relation fields')
            a, p, b = r['subject'], r['predicate'], r['object']
            require(type(a) is int and type(b) is int and 0 <= a < len(entities) and 0 <= b < len(entities) and a != b, 'Invalid relation reference')
            require(r['scope'] == 'all_pairs', 'Relation scope must be all_pairs')
            require(self.legal_relation(cats[a], p, cats[b]), 'Illegal relation roles')
            if p in ('left_of', 'right_of', 'above', 'below'):
                require(r.get('reference_frame') == 'viewer', 'Spatial reference frame')
            else:
                require('reference_frame' not in r, 'Unexpected reference frame')
            if p in ('holding', 'wearing', 'riding'):
                require(entities[a]['count'] == entities[b]['count'] == 1, 'Action arguments must be singleton')
            if p == 'riding':
                require(entities[a]['attributes'].get('pose') == 'sitting', 'Rider must be sitting')
                require(location in self.rules['riding_backgrounds'], 'Unsuitable riding background')
                if self.entities[cats[b]]['kind'] == 'animal':
                    require(entities[b]['attributes'].get('pose') == self.rules['ridden_animal_pose'], 'Ridden animal must be standing')
            if p == 'inside':
                require(entities[b]['count'] == 1 and render['viewpoint'] == 'overhead', 'Inside visibility')
                if 'openable' in self.entities[cats[b]]['capabilities']:
                    require(entities[b]['attributes'].get('state') == 'open', 'Closed container')
            if p == 'on':
                require(entities[b]['count'] == 1, 'A support must be singleton')
                if 'human' in self.entities[cats[a]]['capabilities']:
                    expected = 'standing' if cats[b] == 'skateboard' else 'lying' if cats[b] == 'bed' else 'sitting'
                    require(entities[a]['count'] == 1 and entities[a]['attributes'].get('pose') == expected, 'Human support posture/count')
        require(len({edge(r['subject'], r['predicate'], r['object']) for r in scene['relations']}) == len(scene['relations']), 'Duplicate relation')
        by_pair = defaultdict(set)
        for r in scene['relations']:
            by_pair[frozenset((r['subject'], r['object']))].add(r['predicate'])
        for predicates in by_pair.values():
            require(not (predicates & {'on', 'inside', 'holding', 'wearing', 'riding'} and
                         predicates & {'left_of', 'right_of', 'above', 'below', 'next_to'}), 'Overlapping action/support and external spatial relation')
        normalized_edges(scene)

    def atoms(self, scene):
        bindings = {(e['category'], a, v) for e in scene['entities'] for a, v in e['attributes'].items()}
        relations = {edge(scene['entities'][a]['category'], p, scene['entities'][b]['category'])
                     for a, p, b in normalized_edges(scene)}
        return bindings, relations

    def expanded_atoms(self, scene):
        bindings, relations = self.atoms(scene)
        return ({(parent, a, v) for c, a, v in bindings for parent in self.ancestors(c)},
                {edge(x, p, y) for a, p, b in relations for x in self.ancestors(a) for y in self.ancestors(b)})
