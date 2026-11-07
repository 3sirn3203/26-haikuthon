"""Scene presentation, English conversion, and structured-response parsing."""


import json
import re


PRESENTATION_ORDERS = [
    'Begin with the setting, then describe the objects and their relationships. '
    'Introduce the rendering conditions later.',
    'Begin with a named object or object group and its specified attributes. '
    'Describe its relationships next, then the setting and rendering conditions.',
    'Begin with the main spatial relationship or interaction. '
    'Introduce any remaining objects, then the setting and rendering conditions.',
    'Begin with the specified visual medium, then describe the scene content. '
    'Introduce viewpoint, lighting, and framing later.',
    'Begin with the specified viewing perspective and the scene seen from it. '
    'Describe the objects and relationships, then the remaining rendering conditions.',
]
PRESENTATION_STRUCTURES = [
    'Use direct, connected sentences with clear subjects and verbs. '
    'Combine closely related facts naturally rather than listing one fact per sentence.',
    'Use relative clauses where useful to connect facts about the same object. '
    'Keep each clause\'s referent unambiguous; repeat the object name when needed.',
    'Connect independent facts with parallel clauses whose subjects are explicit. '
    'Do not merge unrelated objects into the same relationship or attribute scope.',
    'Vary sentence lengths: use a concise opening and fuller descriptive sentences '
    'where related facts fit together. Let the content determine the sentence count.',
]
PRESENTATIONS = [
    f'{order} {structure} Write complete sentences. '
    'Integrate framing and lighting naturally without a stock closing sentence. '
    'Preserve all supplied facts without adding visual details or commentary.'
    for order in PRESENTATION_ORDERS for structure in PRESENTATION_STRUCTURES
]


def english_scene(scene,vocab):
    entities=[]
    for e in scene['entities']:
        entities.append({'id':e['id'],'category':vocab['entities'][e['category']]['label']['en'],'count':e['count'],
           'attributes':{attr:vocab['attributes'][attr]['values'][value]['en'] for attr,value in e['attributes'].items()}})
    relationships=[{**r,'meaning':vocab['relations'][r['predicate']]['label']['en']} for r in scene['relations']]
    background={k:(vocab['backgrounds'][v]['label']['en'] if k=='location' else vocab[k][v]['en']) for k,v in scene['background'].items()}
    rendering={k:('every named object fully visible within the frame' if k=='framing' else vocab[{'style':'styles','viewpoint':'viewpoints','lighting':'lighting'}[k]][v]['en']) for k,v in scene['rendering'].items()}
    return {'entities':entities,'relationships':relationships,'background':background,'rendering':rendering}


def parse_json_output(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Output must be a JSON object')
    return value
