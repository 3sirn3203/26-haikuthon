"""Convert component values to catalog English labels."""


def english_scene(scene,vocab):
    entities=[]
    for e in scene['entities']:
        entities.append({'id':e['id'],'category':vocab['entities'][e['category']]['label']['en'],'count':e['count'],
           'attributes':{attr:vocab['attributes'][attr]['values'][value]['en'] for attr,value in e['attributes'].items()}})
    relationships=[{**{k:r[k] for k in ('subject','predicate','object')},
                    'meaning':vocab['relations'][r['predicate']]['label']['en']} for r in scene['relations']]
    background={k:(vocab['backgrounds'][v]['label']['en'] if k=='location' else vocab[k][v]['en']) for k,v in scene['background'].items()}
    labels={'style':'styles','viewpoint':'viewpoints','lighting':'lighting'}
    rendering={k:vocab[labels[k]][v]['en'] for k,v in scene['rendering'].items() if k in labels}
    return {'entities':entities,'relationships':relationships,'background':background,'rendering':rendering}
