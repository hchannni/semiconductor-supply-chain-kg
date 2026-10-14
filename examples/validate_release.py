"""Check file hashes, record shapes, foreign keys and counts without API calls.

The schema checker implements only the keywords used in this package's schemas;
it is not a general JSON Schema engine. Validation does not establish factual truth.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def shape(value, schema, where):
    actual = {str:'string', int:'integer', float:'number', bool:'boolean',
              list:'array', dict:'object', type(None):'null'}[type(value)]
    allowed = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
    require(actual in allowed or (actual == 'integer' and 'number' in allowed), where + ': wrong type')
    if 'enum' in schema:
        require(value in schema['enum'], where + ': value outside enum')
    if actual == 'array':
        for i, item in enumerate(value):
            shape(item, schema['items'], f'{where}[{i}]')
    if actual == 'object':
        require(set(schema['required']) <= value.keys(), where + ': missing required fields')
        require(set(value) <= schema['properties'].keys(), where + ': extra fields')
        for key, item in value.items():
            shape(item, schema['properties'][key], where + '.' + key)

def load(name):
    path = ROOT / 'data' / (name + '.jsonl')
    schema = json.loads((ROOT / 'schema' / (name + '.schema.json')).read_text(encoding='utf-8'))
    records = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        require(bool(line.strip()), f'{path.name}:{i}: blank line')
        record = json.loads(line)
        shape(record, schema, f'{path.name}:{i}')
        records.append(record)
    return records

def keyed(records, key):
    index = {r[key]:r for r in records}
    require(len(index) == len(records), key + ': duplicate ID')
    return index

def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        content = (ROOT / name).read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected['sha256'], name + ': checksum mismatch')
        require(len(content) == expected['bytes'], name + ': size mismatch')
    entities = keyed(load('canonical_entities'), 'entity_id')
    edges = keyed(load('fused_edges'), 'edge_id')
    claims = keyed(load('claim_provenance'), 'claim_id')
    documents = keyed(load('documents'), 'document_id')
    scores = keyed(load('edge_scores'), 'edge_id')
    flags = keyed(load('edge_review_flags'), 'edge_id')
    require(edges.keys() == scores.keys() == flags.keys(), 'Edge sidecar ID sets differ')
    used_claims = set()
    for eid, e in edges.items():
        require(e['subject'] in entities and e['object'] in entities, eid + ': missing endpoint')
        require(e['claim_ids'] and len(e['claim_ids']) == len(set(e['claim_ids'])), eid + ': invalid claim IDs')
        supporting = [claims[cid] for cid in e['claim_ids']]
        used_claims.update(e['claim_ids'])
        require(set(e['sources']) == {r['document_id'] for r in supporting}, eid + ': documents differ')
        require(e['source_count'] == len(set(e['sources'])), eid + ': source count differs')
        require(set(e['source_types']) == {r['source'] for r in supporting}, eid + ': source families differ')
        for claim in supporting:
            s, rel, o = claim['subject_entity_id'], claim['relation'], claim['object_entity_id']
            if rel == 'CUSTOMER_OF':
                s, rel, o = o, 'SUPPLIES_TO', s
            if rel == 'PARTNERS_WITH':
                s, o = sorted([s, o])
            require((s, rel, o) == (e['subject'], e['relation'], e['object']), eid + ': claim orientation mismatch')
        require(scores[eid]['confidence'] == e['confidence'], eid + ': score label mismatch')
    require(used_claims == claims.keys(), 'Unlinked active claims')
    for c in claims.values():
        require(c['document_id'] in documents, c['claim_id'] + ': missing document')
        require(documents[c['document_id']]['source'].upper() == c['source'], c['claim_id'] + ': source mismatch')
    stats = json.loads((ROOT / 'data/statistics.json').read_text(encoding='utf-8'))
    for key, val in [('entities',len(entities)),('relations',len(edges)),('claims',len(claims)),('documents',len(documents)),
                     ('both_source_relations',sum(len(set(e['source_types'])) == 2 for e in edges.values()))]:
        require(stats[key] == val, key + ': statistics mismatch')
    require(stats['by_relation'] == dict(Counter(e['relation'] for e in edges.values())), 'Relation counts differ')
    print(json.dumps({'status':'PASS','entities':len(entities),'relations':len(edges),
                      'claims':len(claims),'documents':len(documents),'manifest_files':len(manifest['files']),
                      'scope':'Structural/file integrity only; not factual verification'}, indent=2))

if __name__ == '__main__':
    main()
