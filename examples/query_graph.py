"""Offline read-only examples for the released KG. Python standard library only."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]

def rows(name):
    with (ROOT / 'data' / name).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('stats')
    company = sub.add_parser('company', help='Search labels, aliases or an exact entity ID')
    company.add_argument('query')
    company.add_argument('--relation')
    company.add_argument('--limit', type=int, default=10)
    trace = sub.add_parser('trace', help='Follow an edge to its supporting source documents')
    trace.add_argument('edge_id')
    args = parser.parse_args()
    if args.command == 'stats':
        print((ROOT / 'data/statistics.json').read_text(encoding='utf-8'))
        return
    entities = {r['entity_id']: r for r in rows('canonical_entities.jsonl')}
    edges = rows('fused_edges.jsonl')
    def label(eid):
        return entities[eid]['canonical_name'] + ' [' + eid + ']'
    if args.command == 'company':
        if args.limit < 1:
            parser.error('--limit must be positive')
        query = args.query.casefold()
        matches = [r for r in entities.values() if r['entity_type'] == 'Company'
                   and (query == r['entity_id'].casefold()
                        or any(query in n.casefold() for n in [r['canonical_name']] + r['aliases']))]
        if not matches:
            parser.exit(1, 'No matching company; absence of a match is not proof of real-world absence.\n')
        for r in matches:
            print('\n' + label(r['entity_id']))
            selected = [e for e in edges if r['entity_id'] in (e['subject'], e['object'])
                        and (not args.relation or e['relation'] == args.relation.upper())]
            print(f'{len(selected)} related edges; showing at most {args.limit}.')
            for e in selected[:args.limit]:
                print(f"  {label(e['subject'])} --{e['relation']}--> {label(e['object'])}")
                print(f"    {e['edge_id']} | sources={','.join(e['source_types'])} | confidence={e['confidence']} | conflict={e['conflict']}")
    else:
        edge = next((e for e in edges if e['edge_id'] == args.edge_id), None)
        if edge is None:
            parser.exit(1, 'Unknown edge ID.\n')
        claims = {r['claim_id']: r for r in rows('claim_provenance.jsonl')}
        documents = {r['document_id']: r for r in rows('documents.jsonl')}
        flags = {r['edge_id']: r for r in rows('edge_review_flags.jsonl')}
        print(f"{label(edge['subject'])} --{edge['relation']}--> {label(edge['object'])}")
        print('Review flags:', json.dumps(flags[edge['edge_id']]))
        for cid in edge['claim_ids']:
            claim = claims[cid]
            doc = documents[claim['document_id']]
            print(f"  {cid} | {claim['source']} | {claim['document_id']} | observed={claim['publication_date'] or 'unknown'}")
            print('    ' + doc['source_url'])
        print('Source documents are linked, not fetched. Quotation text is not distributed in this package.')

if __name__ == '__main__':
    main()
