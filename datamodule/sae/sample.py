"""Load broadcast JSONL splits and save predictions without source text."""
import hashlib
import json
from pathlib import Path

LABELS = ('nontoxic', 'toxic')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def split_paths(directory):
    directory = Path(directory).expanduser().resolve()
    result = {}
    for split in ('train', 'dev', 'test'):
        candidates = [directory / f'{split}.jsonl', directory / f'kobroadcast_silver_{split}.jsonl']
        found = [path for path in candidates if path.is_file()]
        if len(found) != 1:
            raise ValueError(f'{split}: provide exactly one split file in {directory}')
        result[split] = found[0]
    return result


def read_rows(path):
    data = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not data:
        raise ValueError(f'Empty split: {path}')
    ids = set()
    for row in data:
        if not isinstance(row.get('sample_id'), str) or row['sample_id'] in ids:
            raise ValueError('Each row must have a unique string sample_id')
        ids.add(row['sample_id'])
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Each row must contain nonempty text')
        if row.get('toxicity_label') not in LABELS:
            raise ValueError('toxicity_label must be nontoxic or toxic')
        for key in ('protected_attributes_mentioned', 'bias_categories'):
            if not isinstance(row.get(key), list) or not all(isinstance(x, str) for x in row[key]):
                raise ValueError(f'{key} must be a list of strings')
        if row.get('broadcast_risk_decision') not in ('acceptable', 'needs_review', 'problematic'):
            raise ValueError('Missing or invalid broadcast_risk_decision')
    return data


def text_hash(row):
    return hashlib.sha256(row['text'].encode()).hexdigest()


def audit_splits(paths):
    data = {split: read_rows(path) for split, path in paths.items()}
    result = {'splits': {}, 'overlap': {}}
    for split, rows in data.items():
        result['splits'][split] = {'n': len(rows), 'n_toxic': sum(r['toxicity_label'] == 'toxic' for r in rows), 'sha256': digest(paths[split])}
    for first, second in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        a, b = data[first], data[second]
        shared_ids = {r['sample_id'] for r in a} & {r['sample_id'] for r in b}
        if shared_ids:
            raise ValueError(f'Sample IDs overlap between {first} and {second}')
        hashes = {text_hash(r) for r in a} & {text_hash(r) for r in b}
        documents = lambda rows: {(r.get('source'), r['source_doc_id']) for r in rows if r.get('source_doc_id') is not None}
        result['overlap'][f'{first}:{second}'] = {'same_texts': len(hashes), 'affected_later_rows': sum(text_hash(r) in hashes for r in b), 'shared_source_documents': len(documents(a) & documents(b))}
    # Older released split files have text/document overlap. Keep them intact and disclose it.
    return result


def save_predictions(path, data, logits):
    records = []
    for row, scores in zip(data, logits):
        records.append({'sample_id': row['sample_id'], 'gold_label': row['toxicity_label'], 'labels': list(LABELS), 'logits': [float(v) for v in scores], 'predicted_label': LABELS[int(scores[1] > scores[0])]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
