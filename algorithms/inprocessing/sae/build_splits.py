#!/usr/bin/env python3
"""Build train/dev/test.jsonl from AI Hub 71557/591 label zips and the label files in data/sae/.

Raw broadcast text is not distributed. The text is regenerated from the user's
own AI Hub downloads, joined with data/sae/kobroadcast_silver_*_labels.jsonl by sample_id, and checked
against the recorded text SHA-256 of every row and the EXPECTED values below.
Standard library only (Python 3.10+).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

LABELS_DIR = Path(__file__).resolve().parents[3] / 'data' / 'sae'
SPLITS = ('train', 'dev', 'test')
EXPECTED = {
    'train': {'n': 700, 'n_toxic': 80, 'sha256': 'd19f19acfbac5fcc1994f097ec16a67bb703d1f9cdfc00e68fc4b5535243493c'},
    'dev': {'n': 150, 'n_toxic': 17, 'sha256': '3472133f1c5212c8b0aeb40377d1f94fb9419366f4f59e576e037c9ac4ca5e36'},
    'test': {'n': 150, 'n_toxic': 18, 'sha256': '3e9eb520f54e79b56098e28badeca0039d46cdef0704d3a049ccd2881685e577'},
}
LABEL_FIELDS = ('labeling_source', 'toxicity_label', 'broadcast_risk_decision', 'risk_labels', 'bias_categories',
                'context_types', 'protected_attributes_mentioned', 'severity', 'codex_confidence',
                'needs_second_llm_pass', 'rationale_short', 'risk_spans')
NEWS = dict(source='aihub_71557_news_script_anchor_speech', max_chars=1400,
            rationale='unlabeled AI Hub 71557 grouped news-script segment; GPT/Codex silver-labeling required',
            license='AI Hub 71557; internal research use under AI Hub terms; do not redistribute raw transcript/audio without permission.')
SCRIPT = dict(source='aihub_591_broadcast_script_summary', max_chars=1500,
              rationale='unlabeled AI Hub 591 broadcast-script passage; GPT/Codex silver-labeling required',
              license='AI Hub 591; internal research use under AI Hub terms; do not redistribute raw text without permission.')


def clean_text(text):
    text = str(text).replace('﻿', '')
    text = re.sub(r'\r', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def safe_id(value, fallback='doc'):
    s = re.sub(r'[^0-9A-Za-z가-힣_.-]+', '-', str(value or fallback)).strip('-')
    return s[:120] or fallback


def guess_genre(doc_type):
    dt = str(doc_type or '').lower()
    for genre, keys in (('news', ('뉴스', 'news')), ('current_affairs', ('시사', 'current', 'event')),
                        ('drama', ('드라마', 'drama')), ('entertainment', ('예능', 'enter', 'entertainment')),
                        ('documentary', ('교양', 'culture', 'documentary'))):
        if dt and any(k in dt for k in keys):
            return genre
    return 'other'


def segment(text, max_chars):
    if len(text) <= max_chars:
        return [text]
    chunks, cur = [], ''
    normalized = re.sub(r'\n{2,}', '<PARA_SPLIT>', text)
    normalized = re.sub(r'(?:(?:다\.)|(?:요\.)|[.!?。])\s+', lambda m: m.group(0).rstrip() + '<SENT_SPLIT>', normalized)
    for p in re.split(r'<PARA_SPLIT>|<SENT_SPLIT>', normalized):
        p = p.strip()
        if not p:
            continue
        for h in ([p[i:i + max_chars] for i in range(0, len(p), max_chars)] if len(p) > max_chars else [p]):
            if len(cur) + len(h) + 1 <= max_chars:
                cur = (cur + '\n' + h).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = h
    if cur:
        chunks.append(cur)
    return chunks or [text[:max_chars]]


def row(spec, sample_id, doc_id, genre, date, speakers, text):
    return {'sample_id': sample_id, 'source': spec['source'], 'source_doc_id': doc_id, 'genre': genre,
            'broadcast_date': date, 'speakers': speakers, 'text': text, 'protected_attributes_mentioned': [],
            'toxicity_label': 'nontoxic', 'broadcast_risk_decision': 'acceptable', 'labeling_source': '',
            'label_confidence': 0.0, 'needs_second_llm_pass': False, 'risk_labels': ['no_issue'],
            'bias_categories': ['no_bias'], 'context_types': ['non_discriminatory_context'], 'risk_spans': [],
            'severity': 0, 'annotator_rationale': spec['rationale'], 'recommended_revision': '',
            'license_note': spec['license']}


def zip_json(paths):
    for path in paths:
        with ZipFile(path) as z:
            for name in z.namelist():
                if not name.lower().endswith('.json'):
                    continue
                data = z.read(name)
                for enc in ('utf-8', 'utf-8-sig', 'cp949', 'euc-kr'):
                    try:
                        yield name, json.loads(data.decode(enc))
                        break
                    except Exception:
                        pass


def news_rows(paths, wanted_docs):
    """AI Hub 71557: utterances grouped by script id and index order."""
    docs, speakers, dates = defaultdict(list), defaultdict(set), {}
    for name, obj in zip_json(paths):
        script = obj.get('script', {}) if isinstance(obj, dict) else {}
        speaker = obj.get('speaker', {}) if isinstance(obj, dict) else {}
        if not isinstance(script, dict):
            continue
        doc_id = safe_id(script.get('id') or Path(name).stem)
        text = clean_text(script.get('text') or '')
        if doc_id not in wanted_docs or len(text) < 3:
            continue
        docs[doc_id].append((int(script.get('index') or 0), text))
        dates.setdefault(doc_id, script.get('press_date'))
        for k in ('id', 'age', 'sex', 'job'):
            if isinstance(speaker, dict) and speaker.get(k):
                speakers[doc_id].add(f'{k}:{speaker[k]}')
    for doc_id in sorted(docs):
        joined = '\n'.join(t for _, t in sorted(docs[doc_id], key=lambda x: x[0]))
        date = str(dates[doc_id]) if dates.get(doc_id) else None
        for i, seg in enumerate(segment(joined, NEWS['max_chars']), 1):
            yield row(NEWS, f'aihub71557-{doc_id}-{i:02d}', doc_id, 'news', date, sorted(speakers[doc_id]), seg)


def script_rows(paths):
    """AI Hub 591: Meta.passage, first occurrence of each passage."""
    seen = set()
    for name, obj in zip_json(paths):
        meta = obj.get('Meta', {}) if isinstance(obj, dict) else {}
        if not isinstance(meta, dict):
            continue
        passage = clean_text(meta.get('passage') or '')
        if len(passage) < 20 or passage in seen:
            continue
        seen.add(passage)
        passage_id = safe_id(meta.get('passage_id') or Path(name).stem)
        year = meta.get('published_year')
        for i, seg in enumerate(segment(passage, SCRIPT['max_chars']), 1):
            yield row(SCRIPT, f'aihub591-{passage_id}-{i:02d}', safe_id(meta.get('doc_id') or passage_id),
                      guess_genre(meta.get('doc_type')), str(year) if year else None, [], seg)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def verify(directory, expected):
    ok = True
    for split in SPLITS:
        path = Path(directory) / f'{split}.jsonl'
        rows = read_jsonl(path)
        got = {'n': len(rows), 'n_toxic': sum(r['toxicity_label'] == 'toxic' for r in rows),
               'sha256': sha256(path.read_bytes())}
        rows_ok = all(sha256(r['text'].encode('utf-8')) == r.get('text_sha256') for r in rows)
        passed = got == expected[split] and rows_ok
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {split:5s} n={got['n']} n_toxic={got['n_toxic']} sha256={got['sha256']}")
    print('ALL CHECKS PASSED' if ok else 'VERIFICATION FAILED: compare with README.md')
    return 0 if ok else 1


def build(a, expected):
    labels = {s: read_jsonl(LABELS_DIR / f'kobroadcast_silver_{s}_labels.jsonl') for s in SPLITS}
    wanted = {r['sample_id']: r for rows in labels.values() for r in rows}
    found = defaultdict(list)
    news_docs = {r['source_doc_id'] for r in wanted.values() if r['source'] == NEWS['source']}
    for r in news_rows(a.news_zip, news_docs):
        found[r['sample_id']].append(r)
    for r in script_rows(a.script_zip):
        if r['sample_id'] in wanted:
            found[r['sample_id']].append(r)
    outputs, missing = {}, []
    for split, rows in labels.items():
        outputs[split] = []
        for lab in rows:
            match = [r for r in found.get(lab['sample_id'], []) if sha256(r['text'].encode('utf-8')) == lab['text_sha256']]
            if not match:
                missing.append(f"{split}:{lab['sample_id']}")
                continue
            out = dict(match[0], text_sha256=lab['text_sha256'])
            out.update({k: lab[k] for k in LABEL_FIELDS})
            outputs[split].append(out)
    if missing:
        print(f'FAIL: {len(missing)} of {len(wanted)} rows were not found with the recorded text SHA-256; '
              'check that the four AI Hub label zips are given in the documented order.', file=sys.stderr)
        print('  first: ' + ', '.join(missing[:5]), file=sys.stderr)
        return 1
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        if (a.output_dir / f'{split}.jsonl').exists():
            sys.exit(f'{a.output_dir / split}.jsonl already exists; choose an empty --output-dir')
    for split, rows in outputs.items():
        (a.output_dir / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return verify(a.output_dir, expected)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--news-zip', type=Path, nargs=2, metavar=('VL.zip', 'TL.zip'), help='AI Hub 71557 label zips: Validation VL.zip, Training TL.zip')
    p.add_argument('--script-zip', type=Path, nargs=2, metavar=('VL1.zip', 'TL1.zip'), help='AI Hub 591 label zips: Validation VL1.zip, Training TL1.zip')
    p.add_argument('--output-dir', type=Path, help='Empty folder for train/dev/test.jsonl')
    p.add_argument('--verify-only', type=Path, metavar='DIR', help='Check an existing train/dev/test.jsonl folder')
    a = p.parse_args(argv)
    expected = EXPECTED
    if a.verify_only:
        return verify(a.verify_only, expected)
    if not (a.news_zip and a.script_zip and a.output_dir):
        p.error('--news-zip, --script-zip and --output-dir are required unless --verify-only is used')
    return build(a, expected)


if __name__ == '__main__':
    sys.exit(main())
