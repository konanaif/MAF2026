"""Exactly Base, Base+SWA, FairBatch and LEACE+SWA; no outcome-based retries."""
import csv
import gc
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .runtime import set_random_seed
from MAF.datamodule.kobroadcast.sample import digest, read_rows, save_predictions, write_json
from MAF.metric.kobroadcast.get_metric import calibrate, evaluate
from .models import Classifier
from .preprocess.fairbatch_function import fairbatch_negative_weights
from .constants import METHODS
from .Base import Base
from .SWA import SWA
from .preprocess.FairBatch import FairBatch
from .postprocess.LEACESWA import LEACESWA

MAF_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = MAF_ROOT / 'result' / 'kobroadcast'
METHOD_CLASSES = {'base': Base, 'base_swa': SWA, 'fairbatch': FairBatch, 'leace_swa': LEACESWA}


def seed_all(seed):
    set_random_seed(seed)


def windows(loader, size):
    window = []
    for batch in loader:
        window.append(batch)
        if len(window) == size:
            yield window
            window = []
    if window:
        yield window


def negative_risks(data, logits):
    labels = torch.tensor([int(r['toxicity_label'] == 'toxic') for r in data])
    ce = F.cross_entropy(torch.as_tensor(logits), labels, reduction='none').numpy()
    result = {}
    for group in sorted({g for r in data for g in r['protected_attributes_mentioned']}):
        mask = np.array([r['toxicity_label'] == 'nontoxic' and group in r['protected_attributes_mentioned'] for r in data])
        if mask.any():
            result[group] = float(ce[mask].mean())
    return result


def sample_indices(data, risks, generator):
    labels = [int(r['toxicity_label'] == 'toxic') for r in data]
    groups = [r['protected_attributes_mentioned'] for r in data]
    weights = fairbatch_negative_weights(labels, groups, risks, strength=1., max_weight=5.)
    labels = torch.tensor(labels)
    negative, positive = torch.where(labels == 0)[0], torch.where(labels == 1)[0]
    sampled = negative[torch.multinomial(weights[negative], len(negative), replacement=True, generator=generator)]
    order = torch.cat([sampled, positive])
    order = order[torch.randperm(len(order), generator=generator)].tolist()
    assert len(order) == len(data) and sum(int(labels[i]) for i in order) == len(positive)
    return order


def checkpoint_hashes(directory):
    files = list(Path(directory).glob('model/*.safetensors')) + list(Path(directory).glob('adapter/*.safetensors'))
    if not files:
        raise ValueError('Checkpoint has no safetensors weights')
    return {str(p.relative_to(directory)): digest(p) for p in files}


def train_checkpoint(model_name, method, cfg, paths, output, seed, device):
    if method not in ('base', 'base_swa', 'fairbatch'):
        raise ValueError('Unsupported training method')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    train, dev = read_rows(paths['train']), read_rows(paths['dev'])
    if set(r['toxicity_label'] for r in train) != {'nontoxic', 'toxic'}:
        raise ValueError('TRAIN must contain both labels')
    classifier = Classifier(model_name, cfg, device)
    train_ds, dev_ds = classifier.encode(train), classifier.encode(dev)
    model = classifier.model
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg['lr'], weight_decay=.01)
    updates_per_epoch = math.ceil(math.ceil(len(train) / cfg['batch_size']) / cfg['accumulation'])
    total = cfg['epochs'] * updates_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, math.floor(total * .1), total)
    swa = method == 'base_swa'
    swa_scheduler = SWALR(optimizer, swa_lr=cfg['swa_lr']) if swa else None
    averaged, swa_n, update = {}, 0, 0
    generator = torch.Generator().manual_seed(seed)
    risks, history = {}, []
    best_f1, best_state, best_epoch = -1., None, None
    write_json(output / 'started.json', dict(model=model_name, method=method, cfg=cfg, seed=seed, training_data_sha256=digest(paths['train']), dev_data_sha256=digest(paths['dev']), class_weight=1.))
    for epoch in range(cfg['epochs']):
        if method == 'fairbatch':
            loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=sample_indices(train, risks, generator), collate_fn=train_ds.collate)
        else:
            loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, generator=generator, collate_fn=train_ds.collate)
        model.train()
        running, seen = 0., 0
        for window in windows(loader, cfg['accumulation']):
            # Incomplete final microbatch/window uses its actual sample count.
            n = sum(len(batch['labels']) for batch in window)
            optimizer.zero_grad(set_to_none=True)
            for batch in window:
                logits, _ = classifier.forward(batch)
                loss_sum = F.cross_entropy(logits, batch['labels'].to(classifier.device), reduction='sum')
                if not torch.isfinite(loss_sum):
                    raise RuntimeError('Nonfinite training loss')
                (loss_sum / n).backward()
                running += float(loss_sum.detach()); seen += len(batch['labels'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); update += 1
            active = swa and ((update >= max(1, int(total * .5))) if classifier.causal else epoch >= cfg['swa_start'])
            if active:
                swa_scheduler.step(); swa_n += 1
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            value = param.detach().float()
                            if name not in averaged:
                                averaged[name] = value.clone()
                            else:
                                averaged[name].add_((value - averaged[name]) / swa_n)
            else:
                scheduler.step()
        # Diagnostics do not choose a SWA checkpoint or consult TEST.
        dev_logits, _ = classifier.infer(dev_ds)
        dev_metrics = evaluate(dev, dev_logits)['overall']
        if classifier.causal and not swa and dev_metrics['toxic_f1'] > best_f1:
            best_f1 = dev_metrics['toxic_f1']; best_state = classifier.trainable_state(); best_epoch = epoch + 1
        if method == 'fairbatch':
            logits, _ = classifier.infer(train_ds)
            risks = negative_risks(train, logits)
        record = dict(epoch=epoch + 1, updates=update, swa_states=swa_n, loss=running / seen, dev=dev_metrics, group_negative_CE=risks)
        history.append(record)
        write_json(output / 'progress.json', dict(history=history))
        print(f'{model_name} {method}: epoch {epoch + 1}/{cfg["epochs"]}, DEV Toxic-F1={dev_metrics["toxic_f1"]:.4f}', flush=True)
    assert update == total
    if swa:
        if not swa_n:
            raise ValueError('No SWA states: swa_start must precede the final epoch')
        classifier.set_trainable_state(averaged)
    elif classifier.causal:
        classifier.set_trainable_state(best_state)
    classifier.save(output)
    metadata = dict(model=model_name, method=method, cfg=cfg, seed=seed, class_weight=1., updates=update,
                    swa_states=swa_n, best_epoch=best_epoch, training_data_sha256=digest(paths['train']),
                    dev_data_sha256=digest(paths['dev']), checkpoint_sha256=checkpoint_hashes(output),
                    train_truncated=train_ds.truncated, dev_truncated=dev_ds.truncated, history=history)
    write_json(output / 'checkpoint.json', metadata)
    del averaged, best_state, classifier, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def open_checkpoint(model_name, checkpoint, paths, device):
    import json
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / 'checkpoint.json').read_text())
    if metadata['model'] != model_name:
        raise ValueError('Checkpoint backbone differs from --model')
    if metadata['training_data_sha256'] != digest(paths['train']) or metadata['dev_data_sha256'] != digest(paths['dev']):
        raise ValueError('Checkpoint TRAIN/DEV differ from supplied splits')
    if metadata['checkpoint_sha256'] != checkpoint_hashes(checkpoint):
        raise ValueError('Checkpoint weights changed after training')
    return Classifier(model_name, metadata['cfg'], device, checkpoint), metadata


def evaluate_checkpoint(model_name, method, checkpoint, paths, output, device, rank=128, strength=.8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    classifier, metadata = open_checkpoint(model_name, checkpoint, paths, device)
    dev = read_rows(paths['dev'])
    erasure = None
    if method == 'leace_swa':
        if metadata['method'] != 'base_swa':
            raise ValueError('LEACE+SWA requires a base_swa checkpoint')
        from .postprocess import leace_function as leace
        train = read_rows(paths['train'])
        train_logits, train_hidden = classifier.infer(classifier.encode(train), features=True)
        erasure, diagnostics = leace.fit(train_hidden, train_logits, train, rank=rank, strength=strength)
        np.savez_compressed(output / 'leace.npz', **erasure)
        write_json(output / 'leace_fit.json', diagnostics)
        del train_hidden, train_logits
    dev_logits, dev_hidden = classifier.infer(classifier.encode(dev), features=erasure is not None)
    if erasure is not None:
        dev_logits = leace.predict(dev_hidden, dev_logits, erasure)
    save_predictions(output / 'dev_predictions.jsonl', dev, dev_logits)
    selection = calibrate(dev, dev_logits)
    write_json(output / 'selection.json', selection)
    # Both eraser and threshold are fixed before the TEST forward pass.
    test = read_rows(paths['test'])
    test_ds = classifier.encode(test)
    test_logits, test_hidden = classifier.infer(test_ds, features=erasure is not None)
    if erasure is not None:
        test_logits = leace.predict(test_hidden, test_logits, erasure)
    save_predictions(output / 'test_predictions.jsonl', test, test_logits)
    result = dict(model=model_name, method=method, checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=metadata['checkpoint_sha256'],
                  n_test=len(test), threshold=selection['threshold'], argmax=evaluate(test, test_logits),
                  calibrated=evaluate(test, test_logits, selection['threshold']), test_data_sha256=digest(paths['test']),
                  test_predictions_sha256=digest(output / 'test_predictions.jsonl'), test_truncated=test_ds.truncated,
                  selection_sha256=digest(output / 'selection.json'), test_used_for_tuning=False)
    if erasure is not None:
        result['leace_model_sha256'] = digest(output / 'leace.npz')
    write_json(output / 'metrics.json', result)
    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def export_tables(output, results):
    for decision in ('argmax', 'calibrated'):
        table = []
        for result in results:
            value = result[decision]
            table.append(dict(model=result['model'], method=result['method'], decision=decision, **value['overall'],
                              max_supported_fpr=value['max_supported_fpr'], max_unsupported_fpr=value['max_unsupported_fpr'],
                              TN_FP_FN_TP=' / '.join(str(value['overall'][k]) for k in ('tn', 'fp', 'fn', 'tp'))))
        with (Path(output) / f'metrics_{decision}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    write_json(Path(output) / 'all_results.json', results)


def execute(args, paths, audit, configs):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --dry-run or a CPU RoBERTa smoke check')
    if 'qwen3-4b' in configs and device.type != 'cuda':
        raise ValueError('Qwen3-4B requires CUDA; select one RoBERTa model for CPU execution')
    output = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    methods = METHODS if args.method == 'all' else (args.method,)
    # MAF 배치: 패키지 소스 + 분리된 로더·지표 파일. 호스트 전체 requirements.txt는 제외.
    package_root = Path(__file__).resolve().parent
    sources = sorted(p for p in package_root.rglob('*.py') if 'tests' not in p.relative_to(package_root).parts)
    sources += [package_root / 'prompts/toxicity.md', MAF_ROOT / 'metric/kobroadcast/get_metric.py', MAF_ROOT / 'datamodule/kobroadcast/sample.py']
    source_root = MAF_ROOT
    write_json(output / 'protocol.json', dict(models=list(configs), methods=list(methods), configs=configs, seed=args.seed,
                                             data=audit, class_weight=1., leace_rank=args.leace_rank, leace_strength=args.leace_strength,
                                             source_sha256={str(p.relative_to(source_root)): digest(p) for p in sources},
                                             environment=dict(python=sys.version, torch=torch.__version__, transformers=__import__('transformers').__version__)))
    results = []
    for model_name, cfg in configs.items():
        for method in methods:
            algorithm = METHOD_CLASSES[method](model_name, cfg, paths, output, args.seed, args.device,
                                               rank=args.leace_rank, strength=args.leace_strength,
                                               checkpoint=args.checkpoint)
            result = algorithm.run()
            results.append(result)
            export_tables(output, results)
            print(f'Completed {model_name} / {method}: {result["argmax"]["overall"]}', flush=True)
    print(f'Results: {output}', flush=True)
    return output
