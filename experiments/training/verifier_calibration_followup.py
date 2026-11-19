"""Validation-only fusion calibration with a frozen, disjoint paired holdout.

This changes one scalar multiplying V's log probability. It never retrains or
selects checkpoints, and never loads the original study's test predictions.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from forum_expert_iteration import causal_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fused_action(action_log_probs, verifier_log_probs, weight):
    # Match the original runner's float32 arithmetic, including its tie rule.
    values = torch.tensor(action_log_probs, dtype=torch.float32)
    values += float(weight) * torch.tensor(verifier_log_probs, dtype=torch.float32)
    return int(values.argmax())


def predictions(records, weight):
    return {row['id']: {category: {
        'action': fused_action(scores['A'], scores['V'], weight),
        'bbox': scores['bbox'],
    } for category, scores in row['scores'].items()} for row in records}


def choose_weight(reports, grid, original_weight=1.0):
    def values(report):
        m = report['metrics']
        return (m['PairCausalAcc']['numerator'],
                m['TargetedAcc']['numerator'] + m['UnrelatedAcc']['numerator'])
    old = values(reports[str(original_weight)])
    eligible = [w for w in grid if values(reports[str(w)])[0] > old[0]
                and values(reports[str(w)])[1] >= old[1]]
    if not eligible:
        return None
    return max(eligible, key=lambda w: (*values(reports[str(w)]), -abs(w-original_weight)))


def collect(net, rows, destination, m, status, include_i=False):
    existing = []
    if destination.exists():
        existing = [json.loads(line) for line in destination.read_text().splitlines()]
    ids = [row['id'] for row in existing]
    expected = {row['id'] for row in rows}
    if len(set(ids)) != len(ids) or not set(ids) <= expected:
        raise RuntimeError('Existing predictions conflict with the frozen split')
    seen = set(ids)
    net.eval()
    with destination.open('a') as stream, torch.no_grad():
        for row in rows:
            if row['id'] in seen:
                continue
            result = {'id': row['id'], 'forum': row['forum'], 'scores': {}}
            for category in ['original'] + m.CATEGORIES:
                a = net('A', row, category)
                v = [float(F.log_softmax(net('V', row, category, candidate), dim=-1)[0, 1])
                     for candidate in range(2)]
                result['scores'][category] = {
                    'A': F.log_softmax(a[:, :2], dim=-1)[0].cpu().tolist(),
                    'V': v, 'bbox': a[0, 2:].sigmoid().cpu().tolist(),
                }
            if include_i:
                result['I_logits'] = net('I', row)[0].cpu().tolist()
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            existing.append(result)
            status(len(existing), len(rows))
    assert {row['id'] for row in existing} == expected
    return existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--method', choices=['full', 'action_verifier'], required=True)
    args = ap.parse_args()
    run = Path(args.run)
    protocol = json.loads((run / 'protocol.json').read_text())
    for name, sha256 in protocol['code_sha256'].items():
        assert digest(run / 'code' / name) == sha256
    source = Path(protocol['trainer'])
    assert digest(source) == protocol['trainer_sha256']
    spec = importlib.util.spec_from_file_location('frozen_forum_trainer', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = run / args.method
    out.mkdir(exist_ok=True)
    if (out / 'pair_report.json').exists():
        return
    data = Path(protocol['data'])
    assert digest(data) == protocol['data_sha256']
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    split = {s: [row for row in rows if row['split'] == s] for s in ['train', 'val', 'test']}
    assert [len(split[s]) for s in ['train', 'val', 'test']] == [160, 40, 80]
    groups = {s: {r['forum'] for r in rr} for s, rr in split.items()}
    assert not (groups['train'] & groups['test'] or groups['val'] & groups['test'])
    assert len(groups['test']) == 20
    original_data = Path(protocol['original_data'])
    assert digest(original_data) == protocol['original_data_sha256']
    old_forums = {json.loads(line)['forum'] for line in original_data.read_text().splitlines()}
    assert not groups['test'] & old_forums
    for row in rows:
        for frame in row['variants'].values():
            path = data.parent / frame['image']
            assert digest(path) == frame['sha256']
            frame['image'] = str(path)
    random.seed(42)
    torch.manual_seed(42)
    net = m.Modules(protocol['model']).to('cuda')
    original = Path(protocol['original_run']) / args.method
    checkpoints = {}
    for role in ('AVI' if args.method == 'full' else 'AV'):
        record = json.loads((original / ('round2_' + role + '.json')).read_text())
        path = original / record['checkpoint']
        assert digest(path) == record['sha256'] == protocol['checkpoints'][args.method][role]['sha256']
        net.restore(torch.load(path, map_location='cpu', weights_only=True))
        checkpoints[role] = record
    def progress(phase):
        return lambda done, planned: m.atomic(out / 'status.json', {
            'phase': phase, 'complete_cases': done, 'planned_cases': planned,
            'not_end_to_end': True,
        })
    validation = collect(net, split['val'], out / 'validation_raw_scores.jsonl', m,
                         progress('validation-only calibration'), args.method == 'full')
    grid = protocol['validation_weight_grid']
    reports = {str(w): causal_report(split['val'], predictions(validation, w)) for w in grid}
    selected = choose_weight(reports, grid)
    decision = {'passed': selected is not None, 'selected_weight': selected,
                'verifier_disabled': selected == 0.0,
                'original_weight': 1.0, 'selection_split': 'validation only',
                'validation_reports': reports, 'checkpoints': checkpoints,
                'validation_raw_sha256': digest(out / 'validation_raw_scores.jsonl'),
                'protocol_sha256': digest(run / 'protocol.json'),
                'holdout_inference_before_freeze': False}
    frozen = out / 'frozen_config.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == decision
    else:
        m.atomic(frozen, decision)
    if selected is None:
        m.atomic(out / 'status.json', {'phase': 'finished; validation gate did not pass',
                                     'valid_test_denominator': 0, 'planned_test_cases': 80})
        return
    # Only this branch opens the independent test images for model inference.
    test = collect(net, split['test'], out / 'heldout_raw_scores.jsonl', m,
                   progress('frozen paired holdout'), args.method == 'full')
    paired = {name: causal_report(split['test'], predictions(test, weight))
              for name, weight in [('original', 1.0), ('calibrated', selected)]}
    m.atomic(out / 'pair_report.json', {
        **paired, 'paired_cases': 80, 'independent_forums': 20,
        'case_structure': '2goals x2paraphrases per forum; not IID',
        'selected_weight': selected, 'original_weight': 1.0,
        'data_sha256': protocol['data_sha256'], 'raw_scores_sha256': digest(out / 'heldout_raw_scores.jsonl'),
        'frozen_config_sha256': digest(frozen), 'checkpoints': checkpoints,
        'scope': 'semantic click/stop and evidence; no executed pointer success or real-world safety',
        'original_study_test_predictions_loaded': False,
        'paired_inference': 'same raw A/V outputs, same images and checkpoints; fusion scalar only',
    })
    m.atomic(out / 'status.json', {'phase': 'finished', 'paired_holdout_cases': 80})


if __name__ == '__main__':
    main()
