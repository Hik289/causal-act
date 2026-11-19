"""One pre-registered training-prior correction; holdout is validation-gated."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import forum_expert_iteration as m


def weighted_ce(logits, label, weights):
    return F.cross_entropy(logits, label) * weights[int(label.item())]


def weighted_loss(net, row, categories, variant, rng):
    weights = variant['class_weights']
    original = net('A', row)
    label = torch.tensor([0], device=original.device)
    box = torch.tensor([m.norm_box(row['variants']['original'], row['size'])], device=original.device)
    loss = weighted_ce(original[:, :2], label, weights) + F.mse_loss(original[:, 2:].sigmoid(), box)
    category = rng.choice(categories)
    edited = net('A', row, category)
    frame = row['variants'][category]
    label2 = torch.tensor([frame['action']], device=edited.device)
    box2 = torch.tensor([m.norm_box(frame, row['size'])], device=edited.device)
    loss = loss + weighted_ce(edited[:, :2], label2, weights) + F.mse_loss(edited[:, 2:].sigmoid(), box2)
    if variant['relations'] and category == 'unrelated':
        loss = loss + F.kl_div(F.log_softmax(edited[:, :2], dim=-1),
                             F.softmax(original[:, :2].detach(), dim=-1), reduction='batchmean')
    if variant['relations'] and category == 'transform':
        shift = torch.tensor([[row['tau']['dx'], row['tau']['dy']] * 2], device=edited.device)
        loss = loss + F.mse_loss(edited[:, 2:].sigmoid(), original[:, 2:].sigmoid().detach() + shift)
    return loss / 2


def validation_report(net, rows, out):
    preds = {}
    net.eval()
    with (out / 'validation_predictions.jsonl').open('x') as stream, torch.no_grad():
        for row in rows:
            values = {}
            for category in ['original'] + m.CATEGORIES:
                p = net('A', row, category)
                values[category] = {'action': int(p[0, :2].argmax()),
                                    'bbox': p[0, 2:].sigmoid().cpu().tolist()}
            preds[row['id']] = values
            stream.write(json.dumps({'id': row['id'], 'predictions': values}) + '\n')
    report = m.causal_report(rows, preds)
    m.atomic(out / 'validation_report.json', report)
    return report


def load_final(net, source):
    record = json.loads((source / 'round2_A.json').read_text())
    path = source / record['checkpoint']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    net.restore(torch.load(path, map_location='cpu', weights_only=True))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--method', choices=['static_cf', 'action_refiner', 'base'], required=True)
    args = ap.parse_args()
    run = Path(args.run)
    protocol = json.loads((run / 'protocol.json').read_text())
    out = run / args.method
    out.mkdir(exist_ok=False)
    data = run / 'data/dataset.jsonl'
    meta = json.loads((run / 'data/dataset_complete.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == meta['sha256']
    rows = [json.loads(x) for x in data.read_text().splitlines()]
    for row in rows:
        for frame in row['variants'].values():
            path = data.parent / frame['image']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == frame['sha256']
            frame['image'] = str(path)
    split = {s: [r for r in rows if r['split'] == s] for s in ['train', 'val', 'test']}
    assert [len(split[s]) for s in ['train', 'val', 'test']] == [160, 40, 80]
    groups = {s: {r['forum'] for r in rows_} for s, rows_ in split.items()}
    assert not (groups['train'] & groups['test'] or groups['val'] & groups['test'])
    random.seed(42)
    torch.manual_seed(42)
    net = m.Modules(protocol['model']).to('cuda')
    if args.method == 'base':
        m.atomic(out / 'frozen_config.json', {'training': False, 'selection': 'unchanged pretrained base'})
        m.evaluate(net, split['test'], dict(m.VARIANTS['base']), out, base=True)
        m.atomic(out / 'status.json', {'phase': 'finished', 'valid': 80})
        return
    original = Path(protocol['original_run']) / args.method
    old_checkpoint = load_final(net, original)
    before = out / 'original_validation'
    before.mkdir()
    original_val = validation_report(net, split['val'], before)
    # Reset to exactly the same pretrained initialization. The old test is never loaded.
    net.inputs.cache_clear()
    del net
    torch.cuda.empty_cache()
    random.seed(42)
    torch.manual_seed(42)
    net = m.Modules(protocol['model']).to('cuda')
    variant = dict(m.VARIANTS[args.method], relations=args.method != 'static_cf',
                   class_weights=protocol['class_weights'])
    m.a_loss = weighted_loss
    settings = SimpleNamespace(seed=42, epochs=3, lr=1e-5, accumulation=4)
    training = out / 'balanced_training'
    training.mkdir()
    m.atomic(training / 'protocol.json', {'variant': variant, 'two_rounds': True,
             'selection': 'minimum weighted validation loss; no holdout selection',
             'parameter_audit': net.audit, 'source_sha256': {
                 p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
    for round_index in range(2):
        m.train_role(net, 'A', split['train'], split['val'], variant, settings, round_index, training)
    after = out / 'balanced_validation'
    after.mkdir()
    adjusted_val = validation_report(net, split['val'], after)
    old, new = original_val['metrics'], adjusted_val['metrics']
    gate = (new['PairCausalAcc']['value'] > old['PairCausalAcc']['value'] and
            new['TargetedAcc']['value'] + new['UnrelatedAcc']['value'] >=
            old['TargetedAcc']['value'] + old['UnrelatedAcc']['value'])
    decision = {'passed': gate, 'selection_data': 'validation only',
                'rule': protocol['validation_gate'], 'original': original_val,
                'adjusted': adjusted_val, 'original_checkpoint': old_checkpoint,
                'class_weights': protocol['class_weights'], 'holdout_cases_planned': 80,
                'no_test_inference_before_decision': True}
    m.atomic(out / 'frozen_config.json', decision)
    if not gate:
        m.atomic(out / 'status.json', {'phase': 'finished; validation did not improve',
                 'test_scored': False, 'valid_test_denominator': 0, 'planned_test_cases': 80})
        return
    for setting in ['adjusted', 'original']:
        destination = out / setting
        destination.mkdir()
        if setting == 'original':
            load_final(net, original)
        m.evaluate(net, split['test'], variant, destination)
    m.atomic(out / 'status.json', {'phase': 'finished', 'paired_holdout_cases': 80})


if __name__ == '__main__':
    main()
