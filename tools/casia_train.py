"""Use frozen trainers and select CASIA checkpoints on tampered-only F1."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path


def add_tampered_metrics(summary):
    families = summary.get('by_family', {})
    manipulated = [v for k, v in families.items() if k in ('copy_move', 'splicing')]
    if not manipulated:
        return summary
    summary = dict(summary)
    count = sum(v['images'] for v in manipulated)
    summary['tampered_images'] = count
    for metric in ('f1', 'iou', 'mae', 'fpr'):
        values = [v for v in manipulated if v.get(metric) is not None]
        summary['tampered_' + metric] = sum(v[metric] * v['images'] for v in values) / sum(v['images'] for v in values)
    n_auc = sum(v['auroc_defined_images'] for v in manipulated)
    summary['tampered_pixel_auroc_macro_hist'] = (sum(
        v['pixel_auroc_macro_hist'] * v['auroc_defined_images'] for v in manipulated
        if v.get('pixel_auroc_macro_hist') is not None) / n_auc) if n_auc else None
    if 'authentic' in families:
        summary['authentic_fpr'] = families['authentic']['fpr']
        summary['authentic_images'] = families['authentic']['images']
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--trainer', choices=['rgb', 'dct'], required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    trainer = importlib.import_module('cod_iml_ablation.train' if args.trainer == 'rgb' else 'cod_iml_frequency_v3.train')
    config = trainer.resolve_config(args.config)
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if config['casia_evaluation_runner_sha256'] != actual:
        raise ValueError('CASIA evaluation wrapper differs from the frozen configuration')
    original = trainer.evaluate

    def evaluate(*pos, **kwargs):
        return add_tampered_metrics(original(*pos, **kwargs))

    trainer.evaluate = evaluate
    trainer.run_experiment(config, resume=args.resume)


if __name__ == '__main__':
    main()
