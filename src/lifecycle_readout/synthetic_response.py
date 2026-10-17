"""Small oracle-probability response diagnostic, not a temporal graph benchmark."""
from pathlib import Path
import copy
import json
import platform
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .benchmark import save_json, sha256
from .modeling.monotone_residual import NestedMonotoneResidual


def oracle_logits(context, elapsed, scenario):
    """Ground-truth functions deliberately differ from exponential mixtures."""
    c = context[:, 0]
    base = 1. + .4 * c
    if scenario == 'null':
        correction = np.zeros_like(elapsed)
    elif scenario == 'homogeneous_decay':
        correction = -1.8 * np.log1p(elapsed / 1.5)
    elif scenario == 'contextual_decay':
        amplitude = .2 + (c + 1.)
        scale = .4 + 1.3 * (1. - c)
        correction = -amplitude * np.log1p(elapsed / scale)
    elif scenario == 'increasing':
        correction = .8 * np.log1p(elapsed)
    else:
        raise ValueError(scenario)
    return base, base + correction


def probability(logits):
    return np.exp(-np.logaddexp(0., -np.asarray(logits, dtype=np.float64)))


def oracle_metrics(logits, truth_logits):
    """Expected Bernoulli loss over known probabilities, without sampled labels."""
    logits, truth_logits = np.asarray(logits, dtype=np.float64), np.asarray(truth_logits, dtype=np.float64)
    if logits.shape != truth_logits.shape or not np.isfinite(logits).all() or not np.isfinite(truth_logits).all():
        raise ValueError('Finite matching logit arrays required')
    p = probability(truth_logits)
    expected = np.logaddexp(0., logits) - p * logits
    entropy = np.logaddexp(0., truth_logits) - p * truth_logits
    return {'excess_expected_bce': float(np.mean(expected - entropy)),
            'probability_mse': float(np.mean((probability(logits) - p) ** 2))}


class FlexibleResidual(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, context, elapsed):
        x = torch.cat((context, torch.log1p(elapsed[:, None])), dim=1)
        return {'logits': base + self.net(x).squeeze(-1)}


def tensors(part):
    return tuple(torch.as_tensor(part[k], dtype=torch.float32) for k in ('base', 'context', 'elapsed'))


def train_response(kind, seed, parts, settings, folder):
    torch.manual_seed(seed)
    if kind not in ('scalar', 'mixture', 'flexible'):
        raise ValueError('Unknown response model')
    if kind == 'flexible':
        model = FlexibleResidual(settings['hidden_dim'])
    else:
        model = NestedMonotoneResidual(1, n_components=1 if kind == 'scalar' else settings['components'],
                                       hidden_dim=settings['hidden_dim'], rate_init=settings['rate_init'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    train = tensors(parts['train']); validation = tensors(parts['validation'])
    target = torch.tensor(probability(parts['train']['truth']), dtype=torch.float32)
    best, checkpoint, history = float('inf'), None, []
    started = time.monotonic()
    for step in range(1, settings['steps'] + 1):
        model.train(); out = model(*train)
        loss = F.binary_cross_entropy_with_logits(out['logits'], target)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite synthetic training loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if isinstance(model, NestedMonotoneResidual):
            model.project_parameters_()
        model.eval()
        with torch.no_grad():
            logits = model(*validation)['logits'].numpy()
        metric = oracle_metrics(logits, parts['validation']['truth'])
        history.append({'step': step, 'train_loss': float(loss.detach()), **metric})
        if metric['excess_expected_bce'] < best:
            best, selected_step = metric['excess_expected_bce'], step
            checkpoint = copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint); model.eval()
    checkpoint_path = folder / f'{kind}.pt'
    torch.save({'model': checkpoint, 'selected_step': selected_step, 'settings': settings, 'kind': kind}, checkpoint_path)
    result = {'selected_step': selected_step, 'parameter_count': sum(p.numel() for p in model.parameters()),
              'elapsed_seconds': time.monotonic() - started, 'validation_history': history,
              'checkpoint_sha256': sha256(checkpoint_path), 'endpoints': {}}
    for split, part in parts.items():
        if split == 'train':
            continue
        with torch.no_grad():
            logits = model(*tensors(part))['logits'].numpy()
        np.savez_compressed(folder / f'{kind}_{split}.npz', logits=logits)
        result['endpoints'][split] = oracle_metrics(logits, part['truth'])
    if isinstance(model, NestedMonotoneResidual):
        result['amplitudes'] = model.amplitude.detach().tolist()
        result['rates'] = F.softplus(model.raw_rate).detach().tolist()
    return result


def run_synthetic(settings, protocol, output):
    """Bounded local CPU toy fitting; use Colab for larger experiments."""
    for key in ('seeds', 'models', 'scenarios'):
        if settings[key] != protocol[key] or len(set(settings[key])) != len(settings[key]):
            raise ValueError('Settings must match the declared study without duplicates')
    if (settings['train_count'] > 2048 or settings['validation_count'] > 2048 or settings['steps'] > 500
            or len(settings['seeds']) > 5 or settings['hidden_dim'] > 64
            or settings['components'] > 16 or len(settings['scenarios']) > 4 or settings['evaluation_count'] > 4096):
        raise ValueError('Beyond small CPU diagnostic budget; use Colab')
    if any(settings[k] <= 0 for k in ('train_count', 'validation_count', 'evaluation_count', 'steps',
                                      'hidden_dim', 'components', 'learning_rate', 'rate_init')):
        raise ValueError('Positive sizes and optimization settings required')
    if not 0 < settings['train_time_max'] < settings['tail_time_max']:
        raise ValueError('Tail support must follow training support')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result = {'protocol_id': protocol['protocol_id'], 'settings': settings,
              'scope': 'oracle probability approximation only; not graph or phase recovery',
              'publication_eligible': False, 'runs': [],
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__}}
    for seed in settings['seeds']:
        rng = np.random.RandomState(seed)
        draws = {}
        for split, count, lower, upper in [('train', settings['train_count'], 0., settings['train_time_max']),
                ('validation', settings['validation_count'], 0., settings['train_time_max']),
                ('test', settings['evaluation_count'], 0., settings['train_time_max']),
                ('tail', settings['evaluation_count'], settings['train_time_max'], settings['tail_time_max'])]:
            draws[split] = {'context': rng.uniform(-1., 1., (count, 1)).astype(np.float32),
                            'elapsed': rng.uniform(lower, upper, count).astype(np.float32)}
        for scenario in settings['scenarios']:
            folder = output / f'{scenario}_seed{seed}'; folder.mkdir()
            parts = {}
            for split, draw in draws.items():
                base, truth = oracle_logits(draw['context'].astype(np.float64), draw['elapsed'].astype(np.float64), scenario)
                parts[split] = {**draw, 'base': base, 'truth': truth}
                np.savez_compressed(folder / f'{split}_data.npz', **parts[split])
            entry = {'seed': seed, 'scenario': scenario, 'models': {}}
            entry['models']['base'] = {'parameter_count': 0, 'endpoints': {}}
            for split in ('validation', 'test', 'tail'):
                part = parts[split]
                np.savez_compressed(folder / f'base_{split}.npz', logits=part['base'])
                entry['models']['base']['endpoints'][split] = oracle_metrics(part['base'], part['truth'])
            for kind in settings['models']:
                entry['models'][kind] = train_response(kind, seed, parts, settings, folder)
            result['runs'].append(entry)
            save_json(output / 'partial_results.json', result)
            print(f'Completed oracle diagnostic: {scenario} seed={seed}', flush=True)
    result['aggregate'] = {}
    for scenario in settings['scenarios']:
        result['aggregate'][scenario] = {}
        for kind in ['base', *settings['models']]:
            values = [r['models'][kind]['endpoints']['test']['excess_expected_bce']
                      for r in result['runs'] if r['scenario'] == scenario]
            result['aggregate'][scenario][kind] = {'excess_expected_bce_mean': float(np.mean(values)),
                'std_ddof1': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    save_json(output / 'results.json', result)
    return result
