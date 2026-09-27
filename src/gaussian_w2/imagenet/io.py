"""Small, explicit feature-file and provenance helpers."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(path):
    """JSON is the study contract; expand environment variables in strings."""
    def expand(value):
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        if isinstance(value, list):
            return [expand(x) for x in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value
    config = read_json(path)
    if 'inputs_file' in config:
        if 'cases' in config:
            raise ValueError('use either inputs_file or embedded cases')
        input_path = Path(path).parent / config.pop('inputs_file')
        inputs = read_json(input_path)
        if set(inputs) != {'extractor_signature', 'cases'}:
            raise ValueError('inputs_file must contain exactly extractor_signature and cases')
        config['cases'] = [{**case, 'extractor_signature': inputs['extractor_signature']} for case in inputs['cases']]
    if 'estimator_grid' in config:
        if 'estimators' in config:
            raise ValueError('use either estimator_grid or estimators')
        grid = config.pop('estimator_grid')
        bank = [{'configuration_id': name, 'kind': kind} for name, kind in grid['baselines'].items()]
        for sweep in grid['sweeps']:
            for order in sweep['orders']:
                for weighting in sweep['weightings']:
                    for schedule in sweep['schedules']:
                        for count in sweep['num_points']:
                            ell = order + 1 if count == 'order+1' else int(count)
                            family = 'ols' if weighting == 'ordinary_ols' else 'va'
                            bank.append({'configuration_id': f'{family}_p{order}_{schedule}_l{ell}', 'kind': 'extrapolation',
                                         'order': order, 'weighting': weighting, 'sample_schedule': schedule, 'num_points': ell})
        config['estimators'] = {embedding: bank for embedding in sorted({c['embedding'] for c in config['cases']})}
    return expand(config)


def binding(path):
    """Describe an input file without reading its contents."""
    path = Path(path).resolve()
    st = path.stat()
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def load_moments(path):
    """Read the documented NPZ contract, including original feature-stat files."""
    with np.load(path, allow_pickle=False) as z:
        mean, cov = np.asarray(z['mean'], dtype=np.float64), np.asarray(z['covariance'], dtype=np.float64)
        n = int(z['sample_count'])
        normalization = str(z['covariance_normalization'])
        metadata = json.loads(str(z['metadata_json']))
    if normalization != '1/(N-1)' or n < 2:
        raise ValueError('moments require sample covariance 1/(N-1) and at least two rows')
    if mean.ndim != 1 or cov.shape != (mean.size, mean.size) or not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError('invalid moment shape or nonfinite values')
    return mean, cov, n, metadata


def save_moments(path, mean, covariance, sample_count, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, covariance=covariance, sample_count=np.int64(sample_count),
             dimension=np.int64(len(mean)), covariance_normalization=np.asarray('1/(N-1)'),
             metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def open_pool(path, shape=None):
    x = np.load(path, mmap_mode='r', allow_pickle=False)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError('features must be a two-dimensional floating point NPY array')
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise ValueError(f'feature shape {x.shape} differs from declared shape {shape}')
    return x
