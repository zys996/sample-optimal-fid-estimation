"""Save complete task results and check their Gaussian design when collecting."""
from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import os
from pathlib import Path
import tempfile

import numpy as np

from .config import stable_seed


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


CASE_FIELDS = ('suite', 'family', 'case_label', 'dimension', 'instance_id')


def make_manifest(config, xp):
    environment = {'python': platform.python_version(), 'platform': platform.system(),
                   'machine': platform.machine(), 'numpy': np.__version__,
                   'scipy': importlib.metadata.version('scipy')}
    if config['backend'] == 'cupy':
        device = config['gpu']['device']
        name = xp.cuda.runtime.getDeviceProperties(device)['name']
        environment.update(cupy=xp.__version__, cuda_runtime=xp.cuda.runtime.runtimeGetVersion(),
                           cuda_driver=xp.cuda.runtime.driverGetVersion(),
                           gpu=name.decode('utf-8') if isinstance(name, bytes) else name)
    return {'schema': 'gaussian-run-v3', 'config': config,
            'numerical_environment': environment}


def read_manifest(output):
    path = output/'manifest.json'
    if not path.exists():
        raise ValueError('Gaussian run is missing its manifest; use a new output directory')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    config = json.loads((output/'config.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'gaussian-run-v3' or manifest.get('config') != config:
        raise ValueError('Gaussian manifest and frozen configuration disagree')
    return manifest


def load_task(path, config, task, task_index):
    """Check complete task/trial coverage and derive errors from saved estimates."""
    record = json.loads(path.read_text(encoding='utf-8'))
    covariance_seed = stable_seed(config['master_seed'], 'covariance', task['case_label'],
                                  task['dimension'], task['instance_id'])
    identity = {'schema': 'gaussian-task-v1', 'config': config, 'task': task,
                'task_index': task_index, 'covariance_seed': covariance_seed}
    if any(record.get(k) != v for k, v in identity.items()):
        raise ValueError(f'task identity mismatch: {path}')
    trials = record['trials']
    if (len(trials) != task['num_trials']
            or {trial['trial_id'] for trial in trials} != set(range(task['num_trials']))):
        raise ValueError(f'trial coverage mismatch: {path}')
    expected = {(n, m) for n in config['experiment']['sample_budgets']
                for m in config['experiment']['methods']}
    truth = None
    rows = []
    for trial in trials:
        sampling_seed = stable_seed(config['master_seed'], 'sample', task['case_label'],
                                    task['dimension'], task['instance_id'], trial['trial_id'])
        if trial['sampling_seed'] != sampling_seed:
            raise ValueError(f'trial sampling seed mismatch: {path}')
        trial_rows = trial['rows']
        if ({(row['N'], row['method']) for row in trial_rows} != expected
                or len(trial_rows) != len(expected)):
            raise ValueError(f'method/budget coverage mismatch: {path}')
        if truth is None:
            truth = trial_rows[0]['true_w2_sq']
            if truth is None or not math.isfinite(truth):
                raise ValueError(f'task has nonfinite population truth: {path}')
        row_identity = {k: task[k] for k in CASE_FIELDS}
        row_identity.update(trial_id=trial['trial_id'], sampling_seed=sampling_seed,
                            covariance_seed=covariance_seed, backend=config['backend'])
        for row in trial_rows:
            if any(row.get(k) != v for k, v in row_identity.items()) or row['true_w2_sq'] != truth:
                raise ValueError(f'row identity or truth mismatch: {path}')
            raw = row['estimate_w2_sq_raw']
            if row['status'] not in ('ok', 'numerical_failure', 'pilot_rank_failure', 'skipped_runtime_cap'):
                raise ValueError(f'unknown estimator status: {path}')
            if row['status'] == 'ok' and (raw is None or not math.isfinite(raw)):
                raise ValueError(f'successful estimate is nonfinite: {path}')
            error = raw - truth if row['status'] == 'ok' else None
            row['signed_error_w2_sq_raw'] = error
            row['abs_error_w2_sq_raw'] = abs(error) if error is not None else None
        rows.extend(trial_rows)
    return rows
