"""Collect completed Gaussian task results without running any experiment."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

from .config import tasks
from .records import atomic_json, load_task, read_manifest


def collect(output):
    output = Path(output)
    config = read_manifest(output)['config']
    expected = {f'task_{index:05d}.json': (index, task)
                for index, task in enumerate(tasks(config))}
    paths = sorted((output/'records').glob('*.json'))
    rows = []
    for path in paths:
        if path.name not in expected:
            raise ValueError(f'task result is outside the configured design: {path}')
        index, task = expected[path.name]
        rows.extend(load_task(path, config, task, index))
    if not rows:
        raise ValueError('no completed Gaussian task results to collect')
    fd, temporary = tempfile.mkstemp(prefix='.raw-', suffix='.csv', dir=output)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output/'raw.csv')
    finally:
        Path(temporary).unlink(missing_ok=True)
    summary = {'tasks': len(paths), 'expected_tasks': len(expected),
               'rows': len(rows), 'ok': sum(r['status'] == 'ok' for r in rows),
               'failed': sum(r['status'] != 'ok' for r in rows), 'output': str(output)}
    atomic_json(output/'run_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Gaussian output directory')
    args = parser.parse_args()
    print(json.dumps(collect(args.run), indent=2))


if __name__ == '__main__':
    main()
