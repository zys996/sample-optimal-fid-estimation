"""Importing an estimator should not require CUDA or image-generation packages."""
import os
from pathlib import Path
import subprocess
import sys


def test_estimator_import_has_no_image_or_plotting_dependencies():
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class RejectHeavyDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'cupy', 'matplotlib'}:
            raise AssertionError('Unexpected import: ' + fullname)

sys.meta_path.insert(0, RejectHeavyDependencies())
import gaussian_w2.estimators
print(gaussian_w2.estimators.__file__)
"""
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, check=True)
    assert str(root / "src") in result.stdout
