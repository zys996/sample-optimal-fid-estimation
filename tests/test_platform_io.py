"""Portable text files and byte-preserving submission exports."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def small_bundle(root):
    data = root / 'reproduce' / 'data'
    data.mkdir(parents=True)
    content = '{\n  "label": "ℓ 中文"\n}\n'.encode('utf-8')
    (data / 'example.json').write_bytes(content)
    manifest = {'description': '归档', 'datasets': {'example': {
        'file': 'data/example.json',
    }}}
    (data.parent / 'sources.json').write_text(
        json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    shutil.copyfile(ROOT / '.gitattributes', root / '.gitattributes')
    return data, content


def test_export_uses_only_stdlib_and_preserves_utf8_bytes(tmp_path):
    data, content = small_bundle(tmp_path)
    output = tmp_path / 'submission.zip'
    # -S excludes site-packages; EncodingWarning catches implicit text codecs.
    script = (
        'import sys\nfrom pathlib import Path\n'
        'from tools import make_submission_zip as export\n'
        'export.ROOT = Path(sys.argv[1])\n'
        'sys.argv = ["export", "--output", sys.argv[2]]\n'
        'export.main()\n'
    )
    result = subprocess.run(
        [sys.executable, '-S', '-X', 'warn_default_encoding', '-W',
         'error::EncodingWarning', '-c', script, str(tmp_path), str(output)],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
    )
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(output) as archive:
        prefix = 'FD-estimation/'
        assert archive.read(prefix + 'reproduce/data/example.json') == content
        assert archive.read(prefix + '.gitattributes') == (ROOT / '.gitattributes').read_bytes()


@pytest.mark.skipif(shutil.which('git') is None, reason='Git checkout test requires git')
def test_bundled_json_stays_lf_with_windows_autocrlf(tmp_path):
    data, content = small_bundle(tmp_path)
    for args in [('init', '-q'), ('config', 'core.autocrlf', 'true'),
                 ('add', '.gitattributes', 'reproduce/data/example.json')]:
        subprocess.run(['git', *args], cwd=tmp_path, check=True,
                       capture_output=True, text=True, encoding='utf-8')
    (data / 'example.json').unlink()
    subprocess.run(['git', 'checkout-index', '--all', '--force'], cwd=tmp_path,
                   check=True, capture_output=True, text=True, encoding='utf-8')
    assert (data / 'example.json').read_bytes() == content
