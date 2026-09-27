"""Checks for streaming reference moments and the image preparation boundary."""
import json
import pickle
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gaussian_w2.imagenet import adapters
from gaussian_w2.imagenet.extractors import imagenet_extractor_signature
from gaussian_w2.imagenet.moments import MomentState, TorchChanMoments
from gaussian_w2.imagenet.images import ImageNetImageDataset, center_crop_dhariwal

ROOT = Path(__file__).resolve().parents[1]


def moments_of(values):
    mean = values.mean(0)
    return MomentState(values.shape[1], count=len(values), mean=mean,
                       m2=(values - mean).T @ (values - mean))


def test_reference_moments_covariance_matches_numpy():
    values = np.random.default_rng(19).normal(size=(17, 4))
    result = moments_of(values)
    np.testing.assert_array_equal(result.mean, values.mean(0))
    np.testing.assert_allclose(result.covariance(), np.cov(values, rowvar=False))


def test_torch_streaming_moments_match_direct_numpy():
    pytest.importorskip('torch')
    values = np.random.default_rng(23).normal(size=(29, 4)) + 10000
    accumulator = TorchChanMoments(4, device='cpu')
    for batch in np.array_split(values, [13, 19, 24]):
        accumulator.update(batch)
    result = accumulator.to_numpy()
    assert result.count == len(values)
    np.testing.assert_allclose(result.mean, values.mean(0), rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.covariance(), np.cov(values, rowvar=False), rtol=1e-10, atol=1e-10)


def test_paper_feature_signature_is_unchanged():
    inputs = json.loads((ROOT / 'configs/paper/imagenet/inputs.json').read_text())
    assert imagenet_extractor_signature() == inputs['extractor_signature']
    with pytest.raises(ValueError, match='paper uses'):
        imagenet_extractor_signature(dino_resize_mode='pil')


def test_pinned_dino_metadata_matches_paper_inputs():
    from gaussian_w2.imagenet.extractors import DINOV2_REPOSITORY, ImageNetFeatureExtractor

    preparation = json.loads((ROOT / 'configs/paper/imagenet/preparation.json').read_text())
    models = json.loads((ROOT / 'configs/paper/imagenet/models.json').read_text())
    inputs = json.loads((ROOT / 'configs/paper/imagenet/inputs.json').read_text())
    assert preparation['dino_repository'] == DINOV2_REPOSITORY
    assert models['automatic_detector_downloads']['dinov2']['torch_hub_repository'] == DINOV2_REPOSITORY

    # Metadata can be checked without loading the image models.
    extractor = ImageNetFeatureExtractor.__new__(ImageNetFeatureExtractor)
    extractor._edm2 = extractor._clip = SimpleNamespace(metadata={})
    extractor._torch = SimpleNamespace(__version__='test')
    extractor.device = 'cpu'
    extractor.dino_resize_mode = preparation['dino_resize_mode']
    extractor.dino_repository = preparation['dino_repository']
    extractor.dino_source = preparation['dino_source']
    assert extractor.metadata['extractor_signature'] == inputs['extractor_signature']

    extractor.dino_repository = '/custom/dinov2'
    extractor.dino_source = 'local'
    signature = extractor.metadata['extractor_signature']
    assert signature != inputs['extractor_signature']
    assert signature['fd_dinov2']['model_source'] == {
        'repository': '/custom/dinov2', 'source': 'local',
    }


def test_directory_and_zip_preserve_order_crop_and_worker_reopen(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    source = tmp_path / 'images'
    (source / 'class_b').mkdir(parents=True)
    (source / 'class_a').mkdir()
    a = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(16, 12, 3)
    Image.fromarray(a).save(source / 'class_b' / 'b.png')
    Image.fromarray(a[::-1]).save(source / 'class_a' / 'a.png')
    archive_path = tmp_path / 'images.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for p in reversed(sorted(source.rglob('*.png'))):
            archive.write(p, arcname=p.relative_to(source).as_posix())
    directory = ImageNetImageDataset(source, resolution=4, source_format='raw')
    zipped = ImageNetImageDataset(archive_path, resolution=4, source_format='raw')
    assert directory.names == zipped.names == ['class_a/a.png', 'class_b/b.png']
    for i in range(2):
        np.testing.assert_array_equal(directory[i], zipped[i])
        assert zipped[i].shape == (3, 4, 4) and zipped[i].dtype == np.uint8
    # DataLoader workers must reopen ZIP handles after pickling.
    restored = pickle.loads(pickle.dumps(zipped))
    np.testing.assert_array_equal(restored[0], directory[0])
    directory.close(); zipped.close(); restored.close()
    # Cropping an already-sized RGB array is the identity, including orientation.
    square = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    np.testing.assert_array_equal(center_crop_dhariwal(square, 4), square)
    with pytest.raises(ValueError, match='preprocessed image'):
        ImageNetImageDataset(source, resolution=4, source_format='edm2')[0]


def test_all_seven_paper_generators_keep_configured_options(monkeypatch, tmp_path):
    preparation = json.loads((ROOT / 'configs/paper/imagenet/preparation.json').read_text())
    assert len(preparation['pools']) == 7
    for pool in preparation['pools']:
        options = dict(pool['adapter_options'])
        options['repository_root'] = str(tmp_path)
        monkeypatch.setattr(adapters, '_repository_revision', lambda path: options['repository_revision'])
        model = adapters.create_imagenet_generator(pool['adapter'], options=options)
        assert not model._loaded
        assert model.metadata['configuration'] == options
        if pool['adapter'] == 'var':
            assert model.resolution == pool['resolution']
            assert model.cfg == options['cfg']
            assert model.autocast_dtype == options['autocast_dtype']
        if pool['adapter'] == 'edm2':
            assert model.sampler_kwargs == options['sampler_kwargs']
            assert model.use_fp16 is True and model.force_fp32 is False
        if pool['adapter'] == 'stylegan_xl':
            assert model.noise_mode == 'const' and model.truncation_psi == 1
    monkeypatch.setattr(adapters, '_repository_revision', lambda path: '0' * 40)
    with pytest.raises(RuntimeError, match='expected'):
        adapters.create_imagenet_generator(pool['adapter'], options=options)


def test_generator_import_collision_is_rejected(monkeypatch, tmp_path):
    module = ModuleType('models')
    module.__file__ = str(tmp_path / 'elsewhere' / 'models.py')
    monkeypatch.setitem(sys.modules, 'models', module)
    (tmp_path / 'allowed').mkdir()
    with pytest.raises(RuntimeError, match='refusing to mix'):
        adapters._activate_external_repo(tmp_path / 'allowed', ('models',))


def test_var_global_reset_mutations_are_restored_after_failure():
    names = ('Linear', 'LayerNorm', 'BatchNorm2d', 'SyncBatchNorm', 'Conv1d',
             'Conv2d', 'ConvTranspose1d', 'ConvTranspose2d')
    original = lambda self: None
    classes = {name: type(name, (), {'reset_parameters': original}) for name in names}
    del classes['SyncBatchNorm'].reset_parameters
    torch = SimpleNamespace(nn=SimpleNamespace(**classes))
    with pytest.raises(RuntimeError, match='builder failed'):
        with adapters._restored_var_reset_parameters(torch):
            for cls in classes.values():
                cls.reset_parameters = lambda self: 1
            raise RuntimeError('builder failed')
    assert 'reset_parameters' not in classes['SyncBatchNorm'].__dict__
    assert all(cls.reset_parameters is original for name, cls in classes.items() if name != 'SyncBatchNorm')


@pytest.fixture
def feature_lanes(tmp_path):
    from gaussian_w2.imagenet.io import atomic_json

    pool = {'pool_id': 'generator', 'lane_ranges': [[0, 3], [3, 6]],
            'seed_start': 100, 'adapter_options': {'batch_size': 2}}
    config = {'output': str(tmp_path), 'pools': [pool], 'samples_per_pool': 6,
              'feature_dimensions': {'fid': 2}, 'merge_chunk_rows': 2}
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    folders = []
    for index, (start, stop) in enumerate(pool['lane_ranges']):
        folder = tmp_path/'partials'/'generator'/f'lane{index:03d}'
        folder.mkdir(parents=True)
        np.save(folder/'fid.npy', values[start:stop])
        atomic_json(folder/'complete.json', {
            'binding': {'config': config, 'pool_id': 'generator', 'lane_index': index,
                        'row_range': [start, stop], 'seed_range': [100+start, 100+stop],
                        'batch_size': 2},
            'extractor': {'extractor_signature': {'fid': 'test-extractor'}},
        })
        folders.append(folder)
    return config, folders, values


def test_feature_merge_preserves_rows_and_refuses_completed_pool(feature_lanes):
    from gaussian_w2.imagenet.io import read_json
    from gaussian_w2.imagenet.prepare import merge

    config, folders, values = feature_lanes
    merge(config, 'generator')
    output = Path(config['output'])/'features'/'generator'
    np.testing.assert_array_equal(np.load(output/'fid.npy'), values)
    metadata = read_json(output/'fid.metadata.json')
    assert metadata['config'] == config and metadata['shape'] == [6, 2]
    assert metadata['extractor_signature'] == {'fid': 'test-extractor'}
    with pytest.raises(FileExistsError, match='already complete'):
        merge(config, 'generator')


@pytest.mark.parametrize('invalid', ['shape', 'nonfinite', 'coverage', 'extractor', 'seed'])
def test_feature_merge_keeps_structure_and_output_checks(feature_lanes, invalid):
    from gaussian_w2.imagenet.io import atomic_json, read_json
    from gaussian_w2.imagenet.prepare import merge

    config, folders, values = feature_lanes
    if invalid == 'shape':
        np.save(folders[0]/'fid.npy', values[:3, :1])
        message = 'shape/dtype'
    elif invalid == 'nonfinite':
        values[0, 0] = np.nan
        np.save(folders[0]/'fid.npy', values[:3])
        message = 'nonfinite'
    elif invalid == 'coverage':
        config['pools'][0]['lane_ranges'][1][0] = 2
        message = 'cover every row exactly once'
    elif invalid == 'extractor':
        record = read_json(folders[1]/'complete.json')
        record['extractor']['extractor_signature'] = {'fid': 'different-extractor'}
        atomic_json(folders[1]/'complete.json', record)
        message = 'different extractor signatures'
    else:
        record = read_json(folders[1]/'complete.json')
        record['binding']['seed_range'] = [100, 103]
        atomic_json(folders[1]/'complete.json', record)
        message = 'another generation configuration'
    with pytest.raises(ValueError, match=message):
        merge(config, 'generator')


def test_incomplete_merge_restarts_from_first_row(feature_lanes):
    from gaussian_w2.imagenet.prepare import merge

    config, folders, values = feature_lanes
    broken = values[3:].copy()
    broken[-1, 0] = np.nan
    np.save(folders[1]/'fid.npy', broken)
    with pytest.raises(ValueError, match='nonfinite'):
        merge(config, 'generator')
    destination = Path(config['output'])/'features'/'generator'
    assert not (destination/'complete.json').exists()
    # A whole-task rerun rewrites earlier rows too, rather than restoring a cursor.
    values[0, 0] = 99
    np.save(folders[0]/'fid.npy', values[:3])
    np.save(folders[1]/'fid.npy', values[3:])
    merge(config, 'generator')
    np.testing.assert_array_equal(np.load(destination/'fid.npy'), values)
