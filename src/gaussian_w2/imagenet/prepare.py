"""Prepare generated feature lanes, merge them, or stream real-image moments.

The evaluator never needs PyTorch or images. This separate entry point keeps
generator-specific dependencies and data preparation out of estimator jobs.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from .io import atomic_json, binding, load_config, read_json, save_moments


def _lane_identity(config, pool, lane_index):
    start, stop = pool['lane_ranges'][lane_index]
    return {'config': config, 'pool_id': pool['pool_id'], 'lane_index': lane_index,
            'row_range': [start, stop],
            'seed_range': [pool['seed_start'] + start, pool['seed_start'] + stop],
            'batch_size': pool['adapter_options']['batch_size']}


def _open_features(path, shape):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.shape != tuple(shape) or array.dtype != np.float32:
        raise ValueError('invalid feature array shape/dtype')
    return array


def generate(config, pool_id, lane_index):
    from .adapters import create_imagenet_generator
    from .extractors import ImageNetFeatureExtractor
    pool = next(p for p in config['pools'] if p['pool_id'] == pool_id)
    start, stop = pool['lane_ranges'][lane_index]
    root = Path(config['output'])/'partials'/pool_id/f'lane{lane_index:03d}'
    root.mkdir(parents=True, exist_ok=True)
    identity = _lane_identity(config, pool, lane_index)
    complete = root/'complete.json'
    if complete.exists():
        raise FileExistsError(f'feature lane is already complete: {root}')
    batch_size = identity['batch_size']
    arrays = {
        e: np.lib.format.open_memmap(root/f'{e}.partial.npy', mode='w+', dtype=np.float32,
                                     shape=(stop-start, dimension))
        for e, dimension in config['feature_dimensions'].items()
    }
    adapter = create_imagenet_generator(pool['adapter'], options=pool['adapter_options'], device=config['device'])
    adapter.load()
    extractor = ImageNetFeatureExtractor(config['edm2_repository'], device=config['device'],
                                        dino_resize_mode=config['dino_resize_mode'], dino_repository=config['dino_repository'], dino_source=config['dino_source'], clip_download_root=config['clip_download_root'])
    torch = extractor._torch
    for batch_index, cursor in enumerate(range(0, stop-start, batch_size)):
        end = min(cursor+batch_size, stop-start)
        global_indices = start+np.arange(cursor,end,dtype=np.int64)
        seeds = pool['seed_start']+global_indices
        labels = global_indices % config['num_classes']
        images = adapter.generate(seeds.tolist(), labels.tolist(), batch_seed=int(seeds[0]))
        if tuple(images.shape[-2:]) != (pool['resolution'], pool['resolution']):
            raise ValueError('generator output resolution differs from config')
        features = extractor.extract(images)
        for e in arrays:
            values = features[e].detach().to(device='cpu',dtype=torch.float32).numpy()
            if values.shape != (end-cursor, arrays[e].shape[1]):
                raise ValueError('generated feature batch has an unexpected shape')
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite generated features')
            arrays[e][cursor:end] = values
        del images, features
        if (batch_index + 1) % 25 == 0 or end == stop-start:
            print(f'{pool_id} lane={lane_index} rows={end}/{stop-start}', flush=True)
    for array in arrays.values():
        array.flush()
    arrays.clear()
    del array
    for e in config['feature_dimensions']:
        os.replace(root/f'{e}.partial.npy', root/f'{e}.npy')
    atomic_json(complete, {'binding':identity, 'adapter':adapter.metadata, 'extractor':extractor.metadata})


def merge(config, pool_id):
    pool = next(p for p in config['pools'] if p['pool_id'] == pool_id)
    ranges = pool['lane_ranges']
    if (ranges[0][0] != 0 or ranges[-1][1] != config['samples_per_pool']
            or any(start >= stop for start, stop in ranges)
            or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))):
        raise ValueError('lanes must cover every row exactly once')
    root = Path(config['output'])
    destination = root/'features'/pool_id
    complete = destination/'complete.json'
    if complete.exists():
        raise FileExistsError(f'merged feature pool is already complete: {destination}')
    sources = []
    for i in range(len(ranges)):
        folder = root/'partials'/pool_id/f'lane{i:03d}'
        record = read_json(folder/'complete.json')
        if record['binding'] != _lane_identity(config, pool, i):
            raise ValueError('lane belongs to another generation configuration')
        if sources and record['extractor']['extractor_signature'] != sources[0][1]['extractor']['extractor_signature']:
            raise ValueError('feature lanes have different extractor signatures')
        sources.append((folder, record))
    destination.mkdir(parents=True, exist_ok=True)
    for e, dimension in config['feature_dimensions'].items():
        output = destination/f'{e}.npy'
        temporary = output.with_suffix('.partial.npy')
        array = np.lib.format.open_memmap(temporary, mode='w+',dtype=np.float32,shape=(config['samples_per_pool'],dimension))
        for (folder, _), (start, stop) in zip(sources, ranges):
            shard = _open_features(folder/f'{e}.npy', (stop-start, dimension))
            for a in range(0,len(shard),config['merge_chunk_rows']):
                values = shard[a:a+config['merge_chunk_rows']]
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite feature shard')
                array[start+a:start+a+len(values)] = values
        array.flush()
        del array
        os.replace(temporary,output)
        atomic_json(destination/f'{e}.metadata.json', {'config':config, 'pool_id':pool_id,
                    'shape':[config['samples_per_pool'],dimension],
                    'extractor_signature':sources[0][1]['extractor']['extractor_signature'],
                    'seed_start':pool['seed_start'], 'row_order':'sample_major_class_minor',
                    'sources':[binding(folder/'complete.json') for folder,_ in sources]})

    atomic_json(complete, {'config':config, 'pool_id':pool_id})


def reference(config, resolution):
    root = Path(config['output'])/'references'
    complete = root/f'imagenet{resolution}.complete.json'
    if complete.exists():
        raise FileExistsError(f'reference is already complete: {complete}')
    from .images import ImageNetImageDataset
    from .extractors import ImageNetFeatureExtractor
    from .moments import TorchChanMoments
    from torch.utils.data import DataLoader
    dataset = ImageNetImageDataset(config['reference']['images'], resolution=resolution, source_format=config['reference']['source_format'])
    if len(dataset) != config['reference']['sample_count']:
        raise ValueError('real image count differs from configuration')
    extractor = ImageNetFeatureExtractor(config['edm2_repository'], device=config['device'],
                                        dino_resize_mode=config['dino_resize_mode'], dino_repository=config['dino_repository'], dino_source=config['dino_source'],clip_download_root=config['clip_download_root'])
    identity = {'config':config, 'dataset':dataset.metadata(), 'extractor_signature':extractor.metadata['extractor_signature']}
    root.mkdir(parents=True,exist_ok=True)
    moments = {e:TorchChanMoments(d,device=config['device']) for e,d in config['feature_dimensions'].items()}
    loader = DataLoader(dataset,batch_size=config['reference']['batch_size'],
                        shuffle=False,drop_last=False,num_workers=config['reference']['num_workers'])
    processed = 0
    for b, images in enumerate(loader):
        features = extractor.extract(images.to(config['device']))
        for e, accumulator in moments.items():
            accumulator.update(features[e])
        processed += len(images)
        if (b+1)%25==0 or processed==len(dataset):
            print(f'reference resolution={resolution} rows={processed}/{len(dataset)}',flush=True)
    for e,accumulator in moments.items():
        m = accumulator.to_numpy()
        save_moments(root/f'imagenet{resolution}.{e}.npz',m.mean,m.covariance(ddof=1),m.count,identity)
    dataset.close()
    atomic_json(complete, identity)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['generate','merge','reference'])
    p.add_argument('--config',required=True)
    p.add_argument('--pool')
    p.add_argument('--lane',type=int)
    p.add_argument('--resolution',type=int)
    args=p.parse_args()
    config=load_config(args.config)
    if args.command=='generate':
        if args.pool is None or args.lane is None:
            p.error('generate requires --pool and --lane')
        generate(config,args.pool,args.lane)
    elif args.command=='merge':
        if args.pool is None:
            p.error('merge requires --pool')
        merge(config,args.pool)
    else:
        if args.resolution is None:
            p.error('reference requires --resolution')
        reference(config,args.resolution)

if __name__=='__main__':
    main()
