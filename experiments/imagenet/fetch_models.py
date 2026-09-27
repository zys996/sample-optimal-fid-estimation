#!/usr/bin/env python3
"""Fetch the externally hosted, author-provided models in the explicit registry.

Run intentionally: this downloads large model files. All upstream licenses
remain with their upstream checkouts and checkpoints.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
from gaussian_w2.imagenet.io import atomic_json, binding, load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--manifest',required=True,help='Where to record model sources and local file metadata')
    args=parser.parse_args()
    config=load_config(args.config)
    result={'repositories':[],'checkpoints':[]}
    for repo in config['repositories']:
        path=Path(repo['path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            subprocess.run(['git','clone',repo['url'],str(path)],check=True)
            subprocess.run(['git','-C',str(path),'checkout','--detach',repo['revision']],check=True)
        revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if revision!=repo['revision']:
            raise ValueError(f'checkout revision differs from registry: {path}')
        if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():
            raise ValueError(f'upstream checkout has local modifications: {path}')
        result['repositories'].append(repo)
    for checkpoint in config['checkpoints']:
        path=Path(checkpoint['path']);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial')
            try:
                if 'url' in checkpoint:
                    with urllib.request.urlopen(checkpoint['url']) as source,temporary.open('wb') as target:
                        shutil.copyfileobj(source,target)
                else:
                    from huggingface_hub import hf_hub_download
                    source=hf_hub_download(checkpoint['hf_repository'],checkpoint['hf_filename'],revision=checkpoint['hf_revision'])
                    shutil.copyfile(source,temporary)
                os.replace(temporary,path)
            finally:
                temporary.unlink(missing_ok=True)
        result['checkpoints'].append({**checkpoint,'downloaded_file':binding(path)})
    atomic_json(args.manifest,result)

if __name__=='__main__':
    main()
