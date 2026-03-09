import os
import json
import copy
import sys
import importlib
import argparse
import subprocess
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, sha256, output_dir, num_views, gpu_id=None, blender_threads=0, compute_device_type='CUDA'):
    output_folder = os.path.join(output_dir, 'renders', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--compute_device_type', compute_device_type,
        '--save_mesh',
    ]
    if blender_threads > 0:
        args[1:1] = ['--threads', str(blender_threads)]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    call(args, stdout=DEVNULL, stderr=DEVNULL, env=env)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}


def _parse_gpu_ids(gpu_ids):
    if gpu_ids is None:
        return []
    return [int(g.strip()) for g in gpu_ids.split(',') if g.strip()]


def _remove_flag(argv, flag):
    return [arg for arg in argv if arg != flag]


def _effective_workers(max_workers, tasks_per_gpu):
    if tasks_per_gpu is None:
        return max_workers
    return tasks_per_gpu


def _launch_per_gpu(raw_args, gpu_ids):
    world_size = len(gpu_ids)
    base_args = _remove_flag(raw_args, '--launch_per_gpu')
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        child_args = [
            sys.executable,
            __file__,
            sys.argv[1],
            *base_args,
            '--rank', str(rank),
            '--world_size', str(world_size),
            '--gpu_id', str(gpu_id),
        ]
        print(f'[Launcher] Start rank={rank}, gpu={gpu_id}')
        processes.append(subprocess.Popen(child_args))

    exit_codes = [proc.wait() for proc in processes]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f'One or more GPU workers failed: {exit_codes}')


if __name__ == '__main__':
    raw_args = sys.argv[2:]
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--tasks_per_gpu', type=int, default=None,
                        help='Concurrent render tasks per GPU process (overrides --max_workers when set)')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='Single GPU id bound to this process via CUDA_VISIBLE_DEVICES')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU ids used by --launch_per_gpu, e.g. "0,1,2,3"')
    parser.add_argument('--launch_per_gpu', action='store_true',
                        help='Spawn one process per GPU id and shard work across ranks automatically')
    parser.add_argument('--blender_threads', type=int, default=0,
                        help='CPU threads per blender process (0 uses blender default)')
    parser.add_argument('--compute_device_type', type=str, default='CUDA',
                        help='Cycles backend for Blender: CUDA/OPTIX/HIP/METAL/ONEAPI')
    opt = parser.parse_args(raw_args)
    opt = edict(vars(opt))
    opt.max_workers = _effective_workers(opt.max_workers, opt.tasks_per_gpu)

    if opt.launch_per_gpu and opt.rank == 0 and opt.world_size == 1:
        gpu_ids = _parse_gpu_ids(opt.gpu_ids)
        if not gpu_ids:
            raise ValueError('--launch_per_gpu requires non-empty --gpu_ids, e.g. --gpu_ids 0,1,2,3')
        _launch_per_gpu(raw_args, gpu_ids)
        sys.exit(0)

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')
    print(f'Runtime config: max_workers={opt.max_workers}, gpu_id={opt.gpu_id}, world_size={opt.world_size}', flush=True)

    # process objects
    func = partial(
        _render,
        output_dir=opt.output_dir,
        num_views=opt.num_views,
        gpu_id=opt.gpu_id,
        blender_threads=opt.blender_threads,
        compute_device_type=opt.compute_device_type,
    )
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)