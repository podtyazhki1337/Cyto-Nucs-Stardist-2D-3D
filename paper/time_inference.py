"""Inference time, GPU memory and parameters of CytoNucs and the baselines on one GPU used exclusively (the runtime
table of the paper).

Every method runs through the code that scored it, with the arguments of its scoring call after `--`:
  cytonucs          score_hierarchy.py (2D data sets of one file per image): --config CONFIG --weights WEIGHTS
  cytonucs-blocks   evaluate_blocks.py (spheroid blocks): --config CONFIG --weights WEIGHTS [--margin 192]
  stardist          stardist3d/run_files.py --score ... (two StarDist models, one per level)
  stardist-blocks   stardist3d/run_fullres.py --score ... (two StarDist-3D models)
  hsd               hydrastardist/run_hsd.py --score ...
  cellpose          cellpose/run_cellpose.py ... (cellpose env; the fine-tuned models and settings of training.json)
  cellpose-blocks   cellpose/run_cellpose3d.py ... (cellpose env, without --score)

  - inputs: the images of --split, or the windows of the spheroid blocks of --split (block and margin of context, as
    the scorers); read and normalised as the method's scorer does before timing, and held in memory; --limit N takes
    N images spread evenly over the sorted names;
  - timed: one input from memory to the instance labels of both levels at the model's resolution, CytoNucs with its
    links. The baselines predict the levels separately and have no links; the pairing by overlap that the scorers add
    for them (compute_nucleus_cell_assignments) is timed apart in the last pass, as link_seconds_per_input, where its
    module can be imported (not in the cellpose env);
  - the first --warmup inputs are predicted once untimed (graph building, cuDNN autotuning), then --passes passes over
    all inputs; seconds_per_input is the median over the passes of the mean seconds per input;
  - GPU memory: the framework's peak allocation during the passes (TensorFlow get_memory_info, PyTorch
    max_memory_allocated), and the peak memory used on the device from the start of the script (nvidia-smi every
    0.2 s: CUDA context and allocator pools included);
  - parameters of all networks of the method;
  - CytoNucs: the seconds per stage (stage_seconds_per_input): prompt refinement and assembly with the links, as
    cytonucs_inference.predict_instances_prompted records them, and the rest of the time (normalisation, the network's
    StarDist outputs and their peaks / NMS), median over the passes of the mean per input.
Run it with nothing else on the machine.

Usage:
  python paper/time_inference.py --method cytonucs --gpu 0 --output OUT.json [--limit 300] -- --config C --weights W
  python paper/time_inference.py --method hsd --gpu 0 --output OUT.json -- --data DIR --score --name NAME
"""
import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]                   # the repository root
HERE = ROOT / 'baselines'                                    # the baseline scripts
METHODS = ('cytonucs', 'cytonucs-blocks', 'stardist', 'stardist-blocks', 'hsd', 'cellpose', 'cellpose-blocks')


def split_argv(argv):
    """(own arguments, the scorer's arguments after '--')."""
    if '--' not in argv:
        return argv, []
    i = argv.index('--')
    return argv[:i], argv[i + 1:]


own, scorer_argv = split_argv(sys.argv[1:])
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--method', required=True, choices=METHODS)
parser.add_argument('--gpu', required=True, help='index of the GPU (nvidia-smi numbering)')
parser.add_argument('--output', required=True)
parser.add_argument('--split', default='test')
parser.add_argument('--limit', type=int, default=0, help='images spread evenly over the split (0: all)')
parser.add_argument('--warmup', type=int, default=2)
parser.add_argument('--passes', type=int, default=3)
args = parser.parse_args(own)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np


class DevicePeak:
    """Largest memory.used of the GPU in MiB, from one nvidia-smi process that samples every 0.2 s."""

    def __init__(self, gpu):
        self.peak = 0.0
        self.process = subprocess.Popen(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                         '-lms', '200'], stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.peak = max(self.peak, float(line))
            except ValueError:
                pass

    def stop(self):
        time.sleep(0.5)                                    # one more sample after the last prediction
        self.process.terminate()
        self.thread.join(timeout=5)
        return self.peak


def load_script(path, argv):
    """Import a scoring script with its own command line: its module-level setup reads it."""
    sys.argv = [str(path)] + list(argv)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spread(names, limit):
    """`limit` names spread evenly over the sorted list (all with limit 0)."""
    if not limit or limit >= len(names):
        return list(names)
    return [names[i] for i in np.unique(np.linspace(0, len(names) - 1, limit).round().astype(int))]


def tensorflow():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf


def overlap_links():
    """The pairing by overlap of separately predicted levels that the scorers use (None where it cannot be imported)."""
    import contextlib
    import io
    try:
        from cytonucs_losses import compute_nucleus_cell_assignments
    except ImportError:
        return None

    def link(pred):
        with contextlib.redirect_stdout(io.StringIO()):                 # it prints a summary per call
            return compute_nucleus_cell_assignments(pred['nucleus'], pred['cytoplasm'])[0]
    return link


def cytonucs(split, limit, argv, blocks=False):
    import contextlib
    import io
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--margin', type=int, default=192)
    s = p.parse_args(argv)
    tensorflow()
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_model import build_cytonucs_model
    from cytonucs_runtime import read_model_image

    raw = read_config(s.config, s.weights)                 # 'auto' settings: calibrated for these weights
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(s.weights)
    tmp = tempfile.mkdtemp()
    raw['output']['basedir'] = tmp
    path = Path(tmp) / 'config.json'
    path.write_text(json.dumps(raw))
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = CytoNucsConfig.from_json(str(path))
        model = build_cytonucs_model(cfg, name='timing', basedir=tmp)
        model.load_weights(str(s.weights))
    inference = raw['inference']
    settings = {k: inference[k] for k in ('prob_thresh_nucleus', 'prob_thresh_cytoplasm', 'nms_thresh')}
    dataset = Path(raw['data']['dataset_dir'])
    if not blocks:                                         # as score_hierarchy.score
        from cytonucs_hierarchy import predict_hierarchy, read_calibration
        calibration = read_calibration(s.weights)
        folder = dataset / split
        names = spread(sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff')), limit)
        inputs = [read_model_image(folder / 'images' / n, int(cfg.ndim), int(cfg.n_image_channels)) for n in names]

        def predict(image):
            with contextlib.redirect_stdout(io.StringIO()):
                pred = predict_hierarchy(model, cfg, image, calibration=calibration, thresholds=settings)
            return dict(pred['levels'], links=pred['parents'])
    else:                                                  # as evaluate_blocks.evaluate
        from cytonucs_inference import CytoNucsPredictor
        from cytonucs_trainer import DiceCallback
        from cytonucs_runtime import normalize_percentiles as normalize
        from evaluate_blocks import block_windows
        manifest = json.loads((dataset / 'split_manifest.json').read_text())
        factor = int(manifest.get('xy_downscale', 1))
        from score_native_resolution import native_dataset
        native = native_dataset(dataset, manifest) if factor > 1 else dataset
        predictor = CytoNucsPredictor(model=model, config=cfg, **settings)
        cfg.tta_flips = bool(getattr(cfg, 'tta_flips', False))
        runner = SimpleNamespace(model=predictor.model)
        channels = int(getattr(cfg, 'n_image_channels', cfg.n_channel_in))
        names, inputs, volumes = [], [], {}
        for name, block, (y0, y1, x0, x1) in block_windows(json.loads((native / 'split_manifest.json').read_text()), split,
                                                            s.margin, factor):
            if name not in volumes:
                volumes[name] = read_model_image(dataset / 'train_regions' / 'images' / (name + '.tiff'), 3, channels)
            image = volumes[name][:, y0 // factor:y1 // factor, x0 // factor:x1 // factor]
            inputs.append(normalize(image.astype(np.float32), 1, 99.8, axis=(0, 1, 2)).astype(np.float32))
            names.append(f'{name}/{block}')
        settings.update(margin=s.margin, xy_downscale=factor)

        def predict(image):
            with contextlib.redirect_stdout(io.StringIO()):
                nuclei, cells, links = DiceCallback._predict_full_volume(runner, predictor, image)
            return dict(nucleus=nuclei, cytoplasm=cells, links=links)
    return SimpleNamespace(names=names, inputs=inputs, predict=predict, parameters=int(model.keras_model.count_params()),
                           framework='tensorflow', settings=settings, link=None, model=model)


def stardist(split, limit, argv):
    m = load_script(HERE / 'stardist3d' / 'run_files.py', argv + ['--gpu', args.gpu])
    kind = m.StarDist3D if m.NDIM == 3 else m.StarDist2D
    levels = dict(level.split(':') for level in (m.args.inner, m.args.outer))
    models = {level: kind(None, name=name, basedir=m.args.basedir) for level, name in levels.items()}
    names = spread(m.names(split), limit)

    def predict(x):
        return {level: model.predict_instances(x, n_tiles=m.n_tiles(x.shape, x.ndim > m.NDIM), show_tile_progress=False)[0]
                for level, model in models.items()}
    return SimpleNamespace(names=names, inputs=[m.read_image(m.DATA / split / 'images' / n) for n in names], predict=predict,
                           parameters=sum(int(model.keras_model.count_params()) for model in models.values()),
                           framework='tensorflow', settings=dict(models=levels, thresholds={l: md.thresholds._asdict() for l, md in models.items()}),
                           link=overlap_links())


def stardist_blocks(split, limit, argv):
    m = load_script(HERE / 'stardist3d' / 'run_fullres.py', argv + ['--gpu', args.gpu])
    import tifffile
    levels = dict(nucleus=m.args.nucleus_model, cytoplasm=m.args.cytoplasm_model)
    models = {level: m.StarDist3D(None, name=name, basedir=m.args.basedir) for level, name in levels.items()}
    name, _ = m.volume()
    native, factor = m.downscaling()
    image = tifffile.imread(m.DATA / 'train_regions' / 'images' / name)
    names, inputs = [], []
    for _, block, (y0, y1, x0, x1) in m.block_windows(json.loads((native / 'split_manifest.json').read_text()), split,
                                                       m.args.margin, factor):
        inputs.append(m.normalize(image[:, y0 // factor:y1 // factor, x0 // factor:x1 // factor].astype(np.float32), 1, 99.8,
                                  axis=(0, 1, 2)).astype(np.float32))
        names.append(f'{Path(name).stem}/{block}')

    def predict(x):
        n_tiles = (1,) + tuple(int(np.ceil(s / p)) for s, p in zip(x.shape[1:3], m.PATCH[1:])) + (1,)
        return {level: model.predict_instances(x, n_tiles=n_tiles, show_tile_progress=False)[0] for level, model in models.items()}
    return SimpleNamespace(names=names, inputs=inputs, predict=predict,
                           parameters=sum(int(model.keras_model.count_params()) for model in models.values()),
                           framework='tensorflow', settings=dict(models=levels, margin=m.args.margin, xy_downscale=factor,
                                                                 thresholds={l: md.thresholds._asdict() for l, md in models.items()}),
                           link=overlap_links())


def hsd(split, limit, argv):
    m = load_script(HERE / 'hydrastardist' / 'run_hsd.py', argv + ['--gpu', args.gpu])
    model = m.StarDist2D(None, name=m.args.name, basedir=m.args.basedir)
    names = spread(m.names(split), limit)
    thresholds = dict(prob_thresh1=model.thresholds1['prob'], nms_thresh1=model.thresholds1['nms'],
                      prob_thresh2=model.thresholds2['prob'], nms_thresh2=model.thresholds2['nms'])

    def predict(x):                                        # as run_hsd.score: branch 1 learned the outer level
        first, second = model.predict_instances(x, n_tiles=m.n_tiles(x.shape, x.ndim > 2), show_tile_progress=False, **thresholds)
        return {m.args.outer: np.asarray(first[0]), m.args.inner: np.asarray(second[0])}
    return SimpleNamespace(names=names, inputs=[m.read_image(m.DATA / split / 'images' / n) for n in names], predict=predict,
                           parameters=int(model.keras_model.count_params()), framework='tensorflow',
                           settings=dict(model=m.args.name, **{k: float(v) for k, v in thresholds.items()}), link=overlap_links())


def cellpose(split, limit, argv):
    m = load_script(HERE / 'cellpose' / 'run_cellpose.py', argv + ['--gpu', args.gpu])
    import tifffile
    import torch
    from cellpose import models as cp_models
    info = json.loads((m.OUT / 'training.json').read_text())['levels']
    chosen = {level: (cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=str(spec['weights'] if Path(spec['weights']).is_absolute() else m.OUT / spec['weights'])), spec['diameter'],
                      spec['cellprob_threshold'], spec['flow_threshold']) for level, spec in info.items()}
    names = spread(m.names(split), limit)

    def predict(image):
        return {level: m.predict(model, image, level, diameter, cellprob, flow)
                for level, (model, diameter, cellprob, flow) in chosen.items()}
    return SimpleNamespace(names=names, inputs=[tifffile.imread(m.DATA / split / 'images' / n) for n in names], predict=predict,
                           parameters=sum(sum(p.numel() for p in model.net.parameters()) for model, *_ in chosen.values()),
                           framework='torch', link=None,
                           settings={level: dict(diameter=d, cellprob=c, flow=f) for level, (_, d, c, f) in chosen.items()})


def cellpose_blocks(split, limit, argv):
    m = load_script(HERE / 'cellpose' / 'run_cellpose3d.py', argv + ['--gpu', args.gpu])
    import torch
    from cellpose import models as cp_models
    info = json.loads((m.OUT / 'training.json').read_text())['levels']
    chosen = {level: (cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=str(spec['weights'] if Path(spec['weights']).is_absolute() else m.OUT / spec['weights'])), spec['diameter'],
                      (spec['mode'], spec['stitch_threshold'], spec['cellprob_threshold'])) for level, spec in info.items()}
    image = m.read_volume()
    windows = list(m.block_windows(m.manifest(), split, m.args.margin))

    def predict(x):                                        # as run_cellpose3d.main
        return {level: m.masks_of(model, m.flows(model, x, level, diameter, best[0]), *best, diameter)
                for level, (model, diameter, best) in chosen.items()}
    return SimpleNamespace(names=[f'{n}/{b}' for n, b, _ in windows], inputs=[image[:, y0:y1, x0:x1] for _, _, (y0, y1, x0, x1) in windows],
                           predict=predict, parameters=sum(sum(p.numel() for p in model.net.parameters()) for model, *_ in chosen.values()),
                           framework='torch', link=None,
                           settings=dict(margin=m.args.margin, anisotropy=m.args.anisotropy,
                                         **{level: dict(diameter=d, mode=b[0], stitch=b[1], cellprob=b[2]) for level, (_, d, b) in chosen.items()}))


def framework_peak(framework, reset=False):
    """Peak allocation in GB since the last reset (TensorFlow or PyTorch)."""
    if framework == 'tensorflow':
        import tensorflow as tf
        if reset:
            tf.config.experimental.reset_memory_stats('GPU:0')
            return None
        return tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2 ** 30
    import torch
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return None
    return torch.cuda.max_memory_allocated() / 2 ** 30


def main():
    device = DevicePeak(args.gpu)
    started = time.time()
    loaders = {'cytonucs': cytonucs, 'cytonucs-blocks': lambda s, l, a: cytonucs(s, l, a, blocks=True), 'stardist': stardist,
               'stardist-blocks': stardist_blocks, 'hsd': hsd, 'cellpose': cellpose, 'cellpose-blocks': cellpose_blocks}
    run = loaders[args.method](args.split, args.limit, scorer_argv)
    load_s = time.time() - started
    print(f'{args.method}: {len(run.inputs)} inputs, {run.parameters} parameters, loaded in {load_s:.0f} s', flush=True)
    for x in run.inputs[:args.warmup]:
        run.predict(x)
    framework_peak(run.framework, reset=True)
    passes, link_s, objects, stages = [], [], {}, []
    profiled = getattr(run, 'model', None) is not None             # CytoNucs: stage seconds
    for k in range(args.passes):
        seconds, stage = [], []
        for i, x in enumerate(run.inputs):
            if profiled:
                run.model.stage_seconds = {}
            t0 = time.perf_counter()
            pred = run.predict(x)
            seconds.append(time.perf_counter() - t0)
            if profiled:
                parts = dict(run.model.stage_seconds)
                stage.append(dict(parts, detection_other=seconds[-1] - sum(parts.values())))
            if k == args.passes - 1:
                for level in ('nucleus', 'cytoplasm'):
                    objects[level] = objects.get(level, 0) + int(len(np.unique(np.asarray(pred[level]))) - 1)
                if run.link:
                    t0 = time.perf_counter()
                    run.link(pred)
                    link_s.append(time.perf_counter() - t0)
        passes.append(seconds)
        stages.append(stage)
        print(f'pass {k + 1}: {np.mean(seconds):.3f} s per input (median {np.median(seconds):.3f})', flush=True)
    result = dict(method=args.method, scorer_args=scorer_argv, split=args.split, inputs=len(run.inputs), names=run.names,
                  input_shapes=sorted({str(tuple(x.shape)) for x in run.inputs}), warmup=args.warmup, passes=args.passes,
                  seconds_per_input=statistics.median(float(np.mean(s)) for s in passes),
                  pass_means=[float(np.mean(s)) for s in passes], seconds=passes,
                  link_seconds_per_input=float(np.mean(link_s)) if link_s else None,
                  objects_last_pass=objects, parameters=int(run.parameters), framework=run.framework,
                  framework_peak_gb=framework_peak(run.framework), load_seconds=load_s, settings=run.settings)
    if profiled:
        names = sorted({key for stage in stages for parts in stage for key in parts})
        result['stage_seconds_per_input'] = {key: statistics.median(float(np.mean([p.get(key, 0.0) for p in stage]))
                                                                    for stage in stages) for key in names}
    result['device_peak_gb'] = device.stop() / 1024
    gpu = subprocess.run(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    result.update(gpu=gpu, host=platform.node(), cpus=len(os.sched_getaffinity(0)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1, default=str))
    print(f"{args.method}: {result['seconds_per_input']:.3f} s per input over {args.passes} passes "
          f"({', '.join(f'{v:.3f}' for v in result['pass_means'])}), framework peak {result['framework_peak_gb']:.2f} GB, "
          f"device peak {result['device_peak_gb']:.2f} GB, {result['parameters']} parameters -> {args.output}", flush=True)


if __name__ == '__main__':
    main()
