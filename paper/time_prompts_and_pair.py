"""Per-image inference time of a joint CytoNucs model against its number of prompts, and the time of two separate
single-level networks plus their pairing. Protocol of the runtime table (paper/time_inference.py, whose loader and
helpers are used here): one GPU without other jobs, the model and the images loaded first, --warmup images predicted
untimed, then --passes passes over all images; per image the median over the passes.

  f      the joint model at its own settings. Per image: total seconds; the prompted mask stage ('refinement') and the
         assembly with the links ('assembly_linking') as cytonucs_inference records them in model.stage_seconds; the
         rest ('detection': normalisation, the network's StarDist outputs, peaks / NMS); and the prompts of the
         refinement stage per head ('attempted': mask candidates tried per head, after the prob threshold; a nucleus
         prompt tries both heads, a cell prompt the cell head only, so prompts_nucleus = nucleus attempts,
         prompts_cell = cell - nucleus attempts, prompts_total = cell attempts; 'prompts_skipped' as covered;
         'residual_prompts' of the second pass), summed over the refinement calls of the image. Summary: the linear
         fit of seconds against prompts.
  pair   the nucleus-only network (--config / --weights) and the cells-only network (--config2 / --weights2) on the same
         images. Per image: the seconds of each network, and the pairing by overlap of their outputs
         (compute_nucleus_cell_assignments, as the scorers pair separately predicted levels), each the median over the
         passes; their sum is the two-network time.
The machine state is recorded: the other compute processes on all GPUs and the load average, at the start and the end.
Usage (from the repository root):
  python paper/time_prompts_and_pair.py f    --gpu G --split test [--limit N] --output OUT.json --config C --weights W
  python paper/time_prompts_and_pair.py pair --gpu G --split test [--limit N] --output OUT.json --config C --weights W \
      --config2 C2 --weights2 W2
"""
import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('mode', choices=('f', 'pair'))
p.add_argument('--gpu', required=True)
p.add_argument('--split', default='test')
p.add_argument('--limit', type=int, default=0)
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--passes', type=int, default=3)
p.add_argument('--output', required=True)
p.add_argument('--config', required=True)
p.add_argument('--weights', required=True)
p.add_argument('--config2')
p.add_argument('--weights2')
args = p.parse_args()


def machine():
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip().splitlines()
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return dict(time=time.strftime('%F %T'), load_average=os.getloadavg(), cpus=os.cpu_count(), gpus=gpus,
                compute_apps=[a for a in apps if str(os.getpid()) not in a])


state_start = machine()
# time_inference.py reads its own command line when imported; give it the one of this run.
TI = Path(__file__).resolve().parent / 'time_inference.py'
sys.argv = [str(TI), '--method', 'cytonucs', '--gpu', args.gpu, '--output', args.output, '--split', args.split,
            '--limit', str(args.limit)]
spec = importlib.util.spec_from_file_location('time_inference', TI)
ti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ti)
import numpy as np  # noqa: E402

device = ti.DevicePeak(args.gpu)


def accumulate(model):
    """Collect every model.last_refinement_diagnostics an image's refinement calls write (one or more per image)."""
    calls = []
    base = model.__class__

    class Recording(base):
        def __setattr__(self, key, value):
            if key == 'last_refinement_diagnostics' and value:
                calls.append(value)
            base.__setattr__(self, key, value)
    model.__class__ = Recording
    return calls


def prompts(calls):
    out = {}
    for head in ('nucleus', 'cytoplasm'):
        for key in ('attempted', 'prompts_skipped', 'residual_prompts', 'accepted'):
            out[f'{head}_{key}'] = int(sum(int(c.get(head, {}).get(key, 0)) for c in calls))
    out['refinement_calls'] = len(calls)
    # a nucleus prompt makes a nucleus and a cell candidate, a cell prompt a cell candidate only
    out['prompts_nucleus'] = out['nucleus_attempted']
    out['prompts_cell'] = out['cytoplasm_attempted'] - out['nucleus_attempted']
    out['prompts_total'] = out['cytoplasm_attempted']
    return out


def load(config, weights):
    return ti.cytonucs(args.split, args.limit, ['--config', config, '--weights', weights])


def mode_f():
    run = load(args.config, args.weights)
    calls = accumulate(run.model)
    for x in run.inputs[:args.warmup]:
        run.predict(x)
    per = [dict(name=n, shape=list(x.shape), total=[], refinement=[], assembly_linking=[], detection=[])
           for n, x in zip(run.names, run.inputs)]
    for k in range(args.passes):
        for i, x in enumerate(run.inputs):
            run.model.stage_seconds = {}
            calls.clear()
            t0 = time.perf_counter()
            pred = run.predict(x)
            total = time.perf_counter() - t0
            parts = dict(run.model.stage_seconds)
            r = per[i]
            r['total'].append(total)
            r['refinement'].append(parts.get('refinement', 0.0))
            r['assembly_linking'].append(parts.get('assembly_linking', 0.0))
            r['detection'].append(total - sum(parts.values()))
            if k == args.passes - 1:
                r.update(prompts(calls))
                r['objects'] = {lv: int(len(np.unique(np.asarray(pred[lv]))) - 1) for lv in ('nucleus', 'cytoplasm')}
        print(f'pass {k + 1}: {np.mean([r["total"][-1] for r in per]):.3f} s per image', flush=True)
    for r in per:
        for key in ('total', 'refinement', 'assembly_linking', 'detection'):
            r[f'{key}_seconds'] = statistics.median(r.pop(key))
    t = np.array([r['total_seconds'] for r in per])
    n = np.array([r['prompts_total'] for r in per], float)
    fit = np.polyfit(n, t, 1).tolist() if len(set(n)) > 1 else None
    summary = dict(images=len(per), seconds_per_image_mean=float(t.mean()), seconds_per_image_median=float(np.median(t)),
                   refinement_share=float(sum(r['refinement_seconds'] for r in per) / t.sum()),
                   prompts_mean=float(n.mean()), seconds_per_prompt_fit=fit, parameters=run.parameters,
                   settings=run.settings)
    return dict(summary=summary, images=per)


def mode_pair():
    nuc = load(args.config, args.weights)
    cell = load(args.config2, args.weights2)
    assert nuc.names == cell.names, 'the two networks must read the same images'
    link = ti.overlap_links()
    for x in nuc.inputs[:args.warmup]:
        nuc.predict(x)
    for x in cell.inputs[:args.warmup]:
        cell.predict(x)
    per = [dict(name=n, shape=list(x.shape), nucleus=[], cells=[], pairing=[]) for n, x in zip(nuc.names, nuc.inputs)]
    for k in range(args.passes):
        for i, (xn, xc) in enumerate(zip(nuc.inputs, cell.inputs)):
            t0 = time.perf_counter()
            pn = nuc.predict(xn)
            t1 = time.perf_counter()
            pc = cell.predict(xc)
            t2 = time.perf_counter()
            link({'nucleus': pn['nucleus'], 'cytoplasm': pc['cytoplasm']})
            t3 = time.perf_counter()
            r = per[i]
            r['nucleus'].append(t1 - t0)
            r['cells'].append(t2 - t1)
            r['pairing'].append(t3 - t2)
        print(f'pass {k + 1}: {np.mean([r["nucleus"][-1] + r["cells"][-1] + r["pairing"][-1] for r in per]):.3f} s per image',
              flush=True)
    for r in per:
        for key in ('nucleus', 'cells', 'pairing'):
            r[f'{key}_seconds'] = statistics.median(r.pop(key))
        r['two_networks_seconds'] = r['nucleus_seconds'] + r['cells_seconds']
        r['total_seconds'] = r['two_networks_seconds'] + r['pairing_seconds']
    mean = lambda key: float(np.mean([r[key] for r in per]))                                    # noqa: E731
    summary = dict(images=len(per), nucleus_network=mean('nucleus_seconds'), cells_network=mean('cells_seconds'),
                   two_networks=mean('two_networks_seconds'), pairing=mean('pairing_seconds'), total=mean('total_seconds'),
                   parameters=nuc.parameters + cell.parameters, settings=dict(nucleus=nuc.settings, cells=cell.settings))
    return dict(summary=summary, images=per)


result = mode_f() if args.mode == 'f' else mode_pair()
result.update(mode=args.mode, split=args.split, limit=args.limit, warmup=args.warmup, passes=args.passes,
              config=args.config, weights=args.weights, config2=args.config2, weights2=args.weights2,
              device_peak_gb=device.stop() / 1024, machine_start=state_start, machine_end=machine())
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=1, default=str))
print(json.dumps(result['summary'], default=str), '->', args.output, flush=True)
