"""GPU job queue of the paper runs with memory budgets: several jobs share a GPU when their declared memory fits.

Every job script declares the GPU memory it needs in a line '# gpu_gb: N' (jobs without it take a whole GPU).
Slots (queue_slot.sh) ask this module for work; every GPU has a heavy and a light slot:
  - a job starts on a GPU when the memory reserved there plus its need stays within the budget (23 GB of 24);
  - heavy jobs (above 10 GB) run in the heavy slot, one at a time per GPU, in name order, the first that fits;
  - light jobs (up to 10 GB) run in the light slot next to the jobs already on the GPU (an empty GPU is left to
    the heavy slot while heavy jobs wait), and in the heavy slot when no heavy job can start;
  - a heavy job that has waited more than 6 h and fits nowhere makes one GPU empty for it ('drain'): nothing new
    starts there until it does, so the jobs sharing the GPUs cannot starve it;
  - a reservation names the slot's process; reservations of dead processes are dropped, so a killed slot frees its
    memory; reservations of jobs started outside the slots can be added with 'reserve'.
State: <queue>/state.json, guarded by an exclusive lock on <queue>/.lock.

Commands (used by queue_slot.sh):
  python paper_queue.py --queue DIR pick GPU PID KIND -> the job taken by a heavy / light slot (moved jobs/ ->
                                                        running/), or nothing
  python paper_queue.py --queue DIR release GPU PID   -> drops the reservation of that slot
  python paper_queue.py --queue DIR reserve GPU PID GB NAME   -> reservation for a job started elsewhere
  python paper_queue.py --queue DIR status            -> reservations, drains and waiting jobs with their needs
  python paper_queue.py --queue DIR left              -> number of jobs not yet started
"""
import argparse
import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

BUDGET_GB = 23.0
WHOLE_GPU_GB = 23.0
LIGHT_MAX_GB = 10.0
DRAIN_AFTER_H = 6.0


def need_gb(script):
    """Declared GPU memory of a job script ('# gpu_gb: N'); without it the whole GPU."""
    found = re.search(r'^#\s*gpu_gb:\s*([0-9.]+)', Path(script).read_text(), flags=re.M)
    return float(found.group(1)) if found else WHOLE_GPU_GB


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def choose(jobs, needs, reservations, drains, gpu, kind, waited=None, budget=BUDGET_GB, light_max=LIGHT_MAX_GB,
           drain_after_h=DRAIN_AFTER_H):
    """The job a slot of this GPU starts now (or None) and the drains after the decision.

    jobs: names in queue order; needs: name -> GB; reservations: list of dict(gpu, gb); drains: gpu -> the heavy
    job this GPU empties for; kind: 'heavy' or 'light' slot; waited: name -> hours the job has been waiting.
      - heavy slot (no heavy job on its GPU): the first heavy job (above light_max) that fits; a heavy job that has
        waited longer than drain_after_h and fits nowhere makes this GPU empty for it (drain); without a heavy job
        to start, a light job that fits;
      - light slot: a light job that fits, next to the jobs already on the GPU; it leaves an empty GPU, or a
        heavy job that fits while the heavy slot is idle, to the heavy slot, and starts nothing on a GPU that
        empties for a heavy job. Reservations made by a slot name it (slot: heavy / light)."""
    waited = waited or {}
    drains = {g: j for g, j in drains.items() if j in jobs}
    on_gpu = [r for r in reservations if r['gpu'] == gpu]
    free = budget - sum(r['gb'] for r in on_gpu)
    heavy_waiting = [j for j in jobs if needs[j] > light_max]
    if gpu in drains:                                       # this GPU empties for one heavy job
        job = drains[gpu]
        if kind == 'heavy' and needs[job] <= free:
            drains.pop(gpu)
            return job, drains
        return None, drains
    if kind == 'heavy' and not any(r['gb'] > light_max for r in on_gpu):
        for job in heavy_waiting:
            if needs[job] <= free:
                return job, {g: j for g, j in drains.items() if j != job}
            if waited.get(job, 0.0) > drain_after_h and job not in drains.values():
                drains[gpu] = job
                return None, drains
    if kind == 'light' and heavy_waiting:
        if not on_gpu:
            return None, drains                             # an empty GPU is for the heavy slot
        if not any(r.get('slot') == 'heavy' for r in on_gpu) and any(needs[j] <= free for j in heavy_waiting):
            return None, drains                             # the idle heavy slot takes the heavy job that fits
    for job in jobs:
        if needs[job] <= light_max and needs[job] <= free:
            return job, drains
    return None, drains


class Queue:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs_dir, self.running_dir = self.root / 'jobs', self.root / 'running'
        self.state_path = self.root / 'state.json'

    @contextmanager
    def locked(self):
        with open(self.root / '.lock', 'w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {}
            state.setdefault('reservations', [])
            state.setdefault('drains', {})
            state['reservations'] = [r for r in state['reservations'] if alive(r['pid'])]
            yield state
            tmp = self.state_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(state, indent=1))
            os.replace(tmp, self.state_path)

    def waiting(self):
        return sorted(p.name for p in self.jobs_dir.iterdir() if p.suffix == '.sh') if self.jobs_dir.is_dir() else []

    def pick(self, gpu, pid, kind):
        with self.locked() as state:
            jobs = self.waiting()
            needs = {job: need_gb(self.jobs_dir / job) for job in jobs}
            now = time.time()
            state.setdefault('seen', {})
            state['seen'] = {job: state['seen'].get(job, now) for job in jobs}       # first time seen waiting
            waited = {job: (now - t) / 3600 for job, t in state['seen'].items()}
            job, state['drains'] = choose(jobs, needs, state['reservations'], state['drains'], str(gpu), kind, waited)
            if job is None:
                return None
            self.running_dir.mkdir(exist_ok=True)
            os.replace(self.jobs_dir / job, self.running_dir / job)
            state['reservations'].append(dict(gpu=str(gpu), pid=int(pid), gb=needs[job], job=job, slot=kind))
            return job

    def release(self, gpu, pid):
        with self.locked() as state:
            state['reservations'] = [r for r in state['reservations'] if not (r['gpu'] == str(gpu) and r['pid'] == int(pid))]

    def reserve(self, gpu, pid, gb, name):
        with self.locked() as state:
            state['reservations'].append(dict(gpu=str(gpu), pid=int(pid), gb=float(gb), job=name))

    def status(self):
        with self.locked() as state:
            lines = [f"GPU {r['gpu']}: {r['gb']:g} GB {r['job']} (pid {r['pid']})" for r in sorted(state['reservations'], key=lambda r: r['gpu'])]
            lines += [f'GPU {gpu} holds for {job}' for gpu, job in sorted(state['drains'].items())]
            lines += [f'waiting: {job} ({need_gb(self.jobs_dir / job):g} GB)' for job in self.waiting()]
            return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', default='/storage02/miroslavm/cytonucs_paper_runs/queue')
    parser.add_argument('command', choices=('pick', 'release', 'reserve', 'status', 'left'))
    parser.add_argument('values', nargs='*')
    args = parser.parse_args()
    queue = Queue(args.queue)
    if args.command == 'pick':
        job = queue.pick(*args.values[:3])
        print(job or '')
    elif args.command == 'release':
        queue.release(*args.values[:2])
    elif args.command == 'reserve':
        queue.reserve(*args.values[:4])
    elif args.command == 'status':
        print(queue.status())
    else:
        print(len(queue.waiting()))


if __name__ == '__main__':
    main()
