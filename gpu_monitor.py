"""Samples one GPU (utilisation, memory, power) and the host memory of a process tree every --every seconds into
a CSV while a training or evaluation runs, for the resource tables of a paper.

Usage: python gpu_monitor.py --gpu 1 --output run_gpu.csv [--every 10] [--until-session TMUX_SESSION] [--tree PID]
Host memory: resident memory of --tree (default: the monitor's parent, e.g. the queue slot running the job) and
all its descendants except the monitor, in total (with memory-mapped files read) and anonymous.
Stops when the tmux session ends (if given) or on Ctrl-C. `summarise(csv)` returns mean / max utilisation,
peak memory, peak host memory and energy.
"""
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

FIELDS = ('utilization.gpu', 'memory.used', 'memory.total', 'power.draw')


def sample(gpu):
    out = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=' + ','.join(FIELDS), '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout.strip()
    return [float(v) if v.strip() not in ('[N/A]', '') else float('nan') for v in out.split(',')]


def tree_memory(root, skip=None):
    """Resident memory in GB of process `root` and its descendants (without `skip`): (total, anonymous)."""
    children = {}
    for name in os.listdir('/proc'):
        if name.isdigit():
            try:
                stat = Path(f'/proc/{name}/stat').read_text()
                children.setdefault(int(stat[stat.rindex(')') + 2:].split()[1]), []).append(int(name))
            except (OSError, ValueError, IndexError):
                pass
    total = anon = 0
    stack = [int(root)]
    while stack:
        pid = stack.pop()
        stack += children.get(pid, [])
        if pid == skip:
            continue
        try:
            fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        except OSError:
            continue
        total += int(fields.get('VmRSS', '0 kB').split()[0])
        anon += int(fields.get('RssAnon', '0 kB').split()[0])
    return round(total / 2 ** 20, 3), round(anon / 2 ** 20, 3)


def summarise(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    t = [float(r['time_s']) for r in rows]
    util = [float(r['utilization.gpu']) for r in rows]
    mem = [float(r['memory.used']) for r in rows]
    power = [float(r['power.draw']) for r in rows]
    energy = sum(p * (b - a) for p, a, b in zip(power, t, t[1:])) / 3.6e6            # W*s -> kWh
    out = dict(duration_h=(t[-1] - t[0]) / 3600, util_mean=sum(util) / len(util), util_max=max(util),
               mem_peak_gb=max(mem) / 1024, power_mean_w=sum(power) / len(power), energy_kwh=energy)
    if 'host_rss_gb' in rows[0]:
        out.update(host_rss_peak_gb=max(float(r['host_rss_gb']) for r in rows),
                   host_anon_peak_gb=max(float(r['host_anon_gb']) for r in rows))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--every', type=float, default=10.)
    parser.add_argument('--until-session')
    parser.add_argument('--tree', type=int, default=os.getppid(), help='root of the process tree whose host memory is sampled')
    args = parser.parse_args()
    start = time.time()
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('time_s',) + FIELDS + ('host_rss_gb', 'host_anon_gb'))
        while True:
            if args.until_session and subprocess.run(['tmux', 'has-session', '-t', args.until_session],
                                                     capture_output=True).returncode:
                break
            writer.writerow([round(time.time() - start, 1)] + sample(args.gpu) + list(tree_memory(args.tree, os.getpid())))
            f.flush()
            time.sleep(args.every)
    print(summarise(args.output))


if __name__ == '__main__':
    main()
