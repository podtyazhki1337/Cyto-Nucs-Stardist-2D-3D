"""Samples one GPU (utilisation, memory, power) every --every seconds into a CSV while a training or
evaluation runs, for the resource tables of a paper.

Usage: python gpu_monitor.py --gpu 1 --output run_gpu.csv [--every 10] [--until-session TMUX_SESSION]
Stops when the tmux session ends (if given) or on Ctrl-C. `summarise(csv)` returns mean / max utilisation,
peak memory and energy.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

FIELDS = ('utilization.gpu', 'memory.used', 'memory.total', 'power.draw')


def sample(gpu):
    out = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=' + ','.join(FIELDS), '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout.strip()
    return [float(v) if v.strip() not in ('[N/A]', '') else float('nan') for v in out.split(',')]


def summarise(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    t = [float(r['time_s']) for r in rows]
    util = [float(r['utilization.gpu']) for r in rows]
    mem = [float(r['memory.used']) for r in rows]
    power = [float(r['power.draw']) for r in rows]
    energy = sum(p * (b - a) for p, a, b in zip(power, t, t[1:])) / 3.6e6            # W*s -> kWh
    return dict(duration_h=(t[-1] - t[0]) / 3600, util_mean=sum(util) / len(util), util_max=max(util),
                mem_peak_gb=max(mem) / 1024, power_mean_w=sum(power) / len(power), energy_kwh=energy)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--every', type=float, default=10.)
    parser.add_argument('--until-session')
    args = parser.parse_args()
    start = time.time()
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('time_s',) + FIELDS)
        while True:
            if args.until_session and subprocess.run(['tmux', 'has-session', '-t', args.until_session],
                                                     capture_output=True).returncode:
                break
            writer.writerow([round(time.time() - start, 1)] + sample(args.gpu))
            f.flush()
            time.sleep(args.every)
    print(summarise(args.output))


if __name__ == '__main__':
    main()
