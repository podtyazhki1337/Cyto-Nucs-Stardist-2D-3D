"""Recall of the whole chain of a three-level hierarchy (e.g. nucleolus -> nucleus -> cell, the synthetic data) from
saved predictions.

An annotated innermost object counts as a recovered chain when it, its parent and that parent's parent are all matched
one-to-one at IoU 0.5 (stardist matching, as the scorers) and both predicted links are right: the matched innermost
object is linked to the matched middle object and that one to the matched outermost object. Links are made from the
three label maps, with one or both rules:
  overlap  cytonucs_hierarchy.link_children (the largest overlap), as the hierarchy inference links them;
  scorer   compute_nucleus_cell_assignments as in score_predictions.py (overlap > 1 %, then the nearest centroid
           within 100 px), the rule for methods without links of their own.
The pair JTPRs computed the same way are reported next to the chain recall, as a check against the score files; also
the shares of the chain failures by the first missing piece (inner, middle, outer object, inner link, middle link).

Each method is a folder with one subfolder of predicted labels per level (<folder>/<level>/<image>), or, for one
network per level, the level folders given one by one: NAME=LEVEL:DIR,LEVEL:DIR,LEVEL:DIR.

Usage (from the repository root):
  python paper/chain_recall.py --data data/Cytonucs_Synthetic/hela_v1 --split test --output OUT \\
      --method "CytoNucs=RUN/predictions_test" \\
      --method "3 x StarDist-3D=nucleolus:A/nucleolus,nucleus:A/nucleus,cytoplasm:B/cytoplasm" [--links overlap scorer]
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repository root: the CytoNucs modules

FAILS = ('fail_inner', 'fail_middle', 'fail_outer', 'fail_link_inner', 'fail_link_middle')


def level_dirs(spec, levels):
    if ':' in spec.split(',')[0] and not Path(spec).is_dir():
        dirs = dict(part.split(':', 1) for part in spec.split(','))
        return {lv: Path(dirs[lv]) for lv in levels}
    return {lv: Path(spec) / lv for lv in levels}


def scorer_links(child, parent):
    from cytonucs_losses import compute_nucleus_cell_assignments
    with contextlib.redirect_stdout(io.StringIO()):
        links, _, _ = compute_nucleus_cell_assignments(child, parent, max_distance=100.0)
    return links


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--levels', nargs=3, default=['nucleolus', 'nucleus', 'cytoplasm'], help='inner to outer')
    parser.add_argument('--method', action='append', required=True, metavar='NAME=DIR')
    parser.add_argument('--links', nargs='+', choices=('overlap', 'scorer'), default=['overlap'])
    parser.add_argument('--output', required=True, help='output folder')
    args = parser.parse_args()
    import tifffile
    from cytonucs_hierarchy import link_children
    from cytonucs_inference import match_instances
    from score_hierarchy import ground_truth
    inner, middle, outer = levels = tuple(args.levels)
    linkers = {'overlap': link_children, 'scorer': scorer_links}
    folder = Path(args.data) / args.split
    names = sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    truth = {n: ground_truth(folder, n, list(levels), [(inner, middle), (middle, outer)])[:2] for n in names}
    rows = {}
    for spec in args.method:
        method, where = spec.split('=', 1)
        dirs = level_dirs(where, levels)
        for rule in args.links:
            c = dict(chains=0, recovered=0, pairs_inner=0, joint_inner=0, pairs_middle=0, joint_middle=0, **dict.fromkeys(FAILS, 0))
            for n in names:
                gt, truths = truth[n]
                pr = {lv: tifffile.imread(dirs[lv] / n) for lv in levels}
                m = {lv: match_instances(pr[lv], gt[lv]) for lv in levels}                     # gt id -> pred id
                link1 = {int(k): int(v) for k, v in linkers[rule](pr[inner], pr[middle]).items()}
                link2 = {int(k): int(v) for k, v in linkers[rule](pr[middle], pr[outer]).items()}
                present = {lv: set(int(v) for v in np.unique(gt[lv]) if v) for lv in levels}
                a = {k: v for k, v in truths[f'{inner}_{middle}'].items() if k in present[inner] and v in present[middle]}
                b = {k: v for k, v in truths[f'{middle}_{outer}'].items() if k in present[middle] and v in present[outer]}
                for k, v in a.items():                                                      # pair JTPRs as a check
                    c['pairs_inner'] += 1
                    c['joint_inner'] += int(k in m[inner] and v in m[middle] and link1.get(m[inner][k]) == m[middle][v])
                for k, v in b.items():
                    c['pairs_middle'] += 1
                    c['joint_middle'] += int(k in m[middle] and v in m[outer] and link2.get(m[middle][k]) == m[outer][v])
                for o1, o2 in a.items():
                    if o2 not in b:
                        continue                                                            # middle object without an annotated parent
                    c['chains'] += 1
                    p1, p2, p3 = m[inner].get(o1), m[middle].get(o2), m[outer].get(b[o2])
                    if p1 is None:
                        c['fail_inner'] += 1
                    elif p2 is None:
                        c['fail_middle'] += 1
                    elif p3 is None:
                        c['fail_outer'] += 1
                    elif link1.get(p1) != p2:
                        c['fail_link_inner'] += 1
                    elif link2.get(p2) != p3:
                        c['fail_link_middle'] += 1
                    else:
                        c['recovered'] += 1
            key = method if len(args.links) == 1 else f'{method}, {rule} links'
            rows[key] = c
            print(key, c, flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'chain_recall.json').write_text(json.dumps(dict(data=str(folder), levels=levels, methods=rows), indent=1))
    lines = [f'| method | chains | chain recall | JTPR {inner}-{middle} | JTPR {middle}-{outer} | first failure: {inner} / {middle} / '
             f'{outer} / {inner} link / {middle} link |', '|---|---|---|---|---|---|']
    for key, c in rows.items():
        k = max(c['chains'], 1)
        lines.append(f"| {key} | {c['chains']} | {c['recovered'] / k:.3f} | {c['joint_inner'] / max(c['pairs_inner'], 1):.3f} | "
                     f"{c['joint_middle'] / max(c['pairs_middle'], 1):.3f} | " + ' / '.join(f'{c[f] / k:.3f}' for f in FAILS) + ' |')
    (out / 'chain_recall.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
