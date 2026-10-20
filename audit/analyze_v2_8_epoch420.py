"""Read-only aggregation of supplied v2.8 results; never executes model code."""
import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = [r['filename'] for r in rows]
    assert len(names) == len(set(names))
    result = dict(volumes=len(rows), filenames=sorted(names), pair_failures=dict(
        sum((Counter(r['pair_failures']) for r in rows), Counter())))
    for head in ('nucleus', 'cytoplasm'):
        stages, monitoring, stats = Counter(), Counter(), Counter()
        lost = gained = fragmented_missed = 0
        lost_examples = []
        for row in rows:
            a = row['stage_audit'][head]
            monitoring.update(a['monitoring'])
            for name, s in a['stages'].items():
                stages[name] += s['oracle_one_to_one_tp']
            before = a['stages']['before_nms']['best_iou_per_gt']
            final = a['stages']['final']['best_iou_per_gt']
            lost_ids = [g for g in before if before[g] >= .5 and final[g] < .5]
            gained_ids = [g for g in before if before[g] < .5 and final[g] >= .5]
            lost += len(lost_ids); gained += len(gained_ids)
            if lost_ids:
                lost_examples.append(dict(filename=row['filename'], lost=len(lost_ids), ids=lost_ids))
            fragmented_missed += sum(len(parts) > 1 and final[g] < .5 for g, parts in
                                     a['fragmentation']['final']['parts_per_gt'].items())
            ref = row['refinement'][head]
            for key in ('attempted', 'accepted', 'written', 'rejected_remainder', 'rejected_nested'):
                stats[key] += ref.get(key, 0)
            conflicts = ref.get('nested_conflicts', [])
            assert len(conflicts) == ref.get('rejected_nested', 0)
            for c in conflicts:
                stats['conflicts'] += 1
                stats['low_iou_conflicts'] += c['iou'] < .5
                stats['rejected_larger'] += c['rejected_contains_accepted']
                stats['rejected_at_least_twice_larger'] += c['rejected_area'] >= 2*c['accepted_area']
                stats['rejected_at_most_half_size'] += 2*c['rejected_area'] <= c['accepted_area']
                assert c['accepted_quality'] >= c['rejected_quality'] - 1e-7
                assert c['smaller_coverage'] >= .95 - 1e-7
        result[head] = dict(stages=dict(stages), monitoring=dict(monitoring), assembly=dict(stats),
                            assembly_gt_lost=lost, assembly_gt_gained=gained,
                            fragmented_without_good_final=fragmented_missed,
                            worst_assembly_losses=sorted(lost_examples,key=lambda x:-x['lost'])[:5])
    return result


def main():
    folder = ROOT/'fixes_v2.8'
    run, effective, history = (read(folder/n) for n in ('run.json','effective_config.json','history.json'))
    out = dict(inputs={p.name:dict(size=p.stat().st_size,sha256=digest(p)) for p in folder.iterdir() if p.is_file()},
               source_verification={n:dict(local_exists=(ROOT/n).exists(),match=digest(ROOT/n)==s if (ROOT/n).exists() else None)
                                    for n,s in run['source_sha256'].items()},
               started_utc=run['started_utc'],packages=run['packages'],
               config={k:effective.get(k) for k in ('resume_from_checkpoint','mask_quality_target','mask_assembly_nucleus',
                     'mask_assembly_cytoplasm','mask_containment_threshold','seed','nucleus_grid','cytoplasm_grid',
                     'prob_thresh_nucleus','prob_thresh_cytoplasm','mask_thresh','reset_best_on_resume')},
               rays_sha256=hashlib.sha256(json.dumps(effective['rays'],sort_keys=True).encode()).hexdigest(),
               history_epochs=len(history['train_loss']),best_fullval=read(folder/'best_fullval.json'),epochs={})
    for ep,base in ((400,ROOT/'fixes_v2.6'),(415,folder),(420,folder)):
        summary=aggregate(base/f'epoch_{ep:04d}_refinement.jsonl')
        i=history['fullval_epoch'].index(ep)
        summary['history']={k:v[i] for k,v in history.items() if k.startswith('fullval_') and i<len(v)}
        for head in ('nucleus','cytoplasm'):
            assert summary[head]['stages']['final']==summary['history'][f'fullval_{head}_tp']
            for k,v in summary[head]['monitoring'].items():
                assert summary['history'][f'fullval_{head}_{k}']==v,(ep,head,k)
        p=summary['pair_failures'];assert abs(p['joint_tp']/p['total_pairs']-summary['history']['fullval_jtpr'])<1e-10
        out['epochs'][str(ep)]=summary
    assert out['epochs']['400']['filenames']==out['epochs']['415']['filenames']==out['epochs']['420']['filenames']
    out['quality_history']={str(ep):{k:v[ep-1] for k,v in history.items() if 'quality' in k or k in ('lr','val_nucleus_mask','val_cytoplasm_mask')}
                            for ep in (411,415,420,421)}
    with (ROOT/'audit/v2_8_epoch420_artifacts.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    for ep,s in out['epochs'].items():
        print(ep,'pairs',s['pair_failures'])
        for head in ('nucleus','cytoplasm'):
            a=s[head];print(head,{k:v for k,v in a.items() if k not in ('monitoring','worst_assembly_losses')})


if __name__ == '__main__':
    main()
