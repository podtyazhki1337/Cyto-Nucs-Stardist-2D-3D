"""GPU job queue of the paper runs (paper_queue.py): memory budgets, heavy and light slots, drains."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_queue import Queue, choose, need_gb                      # noqa: E402

NEEDS = {'01_big.sh': 23, '02_mid.sh': 16, '03_small.sh': 6, '04_small.sh': 4}
JOBS = sorted(NEEDS)


def held(gpu, gb):
    return dict(gpu=gpu, gb=gb)


class ChooseTests(unittest.TestCase):
    def test_heavy_slot_takes_the_first_heavy_job_that_fits(self):
        self.assertEqual(choose(JOBS, NEEDS, [], {}, '0', 'heavy'), ('01_big.sh', {}))
        # 6 GB in use: the whole-GPU job does not fit, the next heavy one does (backfill, no hold yet)
        self.assertEqual(choose(JOBS, NEEDS, [held('0', 6)], {}, '0', 'heavy'), ('02_mid.sh', {}))

    def test_light_slot_shares_a_busy_gpu_but_leaves_an_empty_one_to_heavy_jobs(self):
        self.assertEqual(choose(JOBS, NEEDS, [held('0', 16)], {}, '0', 'light'), ('03_small.sh', {}))   # 16 + 6
        self.assertEqual(choose(JOBS, NEEDS, [held('0', 16), held('0', 6)], {}, '0', 'light'), (None, {}))  # 22 + 4 > 23
        self.assertEqual(choose(JOBS, NEEDS, [], {}, '0', 'light'), (None, {}))
        self.assertEqual(choose(JOBS[2:], NEEDS, [], {}, '0', 'light'), ('03_small.sh', {}))             # no heavy job waits

    def test_light_slot_leaves_a_fitting_heavy_job_to_the_idle_heavy_slot(self):
        external = [dict(gpu='1', gb=6)]                                         # a job started outside the slots
        self.assertEqual(choose(JOBS[1:], NEEDS, external, {}, '1', 'light'), (None, {}))    # 6 + 16 fits: heavy's
        busy = external + [dict(gpu='1', gb=4, slot='heavy')]                     # the heavy slot runs light work
        self.assertEqual(choose(JOBS[1:], NEEDS, busy, {}, '1', 'light'), ('03_small.sh', {}))

    def test_light_slot_never_takes_heavy_jobs(self):
        self.assertEqual(choose(['02_mid.sh'], NEEDS, [held('0', 4)], {}, '0', 'light'), (None, {}))

    def test_heavy_slot_with_a_heavy_job_on_its_gpu_only_adds_light_work(self):
        self.assertEqual(choose(JOBS, NEEDS, [held('2', 16)], {}, '2', 'heavy'), ('03_small.sh', {}))
        self.assertEqual(choose(JOBS, NEEDS, [held('2', 22)], {}, '2', 'heavy'), (None, {}))

    def test_a_long_waiting_heavy_job_makes_one_gpu_empty_for_it(self):
        waited = {'01_big.sh': 7.0}
        job, drains = choose(JOBS, NEEDS, [held('0', 6)], {}, '0', 'heavy', waited)
        self.assertEqual((job, drains), (None, {'0': '01_big.sh'}))
        self.assertEqual(choose(JOBS, NEEDS, [held('0', 6)], drains, '0', 'light'), (None, drains))   # nothing new
        self.assertEqual(choose(JOBS, NEEDS, [held('1', 6)], drains, '1', 'heavy', waited)[0], '02_mid.sh')  # others go on
        self.assertEqual(choose(JOBS, NEEDS, [], drains, '0', 'heavy', waited), ('01_big.sh', {}))         # GPU 0 empty

    def test_a_held_job_that_starts_elsewhere_releases_its_gpu(self):
        job, drains = choose(JOBS, NEEDS, [], {'0': '01_big.sh'}, '2', 'heavy')
        self.assertEqual((job, drains), ('01_big.sh', {}))
        self.assertEqual(choose(JOBS[2:], NEEDS, [held('0', 6)], {'0': '01_big.sh'}, '0', 'light'), ('03_small.sh', {}))


class QueueTests(unittest.TestCase):
    def test_pick_moves_the_job_reserves_and_release_and_dead_pids_free_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'jobs').mkdir()
            (root / 'jobs' / '01_a.sh').write_text('#!/bin/bash\n# gpu_gb: 8\necho a\n')
            (root / 'jobs' / '02_b.sh').write_text('#!/bin/bash\necho b\n')            # undeclared: whole GPU
            self.assertEqual(need_gb(root / 'jobs' / '02_b.sh'), 23.0)
            queue = Queue(root)
            self.assertEqual(queue.pick('0', os.getpid(), 'heavy'), '02_b.sh')
            self.assertTrue((root / 'running' / '02_b.sh').is_file())
            self.assertIsNone(queue.pick('0', os.getpid(), 'light'))                # 23 GB reserved
            queue.release('0', os.getpid())
            self.assertEqual(queue.pick('0', os.getpid(), 'heavy'), '01_a.sh')      # no heavy job left: light work
            dead = subprocess.run([sys.executable, '-c', 'import os; print(os.getpid())'], capture_output=True, text=True)
            queue.reserve('1', int(dead.stdout), 20, 'gone')
            state = json.loads((root / 'state.json').read_text())
            self.assertIn('gone', [r['job'] for r in state['reservations']])
            with queue.locked() as state:                                         # dead process: dropped
                self.assertNotIn('gone', [r['job'] for r in state['reservations']])


if __name__ == '__main__':
    unittest.main()
