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
from paper_queue import Queue, choose, need_gb, ram_gb, DEFAULT_RAM_GB  # noqa: E402

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
        waited = {'01_big.sh': 25.0}                                              # past DRAIN_AFTER_H (24 h)
        job, drains = choose(JOBS, NEEDS, [held('0', 6)], {}, '0', 'heavy', waited)
        self.assertEqual((job, drains), (None, {'0': '01_big.sh'}))
        self.assertEqual(choose(JOBS, NEEDS, [held('0', 6)], drains, '0', 'light'), (None, drains))   # nothing new
        self.assertEqual(choose(JOBS, NEEDS, [held('1', 6)], drains, '1', 'heavy', waited)[0], '02_mid.sh')  # others go on
        self.assertEqual(choose(JOBS, NEEDS, [], drains, '0', 'heavy', waited), ('01_big.sh', {}))         # GPU 0 empty

    def test_a_held_job_that_starts_elsewhere_releases_its_gpu(self):
        job, drains = choose(JOBS, NEEDS, [], {'0': '01_big.sh'}, '2', 'heavy')
        self.assertEqual((job, drains), ('01_big.sh', {}))
        self.assertEqual(choose(JOBS[2:], NEEDS, [held('0', 6)], {'0': '01_big.sh'}, '0', 'light'), ('03_small.sh', {}))


class HostMemoryTests(unittest.TestCase):
    def test_a_job_starts_only_within_the_host_memory_budget(self):
        ram = {'01_big.sh': 40, '02_mid.sh': 20, '03_small.sh': 30, '04_small.sh': 5}
        held_ram = [dict(gpu='1', gb=4, ram=45, slot='heavy')]                    # 45 of 70 GB host memory in use
        self.assertEqual(choose(JOBS, NEEDS, held_ram, {}, '0', 'heavy', ram_needs=ram), ('02_mid.sh', {}))
        self.assertEqual(choose(JOBS, NEEDS, held_ram, {}, '1', 'light', ram_needs=ram), ('04_small.sh', {}))  # not 03
        tight = [dict(gpu='1', gb=4, ram=68)]
        self.assertEqual(choose(JOBS, NEEDS, tight, {}, '0', 'heavy', ram_needs=ram), (None, {}))
        # reservations without host memory count the default (12 GB): 5 * 12 + 5 fits 70, 6 * 12 does not
        self.assertEqual(choose(['04_small.sh'], NEEDS, [held('1', 1)] * 5, {}, '1', 'light', ram_needs=ram),
                         ('04_small.sh', {}))
        self.assertEqual(choose(['04_small.sh'], NEEDS, [held('1', 1)] * 6, {}, '1', 'light', ram_needs=ram), (None, {}))

    def test_a_job_starved_of_host_memory_is_the_only_one_that_may_start(self):
        ram = {'01_big.sh': 40, '02_mid.sh': 20, '03_small.sh': 30, '04_small.sh': 5}
        held_ram = [dict(gpu='1', gb=4, ram=45)]
        waited = {'01_big.sh': 25.0}
        self.assertEqual(choose(JOBS, NEEDS, held_ram, {}, '0', 'heavy', waited, ram_needs=ram), (None, {}))
        self.assertEqual(choose(JOBS, NEEDS, held_ram, {}, '1', 'light', waited, ram_needs=ram), (None, {}))
        self.assertEqual(choose(JOBS, NEEDS, [], {}, '0', 'heavy', waited, ram_needs=ram), ('01_big.sh', {}))

    def test_declaration_pick_and_setram(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'jobs').mkdir()
            (root / 'jobs' / '01_a.sh').write_text('#!/bin/bash\n# gpu_gb: 8\n# ram_gb: 66\necho a\n')
            (root / 'jobs' / '02_b.sh').write_text('#!/bin/bash\n# gpu_gb: 4\necho b\n')
            self.assertEqual((ram_gb(root / 'jobs' / '01_a.sh'), ram_gb(root / 'jobs' / '02_b.sh')), (66.0, DEFAULT_RAM_GB))
            queue = Queue(root)
            self.assertEqual(queue.pick('0', os.getpid(), 'heavy'), '01_a.sh')
            self.assertIsNone(queue.pick('1', os.getppid(), 'heavy'))              # 66 + 12 > 70
            queue.setram(os.getpid(), 50)
            self.assertEqual(queue.pick('1', os.getppid(), 'heavy'), '02_b.sh')      # 50 + 12 fits
            self.assertIn('host memory reserved: 62 of 70 GB', queue.status())


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
            (root / 'jobs' / '03_c.sh').write_text('#!/bin/bash\n# gpu_gb: 4\necho c\n')
            queue.retire(os.getpid())
            self.assertIsNone(queue.pick('2', os.getpid(), 'heavy'))                  # a retired slot takes nothing
            self.assertEqual(queue.pick('2', os.getppid(), 'heavy'), '03_c.sh')       # another slot does



class MonitorTests(unittest.TestCase):
    def test_host_memory_of_a_process_tree_and_summary(self):
        from gpu_monitor import tree_memory, summarise
        child = subprocess.Popen([sys.executable, '-c', 'import sys, time; b = bytearray(300 * 2 ** 20); '
                                  'print(1, flush=True); time.sleep(30)'], stdout=subprocess.PIPE, text=True)
        try:
            child.stdout.readline()                                               # the 300 MB are allocated
            total, anon = tree_memory(os.getpid())
            self.assertGreater(anon, 0.29)
            self.assertGreaterEqual(total, anon)
            without_child = tree_memory(os.getpid(), skip=child.pid)[1]
            self.assertLess(without_child, anon - 0.25)
        finally:
            child.kill(); child.wait(); child.stdout.close()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run_gpu.csv'
            path.write_text('time_s,utilization.gpu,memory.used,memory.total,power.draw,host_rss_gb,host_anon_gb\n'
                            '0,50,1024,24564,100,2.0,1.5\n10,70,2048,24564,120,6.0,3.5\n')
            out = summarise(path)
            self.assertEqual((out['host_rss_peak_gb'], out['host_anon_peak_gb'], out['mem_peak_gb']), (6.0, 3.5, 2.0))


if __name__ == '__main__':
    unittest.main()
