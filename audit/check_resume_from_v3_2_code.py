"""A checkpoint written by the v3.2 model code (fine head on the first prompt; no layer-name keys, no
optimizer order list, as the real v3.2 run files) resumes with the current code: weights through the
saved graph, Adam state verified by the saved values. Run from tests/:
    cd tests && python ../audit/check_resume_from_v3_2_code.py -> ../audit/v3_3_resume_from_v3_2_code.txt (stdout)
"""
import shutil
from pathlib import Path as _P
_snap = _P.cwd().parent / 'audit' / 'history' / '20260929T072614443625Z_edge_fragments_v3_3' / 'before'
_dst = _P.cwd().parent / '.cache' / 'v32model'; _dst.mkdir(parents=True, exist_ok=True)
for _f in ('cytonucs_model.py', 'cytonucs_fine_mask.py'):
    shutil.copy2(_snap / _f, _dst / _f)
shutil.rmtree(_P.cwd().parent / '.cache' / 'compat_v32', ignore_errors=True)
import sys, subprocess, json, os
from pathlib import Path
root = Path.cwd().parent
tmp = root / '.cache' / 'compat_v32'; tmp.mkdir(parents=True, exist_ok=True)
old = r'''
import sys, contextlib, io, numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '../.cache/v32model')
import cytonucs_fine_mask, cytonucs_model
assert 'v32model' in cytonucs_model.__file__
from test_fixes import config
from test_architecture_v2 import batch
from cytonucs_losses import CytoNucsLoss
from cytonucs_trainer import CytoNucsTrainer
cfg = config(semantic_foreground_weight=1., refinement_full_resolution=True, nucleus_grid=(1,2,2), cytoplasm_grid=(1,2,2),
             prompts_per_crop=3, mask_quality_head=True, refinement_depth=2, refinement_gradient_scale=.1)
with contextlib.redirect_stdout(io.StringIO()):
    m = cytonucs_model.build_cytonucs_model(cfg, basedir=sys.argv[1])
    t = CytoNucsTrainer(m, cfg, CytoNucsLoss(cfg), 0); x, y = batch(cfg); t.current_epoch_var.assign(731)
    t.train_step(x, y); t.train_step(x, y); t.state_epoch.assign(730)
    t.save_checkpoint(sys.argv[1] + '/weights_epoch_730.weights.h5')
import h5py
with h5py.File(sys.argv[1] + '/weights_epoch_730.weights.h5', 'a') as f:   # real v3.2 files: no names, no order list
    for key in ('cytonucs_optimizer_paths',):
        if key in f.attrs: del f.attrs[key]
    if 'cytonucs_by_name' in f: del f['cytonucs_by_name']
    print('v3.2-like file: order attr', 'cytonucs_optimizer_paths' in f.attrs, 'by-name group', 'cytonucs_by_name' in f)
np.save(sys.argv[1] + '/slots.npy', np.concatenate([v.numpy().ravel() for v in t.optimizer.variables[2:]]))
print('saved with', cytonucs_model.__file__)
'''
new = r'''
import sys, contextlib, io, numpy as np
sys.path.insert(0, '..')
import cytonucs_model
assert 'v32model' not in cytonucs_model.__file__
from test_fixes import config
from test_architecture_v2 import batch
from cytonucs_losses import CytoNucsLoss
from cytonucs_trainer import CytoNucsTrainer
cfg = config(semantic_foreground_weight=1., refinement_full_resolution=True, nucleus_grid=(1,2,2), cytoplasm_grid=(1,2,2),
             prompts_per_crop=3, mask_quality_head=True, refinement_depth=2, refinement_gradient_scale=.1)
with contextlib.redirect_stdout(io.StringIO()):
    m = cytonucs_model.build_cytonucs_model(cfg, basedir=sys.argv[1]); m.load_weights(sys.argv[1] + '/weights_epoch_730.weights.h5')
    t = CytoNucsTrainer(m, cfg, CytoNucsLoss(cfg), 0)
    ok = t.restore_training_state(sys.argv[1] + '/weights_epoch_730.weights.h5')
saved = np.load(sys.argv[1] + '/slots.npy')
now = np.concatenate([v.numpy().ravel() for v in t.optimizer.variables[2:]])
print('restored', ok, 'epoch', int(t.state_epoch.numpy()), 'Adam slots identical', saved.shape == now.shape and np.array_equal(saved, now))
x, y = batch(cfg); t.current_epoch_var.assign(731)
with contextlib.redirect_stdout(io.StringIO()):
    r = t.train_step(x, y)
print('train step after resume ok, fine loss keys', sorted(k for k in r[-1] if 'fine' in k))
'''
env = dict(os.environ, PYTHONIOENCODING='utf-8', TF_CPP_MIN_LOG_LEVEL='3')
py = str(root / '.venv-audit' / 'Scripts' / 'python.exe')
for code in (old, new):
    out = subprocess.run([py, '-c', code, str(tmp)], capture_output=True, text=True, env=env)
    print(out.stdout.strip()[-600:]); print(out.stderr.strip()[-800:] if out.returncode else '')
