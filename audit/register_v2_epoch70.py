"""Register immutable user results and derive a review without training changes."""
import ast
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    package = ROOT / 'fixes_v2/Epoch70'
    h = read('fixes_v2/Epoch70/history.json')
    v1 = read('fixes_v1/fixes_v1_110Epochs/history.json')
    run = read('fixes_v2/Epoch70/run.json')
    log = (package / 'console.log').read_text(encoding='utf-8')
    sections = re.split(r'FULL-VOLUME VALIDATION \| epoch (\d+)', log)
    rows = [{k.removeprefix('fullval_'): values[i] for k, values in h.items()
             if k.startswith('fullval_')} for i in range(len(h['fullval_epoch']))]
    checks = []
    for row in rows:
        assert row['nucleus_tp'] + row['nucleus_fn'] == 1439
        assert row['cytoplasm_tp'] + row['cytoplasm_fn'] == 1722
        assert abs(row['jtpr'] * 1439 + sum(row['joint_fail_' + k]
                   for k in ('nucleus', 'cell', 'link')) - 1439) < 1e-8
        checks.append({'epoch': row['epoch'], 'counts_consistent': True})
    hashes = {k: sha(ROOT/k) == value for k,value in run['source_sha256'].items()
              if (ROOT/k).exists()}
    assert all(hashes.values())
    volumes = {}
    aggregates = {}
    for i in range(1, len(sections), 2):
        e, body = int(sections[i]), sections[i+1]
        volumes[e] = re.findall(r'\[\d+/61\] (\S+) \| GT nuc=(\d+) cyto=(\d+)', body)
        stats = re.findall(r"\[refinement mask\] nucleus=(\{[^\n]+?\}) cytoplasm=(\{[^\n]+?\})", body)
        aggregates[e] = {head: {k: sum(ast.literal_eval(s[j])[k] for s in stats)
                        for k in ('proposals','after_mask_nms','written')}
                        for j,head in enumerate(('nucleus','cytoplasm'))}
    assert len(volumes[65]) == 61 and len(volumes[70]) == 42
    oldlog = (ROOT/'fixes_v1/fixes_v1_110Epochs/console.log').read_text(encoding='utf-8')
    oldset = set(re.findall(r'\[\d+/61\] (\S+) \| GT nuc=(\d+) cyto=(\d+)',oldlog))
    same_split = set(volumes[65]).issubset(oldset)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snaprel = 'audit/history/' + stamp + '_v2_epoch70_review'
    snap = ROOT/snaprel
    paths = ['AGENTS.md', 'CHANGELOG.md', 'audit/history/INDEX.json',
             'audit/FIXES_V2_EPOCH70_RU.md','audit/v2_epoch70_artifacts.json',
             'audit/register_v2_epoch70.py']
    before = {}
    for name in paths:
        p = ROOT/name
        # This newly introduced audit script has no previous version.
        existed = p.exists() and name != 'audit/register_v2_epoch70.py'
        before[name] = sha(p) if existed else None
        if existed:
            q = snap/'before'/name
            q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,q)
    artifact = {'reviewed_utc': stamp, 'package': 'fixes_v2/Epoch70',
                'files': [{'path': str(p.relative_to(ROOT)), 'bytes':p.stat().st_size,
                           'sha256':sha(p)} for p in sorted(package.iterdir()) if p.is_file()],
                'source_hash_matches':hashes, 'missing_local_sources': [k for k in run['source_sha256'] if not (ROOT/k).exists()],
                'same_validation_filenames_and_gt_as_v1_log':same_split,
                'history_completed_epochs':len(h['train_loss']), 'last_complete_fullval':65,
                'epoch70_completed_volumes':42, 'checks':checks, 'metrics':rows,
                'refinement_aggregates':aggregates,
                'geometry':read('fixes_v2/Epoch70/geometry.json'),
                'packages':run['packages'], 'seed':42,
                'checkpoint_best_reported':'models/cytonucs_emese_HK_3_architecture_v2/weights_best.weights.h5',
                'checkpoint_best_epoch':65, 'weights_present':False}
    dump(ROOT/'audit/v2_epoch70_artifacts.json',artifact)
    lines = ['# fixes_v2: пакет Epoch70', '',
      '## Достоверность и границы сравнения', '',
      'Проверены 11 файлов пакета; SHA-256 сохранены в v2_epoch70_artifacts.json. '
      'Все 12 корневых Python-модулей совпадают с source_sha256 серверного run.json. '
      'Два дополнительных серверных скрипта overfit.py/run_inference.py локально отсутствуют. '
      'Requested config совпадает с поставляемым v2 за исключением редактирования секрета в run.json. '
      'Веса и исходные TIFF в пакете отсутствуют.', '',
      'history содержит 69 завершённых эпох loss и 13 полных validation (5–65). '
      'В console.log есть train/val loss эпохи70 и сохранение weights_epoch_070.weights.h5; '
      'её fullval обрывается после42/61. PNG70 — визуализация одного тома, не общий результат70. '
      'best_fullval.json указывает epoch65. Проверены балансы TP/FN и JTPR/failure counts на всех13 точках.', '',
      f'Совпадение всех61 имён и GT-counts validation65 с логом v1: {same_split}. '
      'Это сравнение двух прогонов на том же validation, но не изолированный A/B отдельного исправления; '
      'хешей TIFF/split manifest нет. Ранее установленное пересечение вариантов кропов train/val остаётся ограничением. '
      'На длинных3D томах результаты здесь не проверены.', '',
      '## Динамика', '',
      '|Эпоха|JTPR v2|JTPR v1|TP/FP ядра v2|TP/FP клетки v2|',
      '|---|---:|---:|---:|---:|']
    for row in rows:
        e = row['epoch']; j = v1['fullval_epoch'].index(e)
        lines.append(f"|{e}|{row['jtpr']:.4f}|{v1['fullval_jtpr'][j]:.4f}|{row['nucleus_tp']}/{row['nucleus_fp']}|{row['cytoplasm_tp']}/{row['cytoplasm_fp']}|")
    lines += ['', '## Выводы', '',
      '- v2 epoch65: JTPR329/1439=22.863%; лучший v1 epoch105:277/1439=19.249%. '
      'Разница +52 пары, +3.61 процентного пункта. V2 уже достигает более высокого JTPR за меньшее число эпох; '
      'выигрыш по времени и вклад отдельных изменений не измерены.',
      '- Улучшение не равномерное: nucleus precision=516/2061=25.04%, recall=35.86%; '
      'v1 epoch105 precision=617/995=62.01%, recall=42.88%. Cell precision=468/3244=14.43%, recall=27.18%; '
      'v1 epoch105=11.34%/22.47%. Nuclear FP1545 против378. Поэтому считать всю детекцию улучшенной нельзя.',
      '- На65: 923 пары потеряны из-за nuclear matching,183 из-за cells,4 из-за linking. '
      'Это порядок разложения метрики, а не доказательство, что вся причина в probability head. '
      'Основной текущий приоритет — качество отдельных ядер/устранение дублей.',
      '- PNG70 показывает несколько меток внутри области одного GT-объекта, особенно клетки. '
      'Высокий semantic Dice0.743/0.875 на этом томе не означает правильное разделение instances. '
      'Числа объектов на рисунке относятся к отображаемому срезу.',
      '- Средние train/val loss за41–50:6.776/7.264; за60–69:6.147/6.340. '
      'Оба mask loss снижаются. По этим данным обучение продолжается; устойчивое плато или расходимость не подтверждены. '
      'Quality loss не падает (val0.00816→0.00889), но цель IoU меняется вместе с масками; '
      'это не самостоятельное доказательство плохой калибровки.', '',
      '## Конкретный провал перед сборкой масок', '',
      '`hela_spheroid_z016_y0896_x0896_a00.tiff`: GT132 ядер/177 клеток. '
      'На70 найдены214 nuclear и529 cell центров, выполнены743 prompt forwards. '
      'Финал177 ядер/0 клеток; cell proposals=0 уже на входе assemble_masks. '
      'Аналогичный ноль cells есть на20,25,35,45,55; на65 выдано257 клеток, JTPR0.038. '
      'Это не отсутствие клеточных центров и не подавление mask NMS.', '',
      'По коду кандидаты до assemble_masks исчезают при пустом prompt_component '
      '(включая отсутствие подходящей связной компоненты) либо quality<0.25. '
      'В логе нет распределения quality и раздельных counters; установить причину из пакета нельзя. '
      'Quality head усредняет признаки по всему ROI (GlobalAveragePooling); зависимость оценки от плотности/размера ROI '
      '— конкретная гипотеза для проверки, не установленная причина.', '',
      '## Следующий шаг', '',
      '1. Текущий run продолжать до100–110 с уже встроенной validation; новый старт с0 не нужен. '
      'Новый контрольный пакет достаточно передать на100, с полностью завершённой fullval.',
      '2. При следующем изменении inference добавить counters empty-mask / missing-component / quality-rejected '
      'и квантили quality отдельно по heads. На фиксированном65/70 checkpoint проверить этот плотный том с quality_min0.25/0 '
      'при остальных одинаковых настройках. Это проверка конкретного фильтра; полные отдельные inference всех61 томов сейчас не требуются.',
      '3. Затем на том же checkpoint сравнить устранение nuclear duplicates (mask NMS0.8 против более строгого значения). '
      'Проверять TP/FN/JTPR: снижение FP само по себе не успех. Не менять одновременно LR, loss и архитектуру.',
      '4. Только после разделения причин принимать решение о quality pooling или instance-consistency. '
      'Ни отключение quality, ни изменение NMS автоматически не применены.', '',
      '## Условия воспроизводимости', '',
      'Run20260917T115534327262Z; experiment cytonucs_emese_HK_3_architecture_v2; seed42; '
      'TensorFlow2.18.0/Keras3.12.0/StarDist0.9.1/NumPy1.26.4. Rays96; '
      'anisotropy rays19:1:1/28:1:1, grids1:2:2, scales54/99. '
      'Probability0.5/0.3; mask0.45; maskNMS0.8/0.5; quality_min0.25; containment0.95; peaks mode. '
      'Строка Running canonical StarDist NMS в логе — общий неудачный заголовок: '
      'по совпавшему исходнику peaks branch обходит polyhedral NMS.', '',
      f'Снимок документации: `{snaprel}`. Исходники обучения/конфиги и пользовательские файлы не изменены.']
    (ROOT/'audit/FIXES_V2_EPOCH70_RU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    memory = ('\n\n- 2026-09-18: зарегистрирован неизменяемый fixes_v2/Epoch70. History69, полная validation65; '
              'fullval70 только42/61. Best65 JTPR329/1439=0.228631 против best v1 277/1439; '
              'nuclear TP516/FP1545, cell TP468/FP2776. Все12 source hashes совпали. '
              'На плотном томе70 cell529 anchors→0 candidates доNMS, причина empty/quality не разделена. '
              'Отчёт audit/FIXES_V2_EPOCH70_RU.md, хеши audit/v2_epoch70_artifacts.json. '
              'Продолжать текущий run; приоритет duplicate nuclei и диагностика pre-NMS rejection; '
              'training/config не изменены. Снимок '+snaprel+'.\n')
    with (ROOT/'AGENTS.md').open('a',encoding='utf-8') as f: f.write(memory)
    with (ROOT/'CHANGELOG.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 2026-09-18 — анализ пользовательского пакета v2 Epoch70\n\n'
                'Запрос: добавлены результаты70 эпохиv2. Проверены JSON, лог, PNG70, '
                '12 source hashes,13 балансов метрик и61 имя/GT validation. '
                'Выполнены отчёт и регистрация SHA-256; новые гипотезы отделены от измерений. '
                'JTPR65=22.86%, nuclear FP выросли; fullval70 неполная. '
                'Изменений обучения/конфига нет, тесты обучения не запускались. '
                'Ограничения: отсутствуют веса/TIFF, нет завершённой метрики70 и раздельных rejection counters. '
                'Отчёт audit/FIXES_V2_EPOCH70_RU.md; снимок '+snaprel+'.\n')
    index = read('audit/history/INDEX.json')
    index['snapshots'].append({'directory':snaprel,'created_utc':stamp,'manifest':snaprel+'/manifest.json'})
    dump(ROOT/'audit/history/INDEX.json',index)
    entries=[]
    for name in paths:
        p=ROOT/name; q=snap/'after'/name
        q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
        entries.append({'path':name,'status':'new' if before[name] is None else 'modified',
                        'sha256_before':before[name],'sha256_after':sha(p)})
    dump(snap/'manifest.json',{'created_utc':stamp,'status':'complete','files':entries})
    print(json.dumps({'snapshot':snaprel,'source_hashes_matched':len(hashes),
                      'same_validation_names_gt':same_split,'metric_balance_checks':len(checks)},indent=2))

if __name__ == '__main__':
    main()
