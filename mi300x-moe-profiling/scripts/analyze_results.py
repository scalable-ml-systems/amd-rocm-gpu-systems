#!/usr/bin/env python3
"""Reproduce the committed MoE capture analysis. Python standard library only.

IDs and expected counts belong to these captures, not arbitrary future runs.
Memory units are an explicit assumption; see RUNBOOK.md.
"""
import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = Path('benchmark-results/mix300-device/vllm-kernel-64tok/5ab03fd80643')
COUNTER_DIR = Path('benchmark-results/mix300-device/vllm-kernel-64tok-counters/1e0279077382')
MFMA = 'SQ_INSTS_VALU_MFMA_MOPS_BF16'
EXPECTED = {'9119': 48, '9120': 48, '9121': 3024, '9122': 3024}


def read_csv(path, required):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    return rows


def duration(row):
    value = int(row['End_Timestamp']) - int(row['Start_Timestamp'])
    if value <= 0:
        raise ValueError('Non-positive dispatch duration')
    return value


def size(row, prefix):
    return math.prod(int(row[f'{prefix}_{axis}']) for axis in 'XYZ')


def table(headers, rows):
    print('| ' + ' | '.join(headers) + ' |')
    print('| ' + ' | '.join('---' for _ in headers) + ' |')
    for row in rows:
        print('| ' + ' | '.join(str(x) for x in row) + ' |')
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--memory-unit-bytes', type=int, choices=(1000, 1024), default=1024,
                        help='Explicit bytes per FETCH_SIZE/WRITE_SIZE unit; default 1024, provisional')
    args = parser.parse_args()
    root = args.project_root.resolve()
    paths = [root / TRACE_DIR / '1_kernel_trace.csv',
             root / TRACE_DIR / '1_kernel_stats.csv',
             root / TRACE_DIR / '1_domain_stats.csv',
             root / COUNTER_DIR / '1_counter_collection.csv']
    trace = read_csv(paths[0], ['Kernel_Name', 'Kernel_Id', 'Dispatch_Id', 'Start_Timestamp', 'End_Timestamp',
                              *[f'{p}_{a}' for p in ('Grid_Size', 'Workgroup_Size') for a in 'XYZ']])
    stats = read_csv(paths[1], ['Name', 'Calls', 'TotalDurationNs', 'Percentage'])
    domain = read_csv(paths[2], ['Name', 'Calls', 'TotalDurationNs'])
    counters = read_csv(paths[3], ['Kernel_Name', 'Kernel_Id', 'Dispatch_Id', 'Counter_Name', 'Counter_Value',
                                  'Grid_Size', 'Workgroup_Size', 'Start_Timestamp', 'End_Timestamp'])
    families, variants, samples = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in trace:
        families[row['Kernel_Name']].append(duration(row))
        if row['Kernel_Name'] == 'fused_moe_kernel':
            variants[row['Kernel_Id']].append(row)
    for row in counters:
        if row['Kernel_Name'] == 'fused_moe_kernel':
            value = float(row['Counter_Value'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid counter value')
            samples[(row['Kernel_Id'], row['Counter_Name'])].append(row)
    total_ns = sum(sum(v) for v in families.values())
    checks = []
    def check(name, condition):
        checks.append((name, bool(condition)))
    check('Total dispatches = 54,194', len(trace) == 54194)
    check('Four expected MoE variants', set(variants) == set(EXPECTED))
    for kid, count in EXPECTED.items():
        check(f'Kernel {kid}: {count} dispatches', len(variants[kid]) == count)
    summary = {r['Name']: r for r in stats}
    check('Kernel summary names match trace', set(summary) == set(families))
    for name, times in families.items():
        r = summary.get(name, {})
        check(f'Summary agrees: {name}', int(r.get('Calls', -1)) == len(times)
              and int(r.get('TotalDurationNs', -1)) == sum(times)
              and abs(float(r.get('Percentage', -1)) - 100 * sum(times) / total_ns) <= .011)
    d = next((r for r in domain if r['Name'] == 'KERNEL_DISPATCH'), {})
    check('Domain summary agrees', int(d.get('Calls', -1)) == len(trace)
          and int(d.get('TotalDurationNs', -1)) == total_ns)
    for kid, count in EXPECTED.items():
        for counter in (MFMA, 'FETCH_SIZE', 'WRITE_SIZE'):
            rows = samples[(kid, counter)]
            check(f'{kid} / {counter}: {count // 3} unique samples', len(rows) == count // 3
                  and len({r['Dispatch_Id'] for r in rows}) == len(rows))
        if variants[kid]:
            signature = {(size(r, 'Grid_Size'), size(r, 'Workgroup_Size')) for r in variants[kid]}
            check(f'{kid}: geometry agrees across captures', all(
                (int(r['Grid_Size']), int(r['Workgroup_Size'])) in signature
                for counter in (MFMA, 'FETCH_SIZE', 'WRITE_SIZE') for r in samples[(kid, counter)]))
    print('# MI300X MoE capture analysis\n')
    print('## Input fingerprints\n')
    for p in paths:
        print(f'- `{p.relative_to(root)}` — SHA256 `{hashlib.sha256(p.read_bytes()).hexdigest()}`')
    print('\n## 1. Trace summary\n')
    print(f'{len(trace):,} dispatches; {total_ns / 1e6:.6f} ms summed kernel duration.\n')
    table(['Kernel family', 'Dispatches', 'Summed ms', 'Share %'],
          [(k, f'{len(v):,}', f'{sum(v)/1e6:.6f}', f'{100*sum(v)/total_ns:.2f}')
           for k, v in sorted(families.items(), key=lambda kv: sum(kv[1]), reverse=True)[:5]])
    print('## 2. MoE variants\n')
    table(['ID', 'Dispatches', 'Grid size', 'Workgroup size', 'Mean trace µs'],
          [(k, len(v), ', '.join(map(str, sorted({size(r, 'Grid_Size') for r in v}))),
            ', '.join(map(str, sorted({size(r, 'Workgroup_Size') for r in v}))),
            f'{mean(duration(r) for r in v)/1000:.3f}') for k, v in sorted(variants.items()) if v])
    print('Model-pattern interpretation: 48 × 2 + 63 × 48 × 2 = 6,144 launches.\n'
          'This matches one initial pass and 63 decode steps; counts alone do not label phases.\n')
    print('## 3. Counter samples\n')
    table(['ID', 'Counter', 'Samples', 'Mean', 'Min', 'Max'],
          [(k, c, len(v), f'{mean(float(r["Counter_Value"]) for r in v):.6f}',
            f'{min(float(r["Counter_Value"]) for r in v):.6f}',
            f'{max(float(r["Counter_Value"]) for r in v):.6f}')
           for (k, c), v in sorted(samples.items()) if v])
    print('## 4. Decode arithmetic and memory estimates\n')
    print('Executed BF16 matrix FLOPs = mean MFMA counter × 512 (no additional ×2).\n')
    print(f'Memory bytes = (mean FETCH_SIZE + mean WRITE_SIZE) × {args.memory_unit_bytes}. '
          'The memory-unit factor is provisional until the installed metric definition is archived.\n')
    derived = []
    for kid, label, useful in [('9121', 'gate/up (inferred)', 8*1*2048*1536*2),
                               ('9122', 'down (inferred)', 8*1*768*2048*2)]:
        if not variants[kid] or any(not samples[(kid,c)] for c in (MFMA,'FETCH_SIZE','WRITE_SIZE')):
            continue
        averages = {c: mean(float(r['Counter_Value']) for r in samples[(kid,c)])
                    for c in (MFMA,'FETCH_SIZE','WRITE_SIZE')}
        flops = averages[MFMA] * 512
        traffic = (averages['FETCH_SIZE'] + averages['WRITE_SIZE']) * args.memory_unit_bytes
        ns = mean(duration(r) for r in variants[kid])
        derived.append((kid, label, f'{flops:,.0f}', f'{useful:,}', f'{flops/useful:.3f}',
                        f'{traffic:,.2f}', f'{flops/traffic:.4f}', f'{flops/ns/1000:.3f}', f'{traffic/ns:.3f}'))
    table(['ID', 'Projection', 'Executed FLOPs', 'Useful FLOPs', 'Ratio', 'Bytes', 'FLOPs/byte',
           'Est. TFLOP/s', 'Est. GB/s'], derived)
    print('Rates combine counter-run averages with separate trace-run mean durations. '
          'They are cross-run estimates, not same-dispatch measurements.\n'
          'Equal sample counts do not prove representative sampling. The projection mapping is inferred, '
          'not supplied by the CSV. The 16× arithmetic ratio is not a latency penalty or speedup claim.\n')
    print('## Verification\n')
    failures = [name for name, ok in checks if not ok]
    print(f'{len(checks)-len(failures)}/{len(checks)} capture checks passed.\n')
    for name in failures:
        print(f'- FAIL: {name}')
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
