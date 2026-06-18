"""Per-site audit of the LOSO dataset: slide count, class balance, slide-id
prefix breakdown, patient count. Goal = understand why Site_C is hard.
Hospital prefix rules from CLAUDE.md (authoritative):
  SC-01/SC_01 -> Site_A (20x)
  SC-03/SC_03 -> Site_C (20x)
  SC-3-/GC-3-/SC-7- -> Site_D (NOT Site_C) (20x)
  SC-04/SC_04 -> Site_B (40x)
  SC_02/SC-02 -> Site_E (40x)
"""
import csv, re, collections

CSV = "/path/to/data/dataset_csv/tumor_vs_normal_dummy_clean.csv"


def hospital(slide_id):
    s = slide_id
    if re.match(r'^(SC[-_]01)', s):           return 'Site_A'
    if re.match(r'^(SC[-_]02)', s):           return 'Site_E'
    if re.match(r'^(SC[-_]04)', s):           return 'Site_B'
    if re.match(r'^(SC-3-|GC-3-|SC-7-)', s):  return 'Site_D'
    if re.match(r'^(SC[-_]03)', s):           return 'Site_C'
    return f'UNKNOWN:{s[:8]}'


rows = list(csv.DictReader(open(CSV)))
print(f'Total rows: {len(rows)}')
prefix_counter = collections.Counter()
for r in rows:
    prefix_counter[r['slide_id'][:8]] += 1

per_site = collections.defaultdict(list)
for r in rows:
    per_site[hospital(r['slide_id'])].append(r)

print('\n--- Unique slide_id prefixes (top 20) ---')
for p, n in prefix_counter.most_common(20):
    print(f'  {p}  {n}')

print('\n--- Per-site audit ---')
print(f'{"Site":12s} {"slides":>7s} {"patients":>9s} {"tumor":>7s} '
      f'{"normal":>7s} {"t_frac":>7s}  {"prefix_mix"}')
for site in ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']:
    srows = per_site.get(site, [])
    n = len(srows)
    pat = len({r['case_id'] for r in srows})
    tum = sum(1 for r in srows if r['label'] not in ('nm', 'normal', '0'))
    nrm = n - tum
    mix = collections.Counter(r['slide_id'][:5] for r in srows)
    mix_s = ', '.join(f'{k}={v}' for k, v in mix.most_common())
    print(f'{site:12s} {n:7d} {pat:9d} {tum:7d} {nrm:7d} '
          f'{tum/max(n,1):7.3f}  {mix_s}')

unk = [r for r in rows if hospital(r['slide_id']).startswith('UNKNOWN')]
if unk:
    print(f'\n!! {len(unk)} unknown-prefix slides — sample:')
    for r in unk[:10]:
        print('  ', r['slide_id'], r['label'])

# Label values seen
labs = collections.Counter(r['label'] for r in rows)
print('\nLabel value counts:', dict(labs))
