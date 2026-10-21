"""Aggregate read-only GAT-vs-Dijkstra influence audit from V3.2 evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def summarize(path):
    counts = Counter()
    records = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            counts['decisions'] += 1
            gs = r['guide_actions']
            if gs:
                counts['has_guide'] += 1
                counts['raw_top1_on_guide'] += int(r['raw_argmax'] in gs)
                counts['guided_top1_on_guide'] += int(r['guided_argmax'] in gs)
                counts['guidance_changes_top1'] += int(r['raw_argmax'] != r['guided_argmax'])
                counts['has_alternative'] += int(any(x not in gs for x in r['valid_actions']))
            counts['all_seven_nearby_6m'] += int(r['others_count'] == 7 and r['nearby_last_nodes_6m'] == 7)
            counts['some_same_last_node'] += int(r['identical_last_nodes'] > 0)
            counts['stop_available'] += int(0 in r['valid_actions'])
            counts['nearby_1m'] += int(r['nearby_last_nodes_1m'] > 0)
            counts['nearby_2m'] += int(r['nearby_last_nodes_2m'] > 0)
            counts['nearby_6m'] += int(r['nearby_last_nodes_6m'] > 0)
            records.append(r)
    if not records:
        raise RuntimeError('Empty influence trace: no routing decisions')
    all_guide = [r for r in records if r['guide_actions']]
    def ave(key, rows):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(mean(vals), 6) if vals else None
    summary = dict(counts)
    summary.update(
        raw_guide_probability_mean=ave('raw_guide_probability', all_guide),
        guided_guide_probability_mean=ave('guided_guide_probability', all_guide),
        logprob_parity_error_max=max(r['logprob_parity_error'] for r in records),
        other_dist_min_mean=ave('other_last_node_distance_min', records),
        other_dist_max_mean=ave('other_last_node_distance_max', records),
        notes=[
          'All percentages conditional on decisions with a Dijkstra guide unless indicated.',
          'A change in the argmax is a within-observation influence audit, not a performance gain.',
          'Last-node distance should NOT be interpreted as physical continuous AGV separation.',
          '100% guide follow can occur with a greedy policy and logit prior; compare raw vs guided top1.',
          'Do NOT enable Stop blindly; first inspect the original environment mask and safe-stop semantics.',
        ])
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.folder.glob('influence_seed*.jsonl'))
    if not paths:
        p.error('No influence_seed*.jsonl in folder')
    summary = {}
    for path in paths:
        data = summarize(path)
        summary[path.stem] = data
        print('\n===', path.name, '===')
        for k in ('decisions','has_guide','has_alternative','raw_top1_on_guide',
                  'guided_top1_on_guide','guidance_changes_top1',
                  'raw_guide_probability_mean','guided_guide_probability_mean',
                  'stop_available','nearby_1m','nearby_2m','nearby_6m',
                  'all_seven_nearby_6m','some_same_last_node',
                  'other_dist_min_mean','other_dist_max_mean','logprob_parity_error_max'):
            print(f'  {k}: {data.get(k,0)}')
    out = args.folder/'influence_analysis_v3_2.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSaved:', out)


if __name__ == '__main__':
    main()
