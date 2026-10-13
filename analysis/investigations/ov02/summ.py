import json, sys
txt = sys.stdin.read()
i = txt.find('{\n')
d = json.loads(txt[i:])
print(d['candidate'], {k: round(v, 4) for k, v in d['kl'].items()}, 'diff', round(d.get('paired_diff', 0), 4), '+-',
      round(d.get('paired_se', 0), 4))
print('  R2', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.get('head_effect_r2', {}).items()})
print('  strengths', d.get('candidate_strengths'), d['seconds'], 's')
