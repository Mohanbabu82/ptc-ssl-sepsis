import json, re, pandas as pd
def lines(f): return open(f'logs/{f}', encoding='utf8').read().splitlines()
A, B = lines('v2a_lines.txt'), lines('v2b_lines.txt')
def js(L, tag): return [json.loads(l[len(tag)+1:]) for l in L if l.startswith(tag + ' ')]
runs_a = pd.DataFrame(js(A, 'RESULT')); runs_a.insert(0, 'notebook', 'v2a')
runs_b = pd.DataFrame(js(B, 'RESULT')); runs_b.insert(0, 'notebook', 'v2b')
pert = pd.DataFrame(js(A, 'PERT')); pinfo = pd.DataFrame(js(A, 'PI'))
for d in (pert,):
    d['dAUROC'] = d.AUROC - d.AUROC_clean; d['dAUPRC'] = d.AUPRC - d.AUPRC_clean; d['dutility'] = d.utility - d.utility_clean
# MS summaries
def ms(L, nb):
    rows = []
    for l in L:
        if not l.startswith('MS '): continue
        p = l.split()
        if p[2] in ('pooled', 'A2B', 'B2A'):
            base = dict(job=p[1], protocol=p[2], arm=p[3], fraction=float(p[4])); rest = p[5:]
        elif p[1] == 'external':
            base = dict(job='external', protocol=p[2], arm=p[3], fraction=float(p[4])); rest = p[5:]
        else:
            base = dict(job=p[1], protocol='pooled', arm=p[2], fraction=float(p[3])); rest = p[4:]
        for kv in rest:
            k, v = kv.split('='); m, s, n = v.split('|')
            base[k + '_mean'] = float(m); base[k + '_sd'] = float(s); base['n_seeds'] = int(n)
        base['notebook'] = nb; rows.append(base)
    return rows
summ = pd.DataFrame(ms(A, 'v2a') + ms(B, 'v2b'))
# CP
cre = re.compile(r'^CP (.+?) (\S+) (\S+) - (\S+) (AUROC|AUPRC) n=(\d+) perseed=(\S+) \[(\S+),(\S+)\] p=(\S+) holm=(\S+) ens=(\S+) \[(\S+),(\S+)\] p=(\S+) holm=(\S+)$')
cps = []
for nb, L in (('v2a', A), ('v2b', B)):
    for l in L:
        if not l.startswith('CP '): continue
        m = cre.match(l)
        if not m: print('NOMATCH', l); continue
        g = m.groups()
        fam = g[0]
        cps.append(dict(notebook=nb, family=fam, fraction=float(g[1]), arm=g[2], comparator=g[3], metric=g[4], n_seeds=int(g[5]),
            perseed_diff=float(g[6]), perseed_ci_low=float(g[7]), perseed_ci_high=float(g[8]), perseed_p=float(g[9]), perseed_p_holm=float(g[10]),
            ensemble_diff=float(g[11]), ensemble_ci_low=float(g[12]), ensemble_ci_high=float(g[13]), ensemble_p=float(g[14]), ensemble_p_holm=float(g[15])))
cps = pd.DataFrame(cps)
fx = []
for l in A:
    m = re.match(r'^FX (\S+) (\S+) (AUROC|AUPRC) n=(\d+) effect=(\S+) \[(\S+),(\S+)\] p=(\S+) holm=(\S+)$', l)
    if m:
        g = m.groups(); fx.append(dict(fraction=float(g[0]), effect=g[1], metric=g[2], n_seeds=int(g[3]), effect_size=float(g[4]), ci_low=float(g[5]), ci_high=float(g[6]), p=float(g[7]), p_holm=float(g[8])))
fx = pd.DataFrame(fx)
ps = []
for l in A:
    if l.startswith('PS '):
        p = l.split(); d = dict(arm=p[1], fraction=float(p[2]), perturbation=p[3])
        for kv in p[4:]:
            k, v = kv.split('='); m_, s_, n_ = v.split('|'); d[k + '_mean'] = float(m_); d[k + '_sd'] = float(s_); d['n_seeds'] = int(n_)
        ps.append(d)
ps = pd.DataFrame(ps)
sh = []
for l in B:
    if l.startswith('SH '):
        p = l.split(); d = dict(variable=p[1], type=p[2])
        for kv in p[3:]:
            k, v = kv.split('='); d[k] = v
        sh.append(d)
sh = pd.DataFrame(sh)
for c in ['rateA', 'rateB', 'smd', 'ks']: sh[c] = pd.to_numeric(sh[c], errors='coerce')
hc = pd.DataFrame([dict(features=l.split()[1], hospital_classifier_AUROC=float(l.split('=')[1])) for l in B if l.startswith('HC ')])
fq = []
for l in A:
    if l.startswith('FQ '):
        p = l.split(); d = dict(hospital=p[1], variable=p[2], type=p[3])
        for kv in p[4:]:
            k, v = kv.split('='); d[k] = float(v) if v != 'nan' else None
        fq.append(d)
fq = pd.DataFrame(fq)
store = dict(runs_a=runs_a, runs_b=runs_b, pert=pert, pinfo=pinfo, summ=summ, cps=cps, fx=fx, ps=ps, sh=sh, hc=hc, fq=fq)
pd.to_pickle(store, 'store.pkl')
for k, v in store.items(): print(k, v.shape)
print(runs_a.groupby(['job']).size(), runs_b.groupby(['job','protocol']).size())
print(runs_a[runs_a.job=='main'].groupby('fraction')[['n_train_patients','n_train_septic']].first())
