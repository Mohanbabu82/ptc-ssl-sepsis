import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':9,'axes.linewidth':0.6,'savefig.dpi':300})
S = pd.read_pickle('../results/store.pkl')
ra, cps, fx, ps = S['runs_a'], S['cps'], S['fx'], S['ps']
BLUE = '#1f5fa8'
names = {'lightgbm':'LightGBM','supervised_gru':'Supervised GRU','generic_ssl':'Generic SSL','ptc_ssl':'PTC-SSL'}
fr = [0.01,0.02,0.05,0.1,0.25,1.0]; frl = ['1%','2%','5%','10%','25%','100%']

# R1: seed spread at each budget (AUROC), main job
m = ra[ra.job=='main']
fig, ax = plt.subplots(figsize=(7.2,3.0))
for i,(k,n) in enumerate(names.items()):
    for j,f in enumerate(fr):
        v = m[(m.arm==k)&(m.fraction==f)].AUROC.values
        x = j + (i-1.5)*0.18
        ax.scatter(np.full(len(v),x)+np.random.default_rng(i*10+j).uniform(-0.03,0.03,len(v)), v, s=9,
                   color=BLUE if k=='ptc_ssl' else ['0.25','0.55','0.75'][min(i,2)], alpha=0.85, label=n if j==0 else None, edgecolor='none')
        ax.plot([x-0.07,x+0.07],[v.mean()]*2,color='k',lw=1)
ax.set_xticks(range(6)); ax.set_xticklabels(frl); ax.set_xlabel('Labelled training patients'); ax.set_ylabel('Hourly AUROC (test set)')
ax.grid(axis='y',alpha=.25,lw=.4); ax.legend(frameon=False,fontsize=8,ncol=4,loc='lower right')
fig.tight_layout(); fig.savefig('R1_seed_spread.png'); plt.close(fig)

# R2: forest plot of PTC-SSL minus comparators (main, per-seed)
c = cps[(cps.family=='main')&(cps.arm=='ptc_ssl')]
fig, axs = plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
comps = [('generic_ssl','Generic SSL'),('supervised_gru','Supervised GRU'),('lightgbm','LightGBM')]
for a,met in zip(axs,['AUROC','AUPRC']):
    y=0; ticks=[]; labs=[]
    for k,n in comps:
        for j,f in enumerate(fr):
            r = c[(c.comparator==k)&(c.metric==met)&(c.fraction==f)].iloc[0]
            col = BLUE if r.perseed_p_holm<0.05 else '0.35'
            a.errorbar(r.perseed_diff, y, xerr=[[r.perseed_diff-r.perseed_ci_low],[r.perseed_ci_high-r.perseed_diff]], fmt='o', ms=3, color=col, elinewidth=0.8, capsize=1.5)
            ticks.append(y); labs.append(f'{n}, {frl[j]}'); y-=1
        y-=0.6
    a.axvline(0,color='k',lw=0.6); a.set_title(f'PTC-SSL minus comparator, {met}',fontsize=9); a.grid(axis='x',alpha=.25,lw=.4)
axs[0].set_yticks(ticks); axs[0].set_yticklabels(labs,fontsize=7)
fig.text(0.99,0.01,'Blue: significant after Holm correction. Bars: 95% CI (hierarchical bootstrap, 10 seeds).',ha='right',fontsize=7)
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig('R2_comparison_forest.png'); plt.close(fig)

# R3: utility and calibration
sm = S['summ']; sm = sm[(sm.job=='main')&(sm.protocol=='pooled')&(sm.notebook=='v2a')]
fig, axs = plt.subplots(1,2,figsize=(7.2,2.8))
for i,(k,n) in enumerate(names.items()):
    d = sm[sm.arm==k].sort_values('fraction')
    st = dict(color=BLUE,lw=1.6,marker='D') if k=='ptc_ssl' else dict(color=['0.3','0.55','0.7'][min(i,2)],lw=1,marker='os^'[min(i,2)],ls=['-',':','--'][min(i,2)])
    axs[0].errorbar(d.fraction*100, d.utility_mean, yerr=d.utility_sd, ms=3.5, capsize=1.5, elinewidth=.6, label=n, **st)
axs[0].set_xscale('log'); axs[0].set_xticks([1,2,5,10,25,100]); axs[0].set_xticklabels(frl); axs[0].set_xlabel('Labelled training patients'); axs[0].set_ylabel('Normalised utility score'); axs[0].legend(frameon=False,fontsize=7); axs[0].grid(alpha=.25,lw=.4)
axs[0].text(0.02,0.96,'(a)',transform=axs[0].transAxes,va='top')
x = np.arange(4); w=0.38
d5 = sm[sm.fraction==0.05].set_index('arm').loc[list(names)]
axs[1].bar(x-w/2, d5.ECE_mean, w, yerr=d5.ECE_sd, color='0.7', edgecolor='k', lw=.4, error_kw=dict(lw=.6,capsize=1.5), label='Before recalibration')
axs[1].bar(x+w/2, d5.ECE_cal_mean, w, yerr=d5.ECE_cal_sd, color=BLUE, edgecolor='k', lw=.4, error_kw=dict(lw=.6,capsize=1.5), label='After validation-only Platt scaling')
axs[1].set_xticks(x); axs[1].set_xticklabels(['LightGBM','Sup. GRU','Generic\nSSL','PTC-SSL'],fontsize=8); axs[1].set_ylabel('Expected calibration error (5% labels)'); axs[1].legend(frameon=False,fontsize=7); axs[1].grid(axis='y',alpha=.25,lw=.4)
axs[1].text(0.02,0.96,'(b)',transform=axs[1].transAxes,va='top')
fig.tight_layout(); fig.savefig('R3_utility_calibration.png'); plt.close(fig)

# R4: factorial effects forest
lab = {'O':'Organ masking (O)','P':'Physiological loss (P)','T':'Temporal loss (T)','OxP':'O × P','OxT':'O × T','PxT':'P × T','OxPxT':'O × P × T'}
fig, axs = plt.subplots(1,2,figsize=(7.2,2.6),sharey=True)
for a,met in zip(axs,['AUROC','AUPRC']):
    for j,f in enumerate([0.05,0.1]):
        d = fx[(fx.metric==met)&(fx.fraction==f)].set_index('effect').loc[list(lab)]
        y = -np.arange(len(d)) + (0.15 if j==0 else -0.15)
        cols = [BLUE if p<0.05 else '0.4' for p in d.p_holm]
        for yi,(e,r),cc in zip(y,d.iterrows(),cols):
            a.errorbar(r.effect_size, yi, xerr=[[r.effect_size-r.ci_low],[r.ci_high-r.effect_size]], fmt='o' if j==0 else 's', ms=3.5, color=cc, mfc=cc if j==0 else 'white', elinewidth=.8, capsize=1.5, label=('5% labels' if j==0 else '10% labels') if e=='O' else None)
    a.axvline(0,color='k',lw=.6); a.set_title(f'Factorial effect on {met}',fontsize=9); a.grid(axis='x',alpha=.25,lw=.4)
axs[0].set_yticks(-np.arange(7)); axs[0].set_yticklabels(list(lab.values()),fontsize=8); axs[1].legend(frameon=False,fontsize=7,loc='lower right')
fig.text(0.99,0.01,'Blue: Holm-adjusted p < 0.05. Effect = contrast/4 with ±1 coding; 95% CI from hierarchical bootstrap (10 seeds).',ha='right',fontsize=7)
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig('R4_factorial_effects.png'); plt.close(fig)

# R5: perturbations
pl = {'drop_labs_30':'30% of labs removed','drop_vitals_30':'30% of vitals removed','no_lactate':'Lactate removed','labs_max_1_per_24h':'Labs capped at 1 per 24 h'}
arms5 = [('lightgbm','LightGBM'),('supervised_gru','Supervised GRU'),('generic_ssl','Generic SSL'),('ptc_ssl','PTC-SSL')]
fig, axs = plt.subplots(1,2,figsize=(7.2,2.7),sharey=True)
for a,f in zip(axs,[0.05,1.0]):
    for i,(k,n) in enumerate(arms5):
        d = ps[(ps.arm==k)&(ps.fraction==f)].set_index('perturbation').loc[list(pl)]
        a.bar(np.arange(4)+(i-1.5)*0.2, d.dAUROC_mean, 0.2, yerr=d.dAUROC_sd, color=BLUE if k=='ptc_ssl' else ['0.3','0.55','0.8'][i], edgecolor='k', lw=.3, error_kw=dict(lw=.5,capsize=1), label=n)
    a.set_xticks(range(4)); a.set_xticklabels([v.replace(' removed','\nremoved').replace(' capped','\ncapped') for v in pl.values()],fontsize=7)
    a.axhline(0,color='k',lw=.6); a.set_title(f'{int(f*100)}% labels',fontsize=9); a.grid(axis='y',alpha=.25,lw=.4)
axs[0].set_ylabel('Change in hourly AUROC')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=4,frameon=False,fontsize=7.5)
fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig('R5_perturbations.png'); plt.close(fig)

# R6: sensitivity forest
s = cps[(cps.family=='pooled sensitivity')&(cps.metric=='AUROC')].copy()
order = ['sens_lp_0','sens_lp_0.01','sens_lp_0.05','sens_lp_0.5','sens_lp_1','sens_lt_0','sens_lt_0.01','sens_lt_0.05','sens_lt_0.5','sens_lt_1','sens_hidden_64','sens_hidden_128','sens_hidden_256','sens_block_4','sens_block_12','sens_block_16','sens_mask_random','sens_mask_variable']
print(sorted(s.arm.unique())); print(s.comparator.unique())
s = s.set_index('arm').loc[order]
def nm(o):
    t=o.replace('sens_','')
    return (t.replace('lp_',r'$\lambda_{phys}$ = ').replace('lt_',r'$\lambda_{temp}$ = ').replace('hidden_','hidden size ').replace('block_','block ').replace('mask_','masking: '))+(' h' if 'block' in t else '')
fig, ax = plt.subplots(figsize=(5.0,4.2))
for yi,(o,r) in enumerate(s.iterrows()):
    cc = BLUE if r.perseed_p_holm<0.05 else '0.35'
    ax.errorbar(r.perseed_diff, -yi, xerr=[[r.perseed_diff-r.perseed_ci_low],[r.perseed_ci_high-r.perseed_diff]], fmt='o', ms=3.5, color=cc, elinewidth=.8, capsize=1.5)
ax.set_yticks(-np.arange(len(s))); ax.set_yticklabels([nm(o) for o in s.index],fontsize=8); ax.axvline(0,color='k',lw=.6)
ax.set_xlabel('Variant minus default, hourly AUROC (5% labels)'); ax.grid(axis='x',alpha=.25,lw=.4)
fig.tight_layout(); fig.savefig('R6_sensitivity.png'); plt.close(fig)
print('ok')
