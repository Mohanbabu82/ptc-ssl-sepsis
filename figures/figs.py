import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
 'font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'savefig.dpi':600})
# ---------- Fig 2 (label efficiency), real 10-seed means/SD
rows={'LightGBM':[(1,.6883,.0546,.0503,.0107),(2,.6873,.0578,.0566,.0123),(5,.7138,.0651,.0663,.0092),(10,.7727,.0303,.0737,.0086),(25,.8147,.0068,.0958,.0063),(100,.8330,.0043,.1166,.0055)],
'Supervised GRU':[(1,.6953,.0343,.0433,.0077),(2,.7062,.0287,.0534,.0097),(5,.7321,.0399,.0692,.0143),(10,.7571,.0385,.0807,.0110),(25,.7933,.0089,.0960,.0129),(100,.8231,.0056,.1279,.0099)],
'Generic SSL':[(1,.7028,.0394,.0442,.0091),(2,.6960,.0425,.0485,.0097),(5,.7287,.0426,.0631,.0126),(10,.7653,.0344,.0775,.0114),(25,.7933,.0094,.0951,.0085),(100,.8245,.0048,.1222,.0085)],
'PTC-SSL':[(1,.7082,.0313,.0513,.0123),(2,.7011,.0419,.0593,.0146),(5,.7372,.0417,.0734,.0141),(10,.7634,.0376,.0824,.0092),(25,.7956,.0092,.0999,.0099),(100,.8216,.0046,.1231,.0093)]}
sty={'LightGBM':dict(color='0.35',ls='-',marker='s'),'Supervised GRU':dict(color='0.55',ls=':',marker='^'),
     'Generic SSL':dict(color='0.55',ls='--',marker='o'),'PTC-SSL':dict(color='#1f5fa8',ls='-',marker='D',lw=1.6)}
fig,ax=plt.subplots(1,2,figsize=(7.16,2.6))
off={'LightGBM':-1.5,'Supervised GRU':-.5,'Generic SSL':.5,'PTC-SSL':1.5}
for k,v in rows.items():
    f=np.array([r[0] for r in v]); xs=f*np.exp(off[k]*0.035)
    for j,(m,s,lab) in enumerate([(1,2,'AUROC'),(3,4,'AUPRC')]):
        y=np.array([r[m] for r in v]); e=np.array([r[s] for r in v])
        st=dict(sty[k]); lw=st.pop('lw',1.0)
        ax[j].errorbar(xs,y,yerr=e,lw=lw,ms=3.5,capsize=1.5,elinewidth=0.6,label=k,**st)
for j,(lab,lo,hi) in enumerate([('Hourly AUROC',0.62,0.85),('Hourly AUPRC',0.03,0.145)]):
    a=ax[j]; a.set_xscale('log'); a.set_xticks([1,2,5,10,25,100]); a.set_xticklabels(['1','2','5','10','25','100'])
    a.set_xlabel('Labelled training patients (%)'); a.set_ylabel(lab); a.set_ylim(lo,hi); a.grid(alpha=.25,lw=.4)
    a.text(0.02,0.96,'(a)' if j==0 else '(b)',transform=a.transAxes,va='top')
ax[1].axhline(0.0159,color='0.7',lw=.5)
ax[0].legend(loc='lower right',frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig('fig/fig2_label_efficiency.png'); plt.close(fig)

# ---------- Fig 3 measurement pattern vs values (10 seeds, Table VII)
cats=['PTC-SSL','Supervised GRU','LightGBM']
d={'Full':[(0.737,0.042,0.763,0.038),(0.732,0.040,0.757,0.038),(0.714,0.065,0.773,0.030)],
   'Values only':[(0.693,0.033,0.727,0.020),(0.675,0.050,0.706,0.034),(0.743,0.017,0.772,0.008)],
   'Measurement pattern only':[(0.756,0.022,0.779,0.015),(0.763,0.009,0.780,0.010),(0.712,0.036,0.748,0.031)]}
cols={'Full':'0.45','Values only':'0.8','Measurement pattern only':'#1f5fa8'}
fig,ax=plt.subplots(1,2,figsize=(3.5,2.5),sharey=True)
w=0.26
for j,b in enumerate(['5%','10%']):
    for i,(k,v) in enumerate(d.items()):
        x=np.arange(3)+(i-1)*w
        m=[t[2*j] for t in v]; s=[t[2*j+1] for t in v]
        ax[j].bar(x,m,w,yerr=s,color=cols[k],edgecolor='k',lw=.4,error_kw=dict(lw=.6,capsize=1.5),label=k)
    ax[j].set_xticks(range(3)); ax[j].set_xticklabels(['PTC-SSL','Sup.\nGRU','Light-\nGBM'],fontsize=7)
    ax[j].set_ylim(0.6,0.82); ax[j].set_title(f'{b} labels',fontsize=8); ax[j].grid(axis='y',alpha=.25,lw=.4)
ax[0].set_ylabel('Hourly AUROC')
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=3,frameon=False,fontsize=6.5,bbox_to_anchor=(0.55,1.0))
fig.tight_layout(rect=(0,0,1,0.9)); fig.savefig('fig/fig3_measurement_pattern.png'); plt.close(fig)

# ---------- Fig 4 transfer (5 seeds, Table XI) + measurement rates (Table XII)
fig,ax=plt.subplots(1,2,figsize=(7.16,2.5),gridspec_kw=dict(width_ratios=[1.1,1]))
arms=['LightGBM','Supervised GRU','Generic SSL','PTC-SSL']
ab={'A→B':[(0.762,0.020),(0.690,0.049),(0.720,0.033),(0.770,0.019)],'B→A':[(0.729,0.016),(0.705,0.023),(0.706,0.030),(0.701,0.023)]}
c4=['0.35','0.6','0.8','#1f5fa8']
for g,(k,v) in enumerate(ab.items()):
    for i,(m,s) in enumerate(v):
        ax[0].bar(g+(i-1.5)*0.2,m,0.2,yerr=s,color=c4[i],edgecolor='k',lw=.4,error_kw=dict(lw=.6,capsize=1.5),label=arms[i] if g==0 else None)
ax[0].set_xticks([0,1]); ax[0].set_xticklabels(['Train A → test B','Train B → test A']); ax[0].set_ylim(0.62,0.82)
ax[0].set_ylabel('Hourly AUROC (10% labels)'); ax[0].legend(frameon=False,fontsize=6.5,loc='upper right',ncol=2); ax[0].grid(axis='y',alpha=.25,lw=.4)
ax[0].text(0.02,0.96,'(a)',transform=ax[0].transAxes,va='top')
vars_=['BaseExcess','HCO3','Chloride','FiO2','pH','PTT','PaCO2','SaO2','Lactate','Glucose','TroponinI']
A=[10.42,8.05,8.32,14.15,11.47,4.85,8.77,4.95,3.43,12.23,0.12]; B=[0.23,0.19,0.62,2.25,2.23,0.97,2.23,1.89,1.88,22.16,1.81]
y=np.arange(len(vars_))
ax[1].barh(y-0.2,A,0.4,color='0.45',edgecolor='k',lw=.3,label='Hospital A')
ax[1].barh(y+0.2,B,0.4,color='#1f5fa8',edgecolor='k',lw=.3,label='Hospital B')
lab={'HCO3':'HCO$_3$','FiO2':'FiO$_2$','PaCO2':'PaCO$_2$','SaO2':'SaO$_2$'}
ax[1].set_yticks(y); ax[1].set_yticklabels([lab.get(v,v) for v in vars_],fontsize=7); ax[1].invert_yaxis()
ax[1].set_xlabel('Patient-hours with a measurement (%)'); ax[1].legend(frameon=False,fontsize=6.5,loc='center right')
ax[1].grid(axis='x',alpha=.25,lw=.4); ax[1].text(0.9,0.97,'(b)',transform=ax[1].transAxes,va='top')
fig.tight_layout(); fig.savefig('fig/fig4_transfer.png'); plt.close(fig)
print('ok')
