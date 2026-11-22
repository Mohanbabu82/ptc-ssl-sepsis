import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':7.5,'savefig.dpi':600})
fig,ax=plt.subplots(figsize=(7.16,3.0)); ax.set_xlim(0,100); ax.set_ylim(0,42); ax.axis('off')
def box(x,y,w,h,title,body,fc='white',ec='0.2',bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=0.8',fc=fc,ec=ec,lw=0.7))
    ax.text(x+w/2,y+h-1.6,title,ha='center',va='top',fontweight='bold' if bold else 'normal',fontsize=7.5)
    nl=title.count('\n'); ax.text(x+w/2,y+h-4.6-2.6*nl,body,ha='center',va='top',fontsize=5.9,linespacing=1.25)
def arr(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=0.7,color='0.2',mutation_scale=7))
blue='#e3edf8'
# top row: pipeline
box(0.5,27,17,14,'Data verification','40,336 records,\nMD5-checked; units of\nlactate, Mg, FiO$_2$ fixed;\n6 constraints validated')
box(21,27,17,14,'Leakage-audited\npreprocessing','patient-level split,\ntrain-only scaling,\ncausal LOCF; A→B, B→A')
box(41.5,27,17,14,'Causal windows','24 h × 103 channels:\nvalues, observation mask,\ntime since observation,\npadding flag + carry-in')
box(62,27,17,14,'Fine-tuning','nested label budgets\n1, 2, 5, 10, 25, 100%\nweighted BCE, early stop\non budgeted validation')
box(82.5,27,17,14,'Evaluation','AUROC, AUPRC, utility,\nBrier, ECE (val-only Platt);\n10 seeds, hierarchical\nbootstrap, Holm')
for x in (17.8,38.3,58.8,79.3): arr(x+0.2,34,x+3.4,34)
# bottom: PTC-SSL pretraining
ax.add_patch(FancyBboxPatch((8,1),76,22,boxstyle='round,pad=0.3,rounding_size=0.8',fc='#f6f8fb',ec='#1f5fa8',lw=0.8,ls='--'))
ax.text(9.2,21.6,'PTC-SSL pretraining (unlabelled training patients)',fontweight='bold',fontsize=7.5,color='#1f5fa8',va='top')
box(10,3,16,15,'(O) Organ-system\nblock masking','hide 1 of 8 systems\nfor an 8 h block before\nLOCF and Δt are built',fc=blue)
box(30,3,13,15,'GRU encoder','hidden 96\n+ static features\n→ representation z',fc='white')
box(47,10.5,17,8,'Reconstruction','L$_{rec}$ on hidden\nobserved entries',fc='white',bold=True)
box(47,1.5,17,8,'(P) Physiological loss','L$_{phys}$ on validated\nconstraints C1–C6',fc=blue)
box(67.5,3,15,15,'(T) Temporal\nconsistency','L$_{temp}$: predict\nsg(z$_{r-1}$) from z$_r$\n(adjacent windows)',fc=blue)
arr(26.4,10.5,29.6,10.5); arr(43.3,12,46.6,14.5); arr(43.3,9,46.6,5.5); arr(64.3,10.5,67.1,10.5)
arr(50.5,26.7,40,18.6)  # windows -> pretraining
ax.text(48.5,24.2,'same windows',fontsize=6,style='italic',color='0.3')
arr(75,18.4,70.5,26.6)
ax.text(73.5,23.5,'pretrained encoder',fontsize=6,style='italic',color='0.3')
fig.savefig('fig/fig1_pipeline.png',bbox_inches='tight'); print('ok')
