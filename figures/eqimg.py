import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import content as C
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'STIXGeneral'})
eqs=re.findall(r'^EQ (.+)$', C.BODY, flags=re.M)
for i,e in enumerate(eqs,1):
    e2=e.replace('\\operatorname{mean}','\\mathrm{mean}').replace('\\,',' \\, ')
    fig=plt.figure(figsize=(0.01,0.01)); fig.text(0,0,f'${e2}$',fontsize=10)
    fig.savefig(f'fig/eq{i}.png',dpi=600,bbox_inches='tight',pad_inches=0.02,transparent=False); plt.close(fig)
print(len(eqs))
