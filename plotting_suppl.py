"""
This script is used in the context of the GLHMM paper, doi:
https://doi.org/10.48550/arXiv.2312.07151
This script concerns plotting performance of stochastic and non-stochastic training on data from the HCP 
and MEGUK datasets as a function of forgetting rate. The datasets are described in Van Essen et al.(2013) 
and Hunt et al.(2016), and publicly available.
See papers: https://pubmed.ncbi.nlm.nih.gov/23684880/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5127325/

!!!Note!!!
This script plots data resulting from running run_glhmm.sh. This script should be ran first. The script
further assumes run_glhmm.sh was run without modifications, speficially that the number of forgetting rates 
is 5 and the number of runs per forgetting rate is 10. To indicate which dataset to plot, comment/uncomment line 42.

Requirements:
- GLHMM
The script assumes you have already correctly installed version 0.1.13 of the GLHMM toolbox.
You can download the toolbox using "pip install --user git+https://github.com/vidaurre/glhmm".
See documentation at: https://glhmm.readthedocs.io/en/latest/index.html

Other requirements to run this script:
- numpy
- matplotlib
- pandas

Author: Lenno Ruijters
email: au710840@uni.au.dk
Date: 13/12/2023
"""

## Import packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from seaborn import objects as so

## Set dataset

dataset='HCP'
#dataset='MRC'



if dataset=='HCP':
    ## Load HCP results
    in_dir = '/media/8.1/results/lenno/HMM/HCP/'

    batch_size='200'

elif dataset=='MRC':
    ## Load MRC results
    in_dir = '<data path>'

    batch_size='10'

#Define settings
n_fr   =   [0.5, 0.6, 0.7, 0.8, 0.9]
n_runs      =   range(1,11)

#create dataframes
free_energy =   pd.DataFrame(columns=['FE', 'FR','run'])
performance =   pd.DataFrame(columns=['last-cycle','FLOPS', 'FR','run'])

#read data
for fr in n_fr:
    for run in n_runs:
        with open(in_dir + 'FE' + batch_size + '_FR' + str(fr) + '_run' + str(run) + '_output.txt') as f:
            lines = f.read().splitlines()
        
        lines = [float(line) for line in lines]

        with open(in_dir + 'n' + batch_size + '_FR' + str(fr) + '_run' + str(run) + '_report.txt') as f:
            txt = f.read().splitlines()

        txt=' '.join(txt)
        last_cycle=int(re.findall(r'\d+', txt[txt.rfind("Cycle"):txt.rfind("Cycle")+10])[0])
        flops=int(''.join(re.findall(r'\d+', txt[txt.rfind("fp_ret_sse_avx_ops.all")-24:txt.rfind("fp_ret_sse_avx_ops.all")])))

        d1={'FE':lines, 'FR':fr,'run':run}
        d2={'last-cycle':last_cycle, 'FLOPS': flops,'FR':fr,'run':run}

        df=pd.DataFrame(data=d1)
        pf=pd.DataFrame(data=d2,index=[0])
        free_energy =   pd.concat([free_energy,df])
        performance =   pd.concat([performance,pf])

#define dataframe datatypes
free_energy = free_energy.astype({'FE':float, 'FR':float, 'run':int})
performance = performance.astype({'last-cycle':int,'FLOPS':int, 'FR':float, 'run':int})

#find min and max FE for plot scaling
max_FE=max(free_energy.get("FE"))
min_FE=min(free_energy.get("FE"))


## Open figure with subplots
fig,ax=plt.subplots(2,2,dpi =200, figsize=(10,8))
ax[0,0].grid(True)
ax[1,0].grid(True)
ax[0,1].grid(True)
ax[1,1].grid(True)

#set font size
sz=14

#adjust subplot positions
box=ax[0,1].get_position()
box.x0=box.x0+0.03
box.x1=box.x1+0.03
ax[0,1].set_position(box)

box=ax[1,0].get_position()
box.y0=box.y0-0.03
box.y1=box.y1-0.03
ax[1,0].set_position(box)

box=ax[1,1].get_position()
box.x0=box.x0+0.03
box.x1=box.x1+0.03
box.y0=box.y0-0.03
box.y1=box.y1-0.03
ax[1,1].set_position(box)


#define color map
clr=matplotlib.colormaps['viridis'].resampled(5)

##Line plotting
#plot log scale plot
for batch in range(0,5):
    ax[0,0].set(ylim=(min_FE-1e4,max_FE+1e4))

    ax[0,0].set_ylabel('Free energy', fontsize=sz)
    ax[0,0].set_xlabel('no. of cycle', fontsize=sz)
    
    for run in n_runs:
        line=free_energy[(free_energy['FR']==n_fr[batch]) & (free_energy['run']==run)]
        stats=performance[(performance['FR']==n_fr[batch]) & (performance['run']==run)]

        ax[0,0].plot(range(stats.get('last-cycle')[0]-len(line.get('FE')),stats.get('last-cycle')[0]),
            line.get('FE'), 
            label='run ' + str(run),
            color=clr(batch))
        ax[0,0].set_xscale('log')


#plot normal scale line plot
for batch in range(0,5):
    ax[1,0].set(xlim=(min_FE-1e4,max_FE+1e4))

    ax[1,0].set_ylabel('no. of cycles', fontsize=sz)
    ax[1,0].set_xlabel('Free energy', fontsize=sz)
    
    for run in n_runs:
        line=free_energy[(free_energy['FR']==n_fr[batch]) & (free_energy['run']==run)]
        stats=performance[(performance['FR']==n_fr[batch]) & (performance['run']==run)]

        ax[1,0].plot(line.get('FE'), 
            range(stats.get('last-cycle')[0]-len(line.get('FE')),stats.get('last-cycle')[0]),
            label='run ' + str(run),
            color=clr(batch))


##Violin plotting
#define empty variables
stats=pd.DataFrame()
labels = [1] * 5
sct = [1] * 5

#convert dataframe to array Total FLOPS
for n in range(0,5):
    fr=n_fr[n]

    tmp=performance[(performance['FR']==fr)]
    stats[n]=tmp[['FLOPS']]

#plot violinplot total FLOPS
idx=0
violins=ax[1,1].violinplot(stats,showmeans=False,showmedians=False,showextrema=False)
for pc in violins['bodies']:
    pc.set_facecolor(clr(idx))
    pc.set_edgecolor('black')
    pc.set_alpha(0.2)
    idx+=1

#plot scatter points
for n in range(0,5):
    sct[n]=ax[1,1].scatter(np.repeat(n+1,10),stats[n],color=clr(n),zorder=2)

#set figure values
ax[0,1].set_xticks(range(1,6),labels='',rotation=90)
ax[1,1].set_ylabel('Total FLOPs', fontsize=sz)


#convert dataframe to array FLOPs per cycle
for n in range(0,5):
    fr=n_fr[n]

    tmp=performance[(performance['FR']==fr)]
    tmp=tmp['FLOPS']/(tmp['last-cycle']+125)
    stats[n]=tmp
    labels[n] = 'fr = '+str(fr)

#Plot violinplot FLOPs per cycle
idx=0
violins=ax[0,1].violinplot(stats,showmeans=False,showmedians=False,showextrema=False)
for pc in violins['bodies']:
    pc.set_facecolor(clr(idx))
    pc.set_edgecolor('black')
    pc.set_alpha(0.2)
    idx+=1

#plot scatter points
for n in range(0,5):
    sct[n]=ax[0,1].scatter(np.repeat(n+1,10),stats[n],color=clr(n),zorder=2)


#set text values
ax[0,0].text(-0.23, 1.1,'A',transform=ax[0,0].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
ax[0,1].text(-0.17, 1.1,'B',transform=ax[0,1].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
ax[1,0].text(-0.23, 1.1,'C',transform=ax[1,0].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
ax[1,1].text(-0.17, 1.1,'D',transform=ax[1,1].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')

ax[1,1].set_xticks(range(1,6),labels=labels,rotation=90, fontsize=sz)
ax[0,1].set_ylabel('FLOPs per cycle', fontsize=sz)

fig.legend(sct,labels,bbox_to_anchor=(0,0.35,0.5,0.5), fontsize=8)

plt.show()
