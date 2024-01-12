"""
This script is used in the context of the GLHMM paper, doi:
https://doi.org/10.48550/arXiv.2312.07151
This script concerns plotting performance of stochastic and non-stochastic training on data from the HCP 
and MEGUK datasets as a function of batch-size. The datasets are described in Van Essen et al.(2013) and 
Hunt et al.(2016), and publicly available.
See papers: https://pubmed.ncbi.nlm.nih.gov/23684880/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5127325/

!!!Note!!!
This script plots data resulting from running run_glhmm.sh. This script should be ran first. The script
further assumes run_glhmm.sh was run without modifications, speficially that the number of batch-sizes is 9
and the number of runs per batch-size is 10. To indicate which dataset to plot, comment/uncomment line 41.

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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
import re


## Set dataset

dataset='HCP'
#dataset='MRC'



if dataset=='HCP':
    ## Load HCP results
    in_dir = '<data path>'

    #Define settings
    n_batches   =   [0,5,10,20,100,200,500,800,1003]
    n_runs         =   range(1,11)
    
elif dataset=='MRC':
    ## Load MRC results
    in_dir = '<data path>'

    #Define settings
    n_batches   =   [0,1,2,5,10,20,40,50,55]
    n_runs      =   range(1,11)

#create dataframes
free_energy =   pd.DataFrame(columns=['FE', 'batch-size','run'])
performance =   pd.DataFrame(columns=['last-cycle','FLOPS', 'batch-size','run'])

#read data
for batch_size in n_batches:
    for run in n_runs:
        with open(in_dir + 'FE' + str(batch_size) + '_FR0.8_run' + str(run) + '_output.txt') as f:
            lines = f.read().splitlines()
        
        lines = [float(line) for line in lines]

        with open(in_dir + 'n' + str(batch_size) + '_FR0.8_run' + str(run) + '_report.txt') as f:
            txt = f.read().splitlines()

        txt=' '.join(txt)
        last_cycle=int(re.findall(r'\d+', txt[txt.rfind("Cycle"):txt.rfind("Cycle")+10])[0])
        flops=int(''.join(re.findall(r'\d+', txt[txt.rfind("fp_ret_sse_avx_ops.all")-25:txt.rfind("fp_ret_sse_avx_ops.all")])))

        d1={'FE':lines, 'batch-size':batch_size,'run':run}
        d2={'last-cycle':last_cycle, 'FLOPS': flops,'batch-size':batch_size,'run':run}

        df=pd.DataFrame(data=d1)
        pf=pd.DataFrame(data=d2,index=[0])
        free_energy =   pd.concat([free_energy,df])
        performance =   pd.concat([performance,pf])

#define dataframe datatypes
free_energy = free_energy.astype({'FE':float, 'batch-size':int, 'run':int})
performance = performance.astype({'last-cycle':int,'FLOPS':int, 'batch-size':int, 'run':int})

#find min and max FE for plot scaling
min_FE=min(free_energy.get("FE"))
max_FE=max(free_energy.get("FE"))


if dataset=='HCP':
    ## Plot main HCP figure; FE (range limited) over log-scaled cycles and FLOPs per cycle.
    #Redefine max_FE to window results.
    max_FE=min_FE+1e6

    # Open figure with subplots
    fig,ax=plt.subplots(1,2,dpi =200, figsize=(10,4))
    ax[0].grid(True)
    ax[1].grid(True)

    #set font size
    sz=14

    #adjust subplot positions
    box1=ax[1].get_position()
    box1.x0=box1.x0+0.03
    box1.x1=box1.x1+0.03
    ax[1].set_position(box1)

    #define color map
    clr=matplotlib.colormaps['viridis'].resampled(8)

    ##Line plotting
    #plot log scale plot
    idx=0
    for batch in range(-8,1):
        ax[0].set(ylim=(min_FE-1e4,max_FE+1e4),xlim=(1,3001))

        ax[0].set_xlabel('no. of cycles', fontsize=sz)
        ax[0].set_ylabel('Free energy', fontsize=sz)
        
        for run in n_runs:
            line=free_energy[(free_energy['batch-size']==n_batches[batch]) & (free_energy['run']==run)]
            stats=performance[(performance['batch-size']==n_batches[batch]) & (performance['run']==run)]
            if batch==0:
                ax[0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'),
                    label='run ' + str(run),
                    color=(0.6353, 0.0784, 0.1843))
            else:
                ax[0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'), 
                    label='run ' + str(run),
                    color=clr(7-idx))
            ax[0].set_xscale('log')

        idx+=1

    ##Bar plotting
    #define empty variables
    stats=pd.DataFrame()
    labels = [1] * 9
    sct = [1] * 9
    idx=0

    #convert dataframe to array
    for n in range(-8,1):
        batch_size=n_batches[n]

        tmp=performance[(performance['batch-size']==batch_size)]
        tmp=tmp['FLOPS']/(tmp['last-cycle']+125)
        stats[n]=tmp

        #set labels for legend
        if batch_size==0:
            labels[idx] = 'VI'
        else:
            labels[idx] = 'SVI('+str(batch_size)+')'

        idx+=1

    #plot bars
    for n in range (1,10):
        if n==9:
            ax[1].bar(n,np.mean(stats[n-9]),color=(0.6353, 0.0784, 0.1843), alpha=0.2)
        else:
            ax[1].bar(n,np.mean(stats[n-9]),color=clr(8-n),alpha=0.2)

    #plot scatter points
    for n in range(0,9):
        if n==8: #if non-stochastic
            sct[n]=ax[1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=(0.6353, 0.0784, 0.1843),zorder=2, s=15)
        else:
            sct[n]=ax[1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=clr(7-n),zorder=2, s=15)

    #set text values
    ax[0].text(-0.23, 1.1,'A',transform=ax[0].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
    ax[1].text(-0.17, 1.1,'B',transform=ax[1].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')

    ax[1].set_xticks(range(1,10),labels=labels,rotation=90, fontsize=sz)
    ax[1].set_ylabel('FLOPs per cycle', fontsize=sz)

    fig.legend(sct,labels,bbox_to_anchor=(-0.025,0.35,0.5,0.5), fontsize=8)

    plt.show()

    ## Plot supplementary figure HCP data; non-range limited FE over log-sclaed cycles, FE over cycles, Total FLOPs.
    # Open figure with subplots
    fig,ax=plt.subplots(2,2,dpi =200, figsize=(10,8))
    ax[0,0].grid(True)
    ax[0,1].set_visible(False)
    ax[1,0].grid(True)
    ax[1,1].grid(True)

    #set font size
    sz=14

    #adjust subplot positions
    box1=ax[1,0].get_position()
    box1.y0=box1.y0-0.03
    box1.y1=box1.y1-0.03
    ax[1,0].set_position(box1)

    box2=ax[1,1].get_position()
    box2.x0=box2.x0+0.03
    box2.x1=box2.x1+0.03
    box2.y0=box2.y0-0.03
    box2.y1=box2.y1-0.03
    ax[1,1].set_position(box2)

    #define color map
    clr=matplotlib.colormaps['viridis'].resampled(8)

    ##Line plotting
    #plot log scale plot
    idx=0
    for batch in range(-8,1):
        ax[0,0].set(ylim=(min_FE-1e4,max_FE+1e4),xlim=(1,3001))

        ax[0,0].set_xlabel('no. of cycles', fontsize=sz)
        ax[0,0].set_ylabel('Free energy', fontsize=sz)
        
        for run in n_runs:
            line=free_energy[(free_energy['batch-size']==n_batches[batch]) & (free_energy['run']==run)]
            stats=performance[(performance['batch-size']==n_batches[batch]) & (performance['run']==run)]
            if batch==0:
                ax[0,0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'),
                    label='run ' + str(run),
                    color=(0.6353, 0.0784, 0.1843))
            else:
                ax[0,0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'), 
                    label='run ' + str(run),
                    color=clr(7-idx))
            ax[0,0].set_xscale('log')

        idx+=1
        
    #plot normal sacle plot
    idx=0
    for batch in range(-8,1):
        ax[1,0].set(xlim=(min_FE-1e4,max_FE+1e4),ylim=(1,3001))

        ax[1,0].set_ylabel('no. of cycles', fontsize=sz)
        ax[1,0].set_xlabel('Free energy', fontsize=sz)

        for run in n_runs:
            line=free_energy[(free_energy['batch-size']==n_batches[batch]) & (free_energy['run']==run)]
            stats=performance[(performance['batch-size']==n_batches[batch]) & (performance['run']==run)]
            if batch==0:
                ax[1,0].plot(line.get('FE'),
                    range(1,stats.get('last-cycle')[0]+1),
                    label='run ' + str(run),
                    color=(0.6353, 0.0784, 0.1843))
            else:
                ax[1,0].plot(line.get('FE'),
                    range(1,stats.get('last-cycle')[0]+1), 
                    label='run ' + str(run),
                    color=clr(7-idx))
        idx+=1

    ##Bar plotting
    #define empty variables
    stats=pd.DataFrame()
    labels = [1] * 9
    sct = [1] * 9
    idx=0

    #convert dataframe to array
    for n in range(-8,1):
        batch_size=n_batches[n]

        tmp=performance[(performance['batch-size']==batch_size)]
        stats[n]=tmp['FLOPS']

        #set labels for legend
        if batch_size==0:
            labels[idx] = 'VI'
        else:
            labels[idx] = 'SVI('+str(batch_size)+')'

        idx+=1

    #Plot bars
    for n in range (1,10):
        if n==9:
            ax[1,1].bar(n,np.mean(stats[n-9]),color=(0.6353, 0.0784, 0.1843), alpha=0.2)
        else:
            ax[1,1].bar(n,np.mean(stats[n-9]),color=clr(8-n),alpha=0.2)

    #plot scatter points
    for n in range(0,9):
        if n==8: #if non-stochastic
            sct[n]=ax[1,1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=(0.6353, 0.0784, 0.1843),zorder=2, s=15)
        else:
            sct[n]=ax[1,1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=clr(7-n),zorder=2, s=15)

    #set text values
    ax[0,0].text(-0.23, 1.1,'A',transform=ax[0,0].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
    ax[1,0].text(-0.23, 1.1,'B',transform=ax[1,0].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
    ax[1,1].text(-0.17, 1.1,'C',transform=ax[1,1].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')

    ax[1,1].set_xticks(range(1,10),labels=labels,rotation=90, fontsize=sz)
    ax[1,1].set_ylabel('Total FLOPs', fontsize=sz)

    fig.legend(sct,labels,bbox_to_anchor=(-0.025,0.35,0.5,0.5), fontsize=8)

    plt.show()

elif dataset=='MRC':
    ## Plot supplementary figure MRC data; FE over log-sclaed cycles, FLOPs per cycle, FE over cycles, Total FLOPs.
    # Open figure with subplots
    fig,ax=plt.subplots(2,2,dpi =200, figsize=(10,8))
    ax[0,0].grid(True)
    ax[0,1].grid(True)
    ax[1,0].grid(True)
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
    clr=matplotlib.colormaps['viridis'].resampled(8)


    ##Line plotting
    #plot log scale plot
    idx=0
    for batch in range(-8,1):
        ax[0,0].set(ylim=(min_FE-1e4,max_FE+1e4),xlim=(1,3001))

        ax[0,0].set_xlabel('no. of cycles', fontsize=sz)
        ax[0,0].set_ylabel('Free energy', fontsize=sz)
        
        for run in n_runs:
            line=free_energy[(free_energy['batch-size']==n_batches[batch]) & (free_energy['run']==run)]
            stats=performance[(performance['batch-size']==n_batches[batch]) & (performance['run']==run)]
            if batch==0:
                ax[0,0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'),
                    label='run ' + str(run),
                    color=(0.6353, 0.0784, 0.1843))
            else:
                ax[0,0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'), 
                    label='run ' + str(run),
                    color=clr(7-idx))
            ax[0,0].set_xscale('log')

        idx+=1
        
    #plot normal sacle plot
    idx=0
    for batch in range(-8,1):
        ax[1,0].set(ylim=(min_FE-1e4,max_FE+1e4),xlim=(1,3001))

        ax[1,0].set_xlabel('no. of cycles', fontsize=sz)
        ax[1,0].set_ylabel('Free energy', fontsize=sz)

        for run in n_runs:
            line=free_energy[(free_energy['batch-size']==n_batches[batch]) & (free_energy['run']==run)]
            stats=performance[(performance['batch-size']==n_batches[batch]) & (performance['run']==run)]
            if batch==0:
                ax[1,0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'),
                    label='run ' + str(run),
                    color=(0.6353, 0.0784, 0.1843))
            else:
                ax[1,0].plot(range(1,stats.get('last-cycle')[0]+1),
                    line.get('FE'), 
                    label='run ' + str(run),
                    color=clr(7-idx))
        idx+=1


    ##Bar plotting
    #define empty variables
    stats=pd.DataFrame()
    labels = [1] * 9
    sct = [1] * 9
    idx=0

    #convert dataframe to array
    for n in range(-8,1):
        batch_size=n_batches[n]

        tmp=performance[(performance['batch-size']==batch_size)]
        tmp=tmp['FLOPS']/(tmp['last-cycle']+125)
        stats[n]=tmp

        #set labels for legend
        if batch_size==0:
            labels[idx] = 'VI'
        else:
            labels[idx] = 'SVI('+str(batch_size)+')'

        idx+=1

    #Plot bars FLOPs per cycle
    for n in range (1,10):
        if n==9:
            ax[0,1].bar(n,np.mean(stats[n-9]),color=(0.6353, 0.0784, 0.1843), alpha=0.2)
        else:
            ax[0,1].bar(n,np.mean(stats[n-9]),color=clr(8-n),alpha=0.2)

    #plot scatter points
    for n in range(0,9):
        if n==8: #if non-stochastic
            sct[n]=ax[0,1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=(0.6353, 0.0784, 0.1843),zorder=2, s=15)
        else:
            sct[n]=ax[0,1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=clr(7-n),zorder=2, s=15)

    ax[0,1].set_xticks(range(1,10),labels='',rotation=90)
    ax[0,1].set_ylabel('FLOPs per cycle', fontsize=sz)


    #define empty variables
    stats=pd.DataFrame()
    sct = [1] * 9

    #convert dataframe to array
    for n in range(-8,1):
        batch_size=n_batches[n]

        tmp=performance[(performance['batch-size']==batch_size)]
        stats[n]=tmp['FLOPS']
        

    #Plot bars Total FLOPs
    for n in range (1,10):
        if n==9:
            ax[1,1].bar(n,np.mean(stats[n-9]),color=(0.6353, 0.0784, 0.1843), alpha=0.2)
        else:
            ax[1,1].bar(n,np.mean(stats[n-9]),color=clr(8-n),alpha=0.2)

    #plot scatter points
    for n in range(0,9):
        if n==8: #if non-stochastic
            sct[n]=ax[1,1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=(0.6353, 0.0784, 0.1843),zorder=2, s=15)
        else:
            sct[n]=ax[1,1].scatter(np.repeat(n+1,10)+np.random.uniform(-0.3,0.3,10),stats[n-8],color=clr(7-n),zorder=2, s=15)


    #set text values
    ax[0,0].text(-0.23, 1.1,'A',transform=ax[0,0].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
    ax[0,1].text(-0.17, 1.1,'B',transform=ax[0,1].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
    ax[1,0].text(-0.23, 1.1,'C',transform=ax[1,0].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')
    ax[1,1].text(-0.17, 1.1,'D',transform=ax[1,1].transAxes,fontsize=sz+2,fontweight='bold',va='top', ha='right')

    ax[1,1].set_xticks(range(1,10),labels=labels,rotation=90, fontsize=sz)
    ax[1,1].set_ylabel('Total FLOPs', fontsize=sz)

    fig.legend(sct,labels,bbox_to_anchor=(-0.025,0.35,0.5,0.5), fontsize=8)

    plt.show()
