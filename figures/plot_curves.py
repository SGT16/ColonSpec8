import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


LOCAL = ['local_only_supervised','local_frozen_projection','local_pca8_supervised']
ARMS = ['learned','random','pca8']
MODELS = LOCAL + ['nnunet2d_supervised','mamba_mt','unet_mt','gmtssa_train_only','hybridsn_supervised']
NAMES = ['ColonSpec8','Sabit rastgele','PCA8','nnU-Net 2B','Mamba-MT','U-Net-MT','GM–TensorSSA','HybridSN']
COLORS = ['#00665b','#bd5200','#476bb0','#8359a0','#687e28','#ad4e70','#656569','#ad8404']
STYLES = ['-','--','-.',':','--','-.',':','--']


def save(fig, directory, name):
    fig.savefig(directory / (name + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.08)
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def plot(source, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    with (source/'roc.csv').open(encoding='utf-8') as f:
        roc=list(csv.DictReader(f))
    fig, axes=plt.subplots(1,2,figsize=(7.2,4.0),constrained_layout=True)
    for ax, data, models, title in zip(axes,['ColonCancerHSI','Borkovits'],[MODELS,ARMS],['A   ColonCancerHSI · piksel','B   Borkovits · doku örneği']):
        ax.plot([0,1],[0,1],color='#aaa',linewidth=.9,zorder=0)
        for i,model in enumerate(models):
            rows=[r for r in roc if r['model']==model and r['dataset']==data]
            ax.plot([0]+[float(r['fpr']) for r in rows],[0]+[float(r['mean_tpr']) for r in rows],STYLES[i],color=COLORS[i],label=NAMES[i],linewidth=1.6)
        ax.set(xlim=(0,1),ylim=(0,1.018),xlabel='Yanlış pozitif oranı',ylabel='Doğru pozitif oranı')
        ax.set_title(title,loc='left',fontsize=10,weight='bold')
        ax.legend(frameon=False,loc='upper center', bbox_to_anchor=(.5,-.18),ncol=2,fontsize=7.4)
        ax.set_aspect('equal')
    save(fig,out,'roc')
    with (source/'loss.csv').open(encoding='utf-8') as f:
        losses=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(7.4,3.05),constrained_layout=True)
    for ax, data, column, title, models in zip(axes,['ColonCancerHSI','ColonCancerHSI','Borkovits'],['training_loss','validation_ce','training_loss'],['A   Eğitim · HSI','B   Doğrulama · HSI','C   Eğitim · FTIR'],[LOCAL,LOCAL,ARMS]):
        for i,model in enumerate(models):
            by_run=defaultdict(list)
            for row in losses:
                if row['dataset']==data and row['model']==model:
                    by_run[(row['seed'],row['fold'])].append((int(row['epoch']),float(row[column])))
            for points in by_run.values():
                points.sort()
                ax.plot(*zip(*points),color=COLORS[i],alpha=.15,linewidth=.55)
            shared=min(map(len,by_run.values()))
            mean=np.mean([[v for _,v in run[:shared]] for run in by_run.values()],axis=0)
            ax.plot(np.arange(1,shared+1),mean,STYLES[i],color=COLORS[i],linewidth=1.6,label=NAMES[i])
        ax.set_title(title,loc='left',fontsize=9.5,weight='bold')
        ax.set(xlabel='Eğitim turu',ylabel='Çapraz entropi + Dice' if ax is axes[0] else 'Çapraz entropi')
        ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
        ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    save(fig,out,'kayip')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    plot(args.source,args.output)
