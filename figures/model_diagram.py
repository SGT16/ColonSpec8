import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw(out):
    fig, ax = plt.subplots(figsize=(7.2, 1.62))
    ax.set(xlim=(0, 10), ylim=(0, 2))
    ax.axis('off')
    stages = [('Girdi','351 bant','Her piksel için'),('Projeksiyon','351 → 8 kanal','1 × 1 evrişim'),('Yerel ağ','Kodlayıcı ve\nkod çözücü','Uzamsal komşuluk'),('Çıktı','2 sınıf puanı','Her piksel için')]
    for i,(header, label, note) in enumerate(stages):
        x = .03+i*2.6
        ax.add_patch(FancyBboxPatch((x,.28),2.13,1.42,boxstyle='round,pad=0.02,rounding_size=0.07',fc='#edf4f1' if i==1 else '#f4f5f5',ec='#55867e' if i==1 else '#bac7c6',lw=1))
        ax.text(x+1.06,1.43,header,ha='center',va='center',fontsize=10,weight='bold',color='#164940')
        ax.text(x+1.06,1.00,label,ha='center',va='center',fontsize=9.6)
        ax.text(x+1.06,.48,note,ha='center',va='center',fontsize=8,color='#364f4b')
        if i<3: ax.annotate('',(x+2.53,1.0),(x+2.13,1.0),arrowprops={'arrowstyle':'->','color':'#304d49','lw':1})
    for ext in ('png','pdf'): fig.savefig(out / ('model.'+ext),dpi=320,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    draw(parser.parse_args().output)
