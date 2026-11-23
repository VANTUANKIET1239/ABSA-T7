"""Figure 4.18: unchanged coordinates, different mean/median cutoff lines."""
import _bootstrap
from common.config import *
from common.data_utils import load
from common.plot_utils import plt,save,ipa_axes,ipa_legend,label_points

def main():
    values=load('25_ipa_overall'); cuts=load('35_ipa_cutoff_comparison').set_index('Cut_Method')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True)
    fig.subplots_adjust(left=.1,right=.98,top=.9,bottom=.24,wspace=.22)
    for ax,method,title in zip(axes,['mean','median'],['(a) Trung bình','(b) Trung vị']):
        ipa_axes(ax,values,title,importance_cut=cuts.loc[method,'Importance_Cut'],performance_cut=cuts.loc[method,'Performance_Cut'],xlim=(55,100),ylim=(0,28),labels=False)
        ax.set_xlabel('Performance (0-100)',fontsize=9)
    axes[1].set_ylabel('')
    for ax in axes: label_points(ax,values)
    ipa_legend(fig,anchor=(.5,.0),columns=4,short_labels=True)
    save(fig,Path(__file__).stem)

if __name__=='__main__': main()
