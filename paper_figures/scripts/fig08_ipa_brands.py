"""Figure 4.16: all five brands, one independently reproducible multi-panel figure."""
import _bootstrap
from common.config import *
from common.data_utils import load,number
from common.plot_utils import plt,save,ipa_axes,ipa_legend,label_points

def main():
    data=load('31_ipa_brand')
    fig=plt.figure(figsize=(8,9.4))
    grid=fig.add_gridspec(3,4,left=.085,right=.98,top=.95,bottom=.09,wspace=.95,hspace=.35)
    axes=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:]),
          fig.add_subplot(grid[1,:2]),fig.add_subplot(grid[1,2:]),
          fig.add_subplot(grid[2,1:3])]
    parts=[]
    for ax,brand in zip(axes,BRANDS):
        sub=data.loc[data.Brand.eq(brand)].copy(); parts.append(sub)
        ipa_axes(ax,sub,f'{brand} (n = {number(sub.Review_N.iloc[0],0)})',
                 xlim=(35,100),labels=False,quadrant_labels=True,marker_size=42)
        ax.set_xlabel('Performance (0-100)',fontsize=9)
        ax.set_ylabel('Importance (%)',fontsize=9)
        ax.set_xticks([40,60,80,100])
    for ax,sub in zip(axes,parts): label_points(ax,sub)
    ipa_legend(fig,anchor=(.5,.015),columns=4)
    save(fig,Path(__file__).stem)

if __name__=='__main__': main()
