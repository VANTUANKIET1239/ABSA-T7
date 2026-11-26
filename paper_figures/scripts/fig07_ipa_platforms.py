"""All three platform-specific IPA matrices."""
import _bootstrap
from common.config import *
from common.data_utils import load_ipa_by_platform, number
from common.plot_utils import plt,save,ipa_axes,ipa_legend,label_points

def main():
    data=load_ipa_by_platform()
    fig=plt.figure(figsize=(8,6.6))
    grid=fig.add_gridspec(2,4,left=.085,right=.98,top=.94,bottom=.12,wspace=.95,hspace=.42)
    axes=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:]),
          fig.add_subplot(grid[1,1:3])]
    parts=[]
    for ax,p in zip(axes,PLATFORMS):
        sub=data.loc[data.Platform.eq(p)].copy(); parts.append(sub)
        ipa_axes(ax,sub,f'{p} (n = {number(sub.Review_N.iloc[0],0)})',
                 xlim=(40,100),labels=False,marker_size=42)
        ax.set_xlabel('Performance (0-100)',fontsize=9)
        ax.set_ylabel('Importance (%)',fontsize=9)
        ax.set_xticks([40,60,80,100])
    for ax,sub in zip(axes,parts): label_points(ax,sub)
    ipa_legend(fig,anchor=(.5,.015),columns=4)
    save(fig,Path(__file__).stem)

if __name__=='__main__': main()
