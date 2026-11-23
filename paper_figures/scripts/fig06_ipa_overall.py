"""Figure 4.14: global IPA using grand-mean cutoffs."""
import _bootstrap
from common.config import *
from common.data_utils import load
from common.plot_utils import plt,save,ipa_axes,ipa_legend

def main():
    data=load('25_ipa_overall')
    fig,ax=plt.subplots(figsize=(6.8,5.1)); fig.subplots_adjust(left=.11,right=.98,top=.94,bottom=.22)
    ipa_axes(ax,data,'',xlim=(55,100),labels=False)
    from common.plot_utils import label_points
    label_points(ax,data); ipa_legend(fig,anchor=(.5,.01))
    save(fig,Path(__file__).stem)

if __name__=='__main__': main()
