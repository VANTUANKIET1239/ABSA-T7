"""Normalized mean absolute SHAP importance versus mention prevalence."""
import _bootstrap
from common.config import *
from common.data_utils import load_shap_aspect_summary, sentiment_counts, number
from common.plot_utils import plt,save,label_points,decimal_axes

def main():
    data=load_shap_aspect_summary().sort_values('Importance_Pct')
    mentions=sentiment_counts().set_index('Aspect')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.8),layout='constrained',
                          gridspec_kw={'width_ratios':[1.2,1]})
    ax=axes[0]; bars=ax.barh([ASPECT_CAPTIONS[a] for a in data.Aspect],data.Importance_Pct,color='#4477aa')
    ax.bar_label(bars,labels=[number(v)+'%' for v in data.Importance_Pct],padding=4,fontsize=9)
    ax.set(xlabel='Importance (%)',xlim=(0,29)); decimal_axes(ax,x=True,y=False)
    points=data[['Aspect','Importance_Pct']].copy()
    points['Performance_0_100']=[100*mentions.loc[a,'Mentioned_Reviews']/16292 for a in data.Aspect]
    ax=axes[1]; ax.scatter(points.Performance_0_100,points.Importance_Pct,s=65,color='#4477aa')
    ax.set(xlabel='Review đề cập khía cạnh / tổng review (%)',ylabel='Importance (%)',xlim=(0,35),ylim=(0,29))
    label_points(ax,points); decimal_axes(ax,x=True,y=True)
    save(fig,Path(__file__).stem)

if __name__=='__main__': main()
