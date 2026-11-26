"""Categorical IPA overall, by platform, and by brand."""
from matplotlib.colors import ListedColormap,BoundaryNorm
import _bootstrap
from common.config import *
from common.data_utils import (
    load_brand_run_summary, load_ipa_by_brand, load_ipa_by_platform,
    load_ipa_overall, load_platform_run_summary,
)
from common.plot_utils import plt,save,heatmap

def main():
    platforms=load_platform_run_summary()['Platform'].tolist()
    brands=load_brand_run_summary()['Brand'].tolist()
    overall=load_ipa_overall().set_index('Aspect')[['IPA_Quadrant']]
    overall=overall.rename(columns={'IPA_Quadrant':'Tổng thể'}).reindex(ASPECTS[:10])
    panels=[overall]
    for data,col,groups in [(load_ipa_by_platform(),'Platform',platforms),
                            (load_ipa_by_brand(),'Brand',brands)]:
        panels.append(data.pivot(index='Aspect',columns=col,values='IPA_Quadrant')
                      .reindex(index=ASPECTS[:10],columns=groups))
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.8),sharey=True,layout='constrained',
                          gridspec_kw={'width_ratios':[1,3,5]})
    for ax,values in zip(axes,panels):
        if values.isna().any().any() or not values.isin(QUADRANTS).all().all():
            raise ValueError('Every IPA cell must contain a recorded quadrant for one of the 10 aspects.')
        groups=values.columns.tolist()
        numbers=values.map(lambda q:QUADRANTS.index(q))
        heatmap(ax,numbers,cmap=ListedColormap(Q_COLORS),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4))
        ax.set_xticks(range(len(groups)),groups,rotation=25,ha='right')
        ax.set_yticks(range(10),[ASPECT_CAPTIONS[a] for a in ASPECTS[:10]])
        for y in range(10):
            for x in range(len(groups)):
                quadrant=values.iloc[y,x]
                ax.text(x,y,Q_NAMES[quadrant],ha='center',va='center',
                        color='#17314b' if quadrant=='Low Priority' else 'white',fontsize=8.7)
    save(fig,Path(__file__).stem)

if __name__=='__main__': main()
