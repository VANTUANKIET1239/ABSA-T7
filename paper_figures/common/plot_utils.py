"""Common typography, exports, and reusable IPA axes; figure logic stays in its own script."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from matplotlib.font_manager import FontProperties
from .config import FIGURES, CODES, Q_NAMES, Q_COLORS, QUADRANTS
from .data_utils import number

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def decimal_axes(ax,x=False,y=True,digits=0):
    formatter=FuncFormatter(lambda v,pos:number(v,digits))
    if x: ax.xaxis.set_major_formatter(formatter)
    if y: ax.yaxis.set_major_formatter(formatter)

def save(fig,name):
    for ax in fig.axes:
        for collection in ax.collections: collection.set_rasterized(False)
    for kind in ['png','pdf']:
        (FIGURES/kind).mkdir(parents=True,exist_ok=True)
        fig.savefig(FIGURES/kind/f'{name}.{kind}',dpi=300,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    print(f'[+] {name}: PNG 300 dpi + vector PDF',flush=True)

def heatmap(ax,values,**kwargs):
    """Use vector cells rather than an embedded raster image."""
    matrix=np.asarray(values,dtype=float)
    rows,cols=matrix.shape
    result=ax.pcolormesh(np.arange(cols+1)-.5,np.arange(rows+1)-.5,matrix,
                         shading='flat',edgecolors='white',linewidth=.4,rasterized=False,**kwargs)
    ax.set_xlim(-.5,cols-.5); ax.set_ylim(rows-.5,-.5)
    return result

def label_points(ax,frame):
    """Place code labels in display coordinates, without altering data coordinates."""
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer()
    bounds=ax.get_window_extent(renderer).padded(-4)
    reserved=[artist for artist in ax.texts if artist.get_gid()=='ipa_quadrant_label']
    reserved_boxes=[artist.get_window_extent(renderer).padded(3) for artist in reserved]
    used=reserved_boxes.copy(); annotations=[]
    points=ax.transData.transform(frame[['Performance_0_100','Importance_Pct']].to_numpy(float))
    for index,row in enumerate(frame.itertuples()):
        text=CODES[row.Aspect] + ('*' if getattr(row,'Low_Mention_Support',False) else '')
        xy=(float(row.Performance_0_100),float(row.Importance_Pct))
        best=None
        for radius in [9,16,24,33,43,55,68,85,105]:
            for angle in [45,135,-45,-135,0,90,180,-90,22.5,67.5,112.5,157.5,-22.5,-67.5,-112.5,-157.5]:
                rad=np.deg2rad(angle); dx,dy=radius*np.cos(rad),radius*np.sin(rad)
                center=points[index]+[dx,dy]
                width,height,_=renderer.get_text_width_height_descent(text,FontProperties(size=9.2),ismath=False)
                width+=4; height+=5
                box=Bbox.from_bounds(center[0]-width/2,center[1]-height/2,width,height)
                if not bounds.contains(box.x0,box.y0) or not bounds.contains(box.x1,box.y1): continue
                overlap=sum(box.padded(2).overlaps(b) for b in used)
                cover=sum(box.padded(4).contains(*p) for p in points)
                score=1000*overlap+100*cover+radius
                if best is None or score<best[0]: best=(score,dx,dy,box)
                if score<100: break
            if best is not None and best[0]<100: break
        if best is None: best=(0,8,8,Bbox.from_bounds(*points[index],20,12))
        _,dx,dy,box=best; used.append(box.padded(2))
        annotation=ax.annotate(text,xy,xytext=(dx*72/ax.figure.dpi,dy*72/ax.figure.dpi),textcoords='offset points',ha='center',va='center',
                    fontsize=9.2,color='black',zorder=5,
                    bbox={'boxstyle':'round,pad=.08','fc':'white','ec':'none','alpha':.88},
                    arrowprops={'arrowstyle':'-','lw':.45,'color':'#777','shrinkA':2,'shrinkB':4})
        annotations.append(annotation)
    ax.figure.canvas.draw()
    boxes=[a.get_bbox_patch().get_window_extent(ax.figure.canvas.get_renderer()) for a in annotations]
    collisions=[(annotations[i].get_text(),annotations[j].get_text())
                for i in range(len(boxes)) for j in range(i) if boxes[i].overlaps(boxes[j])]
    if collisions: raise RuntimeError(f'Overlapping IPA/scatter labels: {collisions}')
    quadrant_collisions=[(annotation.get_text(),artist.get_text())
                         for annotation,box in zip(annotations,boxes)
                         for artist,reserved_box in zip(reserved,reserved_boxes)
                         if box.overlaps(reserved_box)]
    if quadrant_collisions:
        raise RuntimeError(f'IPA label/quadrant overlap in {ax.get_title()}: {quadrant_collisions}')

def ipa_quadrant_labels(ax,frame,importance_cut,performance_cut):
    """Place short quadrant names in free space without covering data markers."""
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer()
    xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
    ic,pc=importance_cut,performance_cut
    points=ax.transData.transform(frame[['Performance_0_100','Importance_Pct']].to_numpy(float))
    radius=np.sqrt(ax.collections[-1].get_sizes()[0])*ax.figure.dpi/144+1
    point_boxes=[Bbox.from_bounds(x-radius,y-radius,2*radius,2*radius) for x,y in points]
    zones=[(xmin,pc,ic,ymax),(pc,xmax,ic,ymax),
           (xmin,pc,ymin,ic),(pc,xmax,ymin,ic)]
    for index,(quadrant,zone) in enumerate(zip(QUADRANTS,zones)):
        x0,x1,y0,y1=zone
        text=Q_NAMES[quadrant]
        zone_box=Bbox(ax.transData.transform([[x0,y0],[x1,y1]])).padded(-3)
        artist=ax.text(0,0,text,ha='center',va='center',fontsize=7.6,fontweight='bold',
                       color='#2563eb',zorder=6,gid='ipa_quadrant_label',alpha=0,
                       bbox={'boxstyle':'round,pad=.1','fc':'white','ec':'none','alpha':.85})
        for font_size,displayed in [(7.6,text),(7.6,text.replace(' ','\n')),
                                    (7.0,text),(7.0,text.replace(' ','\n'))]:
            artist.set_fontsize(font_size)
            artist.set_text(displayed)
            extent=artist.get_window_extent(renderer)
            width,height=extent.width+2,extent.height+2
            best=None
            for xf in np.linspace(.05,.95,19):
                fractions=np.linspace(.05,.95,19)
                for yf in fractions:
                    xy=(x0+(x1-x0)*xf,y0+(y1-y0)*yf)
                    center=ax.transData.transform(xy)
                    box=Bbox.from_bounds(center[0]-width/2,center[1]-height/2,width,height)
                    if not zone_box.contains(box.x0,box.y0) or not zone_box.contains(box.x1,box.y1):
                        continue
                    overlap=sum(box.overlaps(point_box) for point_box in point_boxes)
                    score=1000*overlap+abs(xf-.5)+abs(yf-(.9 if index<2 else .45))
                    if best is None or score<best[0]: best=(score,xy)
            if best is not None and best[0]<1000: break
        if best is None or best[0]>=1000:
            raise RuntimeError(f'No clear quadrant-label position in {ax.get_title()}: {text}')
        artist.set_position(best[1]); artist.set_alpha(1)


def ipa_axes(ax,frame,title,importance_cut=None,performance_cut=None,xlim=(40,100),ylim=None,labels=True,quadrant_labels=False,marker_size=61):
    ic=float(frame.Importance_Cut.iloc[0]) if importance_cut is None else float(importance_cut)
    pc=float(frame.Performance_Cut.iloc[0]) if performance_cut is None else float(performance_cut)
    ymax=ylim[1] if ylim else max(28,float(frame.Importance_Pct.max())*1.16)
    ax.set(xlim=xlim,ylim=(0,ymax),xlabel='Performance (thang 0–100)',ylabel='Importance (%)',title=title)
    ax.axvspan(xlim[0],pc,ymin=ic/ymax,alpha=.065,color=Q_COLORS[0])
    ax.axvspan(pc,xlim[1],ymin=ic/ymax,alpha=.065,color=Q_COLORS[1])
    ax.axvspan(xlim[0],pc,ymax=ic/ymax,alpha=.045,color=Q_COLORS[2])
    ax.axvspan(pc,xlim[1],ymax=ic/ymax,alpha=.055,color=Q_COLORS[3])
    ax.axvline(pc,color='#555',ls='--',lw=.9)
    ax.axhline(ic,color='#555',ls='--',lw=.9)
    groups=np.where(frame.Importance_Pct.ge(ic),np.where(frame.Performance_0_100.ge(pc),1,0),
                    np.where(frame.Performance_0_100.ge(pc),3,2))
    ax.scatter(frame.Performance_0_100,frame.Importance_Pct,c=[Q_COLORS[i] for i in groups],
               s=marker_size,edgecolors='white',linewidths=.7,zorder=4)
    decimal_axes(ax,x=True,y=True)
    if quadrant_labels: ipa_quadrant_labels(ax,frame,ic,pc)
    if labels: label_points(ax,frame)
    return ic,pc

def ipa_legend(fig,location='lower center',anchor=(.5,0),columns=2):
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markersize=7,label=Q_NAMES[q])
             for q,c in zip(QUADRANTS,Q_COLORS)]
    fig.legend(handles=handles,loc=location,bbox_to_anchor=anchor,ncol=columns,frameon=False,fontsize=9)

