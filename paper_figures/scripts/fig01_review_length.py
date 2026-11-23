"""Selected figure 1: review-length histogram and platform boxplots."""
import _bootstrap
import numpy as np
from common.config import Path
from common.data_utils import load
from common.plot_utils import plt, save


def main():
    data = load('review_observations')
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), layout='constrained')
    axes[0].hist(data.Whitespace_Tokens, bins=np.arange(0, min(500, data.Whitespace_Tokens.max()) + 5, 5),
                 color='#397ea1', edgecolor='white')
    axes[0].set(xlim=(0, min(250, data.Whitespace_Tokens.quantile(.995))),
                title='Phân phối độ dài cleaned_comment', xlabel='Đơn vị cách nhau bằng khoảng trắng', ylabel='Số review')
    groups = ['Lazada', 'Shopee', 'Tiki']
    axes[1].boxplot([data.loc[data.Platform.eq(group), 'Whitespace_Tokens'] for group in groups],
                    tick_labels=groups, showfliers=False, patch_artist=True,
                    boxprops={'facecolor': '#b3cbd7'})
    axes[1].set(title='So sánh độ dài giữa các nền tảng', ylabel='Đơn vị cách nhau bằng khoảng trắng')
    save(fig, Path(__file__).stem)


if __name__ == '__main__':
    main()
