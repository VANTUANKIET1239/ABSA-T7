"""Selected figure 2: within-platform stacked rating shares."""
import _bootstrap
import numpy as np
from common.config import Path
from common.data_utils import rating_counts
from common.plot_utils import plt, save


def main():
    ratings = rating_counts().set_index('Platform')
    fig, ax = plt.subplots(figsize=(10.8, 3.7), layout='constrained')
    left = np.zeros(len(ratings))
    for star, color in zip(range(1, 6), ['#b64e58', '#d58e76', '#d6c38c', '#82afa3', '#247f72']):
        values = 100 * ratings[str(star)].to_numpy() / ratings.sum(axis=1).to_numpy()
        ax.barh(ratings.index, values, left=left, label=f'{star} sao', color=color)
        for i, value in enumerate(values):
            if value >= 5:
                ax.text(left[i] + value / 2, i, f'{value:.1f}%', ha='center', va='center',
                        color='white' if star in [1, 5] else '#17314b', fontsize=10)
        left += values
    ax.set(xlim=(0, 100), xlabel='Tỷ lệ review trong từng nền tảng (%)')
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.22), frameon=False)
    save(fig, Path(__file__).stem)


if __name__ == '__main__':
    main()
