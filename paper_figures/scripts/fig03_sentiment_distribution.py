"""Selected figure 3: eleven-aspect sentiment count heatmap and shares."""
import _bootstrap
import numpy as np
from common.config import Path, ASPECTS, ASPECT_CAPTIONS
from common.data_utils import load
from common.plot_utils import plt, save, heatmap


def main():
    sentiments = ['positive', 'negative', 'neutral']
    counts = load('06_aspect_sentiment_counts').set_index('Aspect').loc[ASPECTS, sentiments].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), layout='constrained',
                            gridspec_kw={'width_ratios': [1, 1.15]})
    heatmap(axes[0], counts, cmap='Blues')
    captions = {**ASPECT_CAPTIONS, 'General': 'Đánh giá chung - A11'}
    axes[0].set_xticks(range(3), ['Tích cực', 'Tiêu cực', 'Trung tính'])
    axes[0].set_yticks(range(11), [captions[aspect] for aspect in ASPECTS])
    for i in range(11):
        for j in range(3):
            axes[0].text(j, i, f'{counts[i, j]:,}', ha='center', va='center', fontsize=9,
                         color='white' if counts[i, j] > counts.max() * .52 else '#17314b')
    left = np.zeros(11)
    for j, (color, label) in enumerate(zip(['#178477', '#c15c64', '#d8a23d'], ['Tích cực', 'Tiêu cực', 'Trung tính'])):
        shares = 100 * counts[:, j] / counts.sum(axis=1)
        axes[1].barh(range(11), shares, left=left, color=color, label=label)
        for i, value in enumerate(shares):
            if value >= 9:
                axes[1].text(left[i] + value / 2, i, f'{value:.0f}%', ha='center', va='center',
                             fontsize=9, color='white' if j != 2 else '#17314b')
        left += shares
    axes[1].set_yticks(range(11), [''] * 11)
    axes[1].set(xlim=(0, 100), ylim=axes[0].get_ylim(), xlabel='Tỷ lệ cặp nhãn (%)')
    axes[1].legend(ncol=3, loc='lower center', bbox_to_anchor=(.5, -.22), frameon=False, fontsize=9)
    save(fig, Path(__file__).stem)


if __name__ == '__main__':
    main()
