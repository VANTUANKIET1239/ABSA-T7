"""Selected figure 4: saved BiRNN/Focal history, without retraining."""
import _bootstrap
from common.config import Path
from common.data_utils import load_training_history
from common.plot_utils import plt, save


def main():
    history = load_training_history()
    with plt.rc_context({'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        ax.plot(history.epoch, history.train_loss, label='Train Loss', marker='o')
        ax.plot(history.epoch, history.val_loss, label='Val Loss', marker='o')
        ax.set(title='Loss Curve - Cleanser ABSA - Scenario: FOCAL_LOSS', xlabel='Epoch', ylabel='Loss')
        ax.set_xticks(history.epoch.astype(int))
        ax.legend()
        ax.grid(True)
        save(fig, Path(__file__).stem)


if __name__ == '__main__':
    main()
