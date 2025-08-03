from matplotlib import pyplot as plt
import pandas as pd


def plot_training_curve(df):
    """
    Plot the training curve from the training history.

    Parameters:
        training_curve: list of dicts with keys 'epoch', 'loss', 'recon_loss_good', 'recon_loss_bad', 'kl_loss'
    """
    # df = df.iloc[2:, :]
    plt.figure(figsize=(12, 6))
    # plt.plot(df['epoch'], df['beta'], label='Beta', color='orange')
    plt.plot(df['epoch'], df['loss'], label='Total Loss')
    plt.plot(df['epoch'], df['kl_loss'], label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Curve')
    plt.legend()
    plt.grid()
    plt.yscale('log')  # Use logarithmic scale for better visibility
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    training_curve = pd.read_csv('vae_training_curve.csv')
    plot_training_curve(training_curve)