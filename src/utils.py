import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)