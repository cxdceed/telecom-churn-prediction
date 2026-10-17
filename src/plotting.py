"""Shared plot style. Every figure carries a 'SYNTHETIC DATA' watermark caption."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config as C

PALETTE = {"primary": "#1f4e79", "accent": "#d95f02", "grey": "#8c8c8c", "good": "#2a9d8f"}


def style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.text(0.99, -0.03, "Synthetic data - methodology demo, not real-world evidence",
             ha="right", va="bottom", fontsize=7, color="#777777", style="italic")
    fig.savefig(C.FIGURES / name, bbox_inches="tight")
    plt.close(fig)
