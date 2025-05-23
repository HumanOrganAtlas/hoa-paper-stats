from hoa_tools.inventory import load_inventory
from hoa_tools.dataset import get_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker


inventory = load_inventory()
datasets = {get_dataset(name) for name in inventory.index}


def plot_voxel_dataset_size():
    df = pd.DataFrame(
        [
            {
                "name": d.name,
                "voxel_size": d.data.voxel_size_um,
                "data_size_GB": np.prod(d.data.shape) * 2 / (1024 * 1024 * 1024),
                "Organ": d.sample.organ,
            }
            for d in datasets
        ]
    )

    fig = sns.relplot(
        data=df,
        x="voxel_size",
        y="data_size_GB",
        hue="Organ",
        aspect=1,
        height=4,
        style="Organ",
    )
    ax = fig.ax
    ax.set_xlabel("Voxel size / μm")
    ax.set_ylabel("Dataset size / GB")
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.grid(alpha=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.savefig(
        "figures/voxel_dataset_size.png",
    )


if __name__ == "__main__":
    plot_voxel_dataset_size()
