import collections
from hoa_tools.inventory import load_inventory
from hoa_tools.dataset import get_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker
import matplotlib.pyplot as plt


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
        "figures/voxel_dataset_size.svg",
    )


def plot_disease_types():
    histories = [d.donor.medical_history for d in datasets]
    histories = [h if h is not None else "" for h in histories]

    causes = [d.donor.cause_of_death for d in datasets]
    causes = [h if h is not None else "" for h in causes]

    histories = {h + " " + c for h, c in zip(histories, causes)}

    keywords = [
        "Gout",
        "Stroke",
        "Cataract",
        "Stent",
        "Diabetes",
        "Renal failure",
        "Nephrectomy",
        "Bacterial superinfection",
        "Predementia",
        "Myocardial infarction",
        "Heart failure",
        "Cancer",
        "Coronary heart disease",
        "COVID-19",
        "Chronic obstructive pulmonary disease",
        "Cognitive disorders of vascular origin",
        "Lung pneumopathy",
        "Liver Failure",
    ]
    counts = {
        k: len([h for h in histories if k.lower() in h.lower()]) for k in keywords
    }
    counts["Diabetes"] = len({d.donor.id for d in datasets if d.donor.diabetes})
    counts["Hypertension"] = len({d.donor.id for d in datasets if d.donor.hypertension})
    counts = collections.OrderedDict(sorted(counts.items()))
    print(counts)

    fig, ax = plt.subplots(tight_layout=True)
    ax.xaxis.grid(linewidth=1, zorder=-10)
    ax.barh(list(counts.keys()), list(counts.values()), zorder=10)
    ax.set_xlabel("Number of donors")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.invert_yaxis()

    fig.savefig(
        "figures/disease_bar.svg",
    )


if __name__ == "__main__":
    plot_disease_types()
    # plot_voxel_dataset_size()
