import collections
from datetime import date
from hoa_tools.inventory import load_inventory
from hoa_tools.dataset import get_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.dates

plt.rcParams.update({"font.sans-serif": "Arial"})

inventory = load_inventory()
datasets = {get_dataset(name) for name in inventory.index}
datasets = {d for d in datasets if d.donor.id not in ["AUMC-005"]}


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
    fig.savefig("figures/voxel_dataset_size.svg")


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
        "Bacterial superinfection",
        "Predementia",
        "Myocardial infarction",
        "Heart failure",
        "Cancer",
        "Coronary heart disease",
        "COVID",
        "Chronic obstructive pulmonary disease",
        "Cognitive disorders of vascular origin",
        "Liver Failure",
        "Dandy-Walker syndrome variant",
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

    fig.savefig("figures/disease_bar.svg", dpi=150)


def plot_donor_table():
    donors = {d.donor.id: d.donor for d in datasets}
    organs = sorted({d.sample.organ for d in datasets})
    cells = []
    cells.append(["Donor ID", "Sex", "Age"] + organs)
    for donor in sorted(donors.values(), key=lambda d: d.id):
        cells.append(
            [
                donor.id,
                donor.sex,
                str(int(donor.age.root)) if donor.age is not None else "",
            ]
        )
        for organ in organs:
            n_zoom = len(
                [
                    d
                    for d in datasets
                    if d.donor == donor
                    and d.sample.organ == organ
                    and d.dataset_type == "zoom"
                ]
            )
            n_overview = len(
                [
                    d
                    for d in datasets
                    if d.donor == donor
                    and d.sample.organ == organ
                    and d.dataset_type == "overview"
                ]
            )
            cells[-1].append(
                (
                    str(n_overview) + " + " + str(n_zoom)
                    if n_overview + n_zoom > 0
                    else ""
                ),
            )

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")
    ax.table(cells, loc="center", cellLoc="left")
    fig.savefig("figures/donor_table.svg")

    df = pd.DataFrame(cells)
    df.to_csv("donor_data.csv", index=False)


def get_scan_data():
    data = collections.defaultdict(list)

    for dataset in datasets:
        scan = dataset.scan
        if (
            scan.scan_type == "zseries"
            and scan.n_scans is not None
            and scan.scan_time is not None
            # and scan.date <= date(2023, 1, 1)
        ):
            total_scan_time = scan.n_scans.root * scan.scan_time.root / 60 / 60
            n_vox = (
                dataset.data.shape[0] * dataset.data.shape[1] * dataset.data.shape[2]
            )
            data_size = (
                dataset.data.shape[0]
                * dataset.data.shape[1]
                * dataset.data.shape[2]
                * 2
                / 1e9
            )
            physical_size = (
                dataset.data.shape[0]
                * dataset.data.shape[1]
                * dataset.data.shape[2]
                * (scan.pixel_size * 1e-3) ** 3
                * np.pi
                / 4
            )

            data["single_scan_time"].append(scan.scan_time.root / 60)
            data["physical_size"].append(physical_size)
            data["data_size"].append(data_size)
            data["scan_time"].append(total_scan_time)
            data["date"].append(scan.date)
            data["Dataset type"].append(dataset.dataset_type)
            data["scan_speed"].append(n_vox / (total_scan_time * 60 * 60))
            data["det_y"].append(
                scan.sensor_roi_y_size.root * scan.n_scans.root
                if scan.sensor_roi_y_size is not None
                else 0
            )
            data["nz"].append(dataset.data.shape[2])
            data["voxel_size"].append(scan.pixel_size)
            data["organ"].append(dataset.sample.organ)

    return pd.DataFrame(data)


def plot_scan_speeds():
    fig, ax = plt.subplots(figsize=(5, 5))

    df = get_scan_data()
    ax = sns.scatterplot(x="date", y="scan_speed", hue="Dataset type", data=df, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Scan date")
    ax.set_ylabel("Voxels per second")
    ax.set_ylim(1e6, 1e8)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    fig.savefig("figures/voxel_speed.svg")


if __name__ == "__main__":
    # plot_disease_types()
    # plot_voxel_dataset_size()
    # plot_donor_table()
    plot_scan_speeds()
