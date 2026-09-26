import argparse

import numpy as np
import pandas as pd

import config as cfg
from generator import generate_meteo, daytime_subset
from filters import build_filters
from parameters import FILTER_PARAMS
from plotting import plot_normalized, COLOR_KEEP

# Filters reported in detail on the real dataset.
TOP_FILTERS = ["DBSCAN", "VarTrimmed_FICP", "GemanMcClure"]
REAL_LEVEL = "medium"


def load_real(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return daytime_subset(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to the operational dataset (parquet).")
    args = parser.parse_args()

    _, location = generate_meteo()
    filters = build_filters(location)
    day = load_real(args.data)
    full = day

    p_ref = day["Potencia"].quantile(cfg.P_REF_QUANTILE)
    gn = day["g"].to_numpy() / cfg.G_REF
    pn = day["Potencia"].to_numpy() / p_ref
    out_dir = cfg.PLOTS_DIR / "real"
    plot_normalized(gn, pn, out_dir / "unfiltered.png")

    rows = []
    for name in filters:
        params = FILTER_PARAMS[REAL_LEVEL][name]
        outliers = filters[name](day, full, **params)
        rows.append({"filter": name, "removed_pct": round(100 * outliers.mean(), 2)})
        if name in TOP_FILTERS:
            keep = ~outliers
            plot_normalized(gn[keep], pn[keep], out_dir / f"{name}.png", color=COLOR_KEEP)

    pd.DataFrame(rows).to_csv(cfg.RESULTS_DIR / "real_removal_rates.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
