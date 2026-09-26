import numpy as np
import pandas as pd

import config as cfg
from generator import (generate_meteo, generate_inverter_dataset, daytime_subset,
                       snr_db, find_scale_for_snr)
from filters import build_filters
from evaluation import evaluate
from plotting import plot_ground_truth, plot_filter_result
from parameters import FILTER_PARAMS


def resolve_noise_scales(meteo):
    scales = {}
    for i, inv in enumerate(cfg.INVERTERS):
        scales[inv["name"]] = {
            level: find_scale_for_snr(meteo, inv, target, cfg.RANDOM_SEED + i)
            for level, target in cfg.TARGET_SNR.items()
        }
    return scales


def main():
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    meteo, location = generate_meteo()
    filters = build_filters(location)
    scales = resolve_noise_scales(meteo)
    pd.DataFrame(scales).T.to_csv(cfg.RESULTS_DIR / "noise_scales.csv")

    rows = []
    for i, inv in enumerate(cfg.INVERTERS):
        seed = cfg.RANDOM_SEED + i
        clean = generate_inverter_dataset(meteo, inv["dc_nominal"], inv["inverter_ac_power"],
                                          np.random.default_rng(seed), noise_scale=0.0)
        for level, scale in scales[inv["name"]].items():
            full = generate_inverter_dataset(meteo, inv["dc_nominal"], inv["inverter_ac_power"],
                                             np.random.default_rng(seed), noise_scale=scale)
            day = daytime_subset(full)
            snr = snr_db(clean, full)
            plot_ground_truth(day, cfg.PLOTS_DIR / f"{inv['name']}_{level}" / "ground_truth.png")
            for name in filters:
                params = FILTER_PARAMS[level][name]
                outliers = filters[name](day, full, **params)
                record = evaluate(day, outliers, inv["name"], name)
                record.update({"level": level, "noise_scale": scale, "snr_db": round(snr, 1)})
                rows.append(record)
                plot_filter_result(day, outliers, cfg.PLOTS_DIR / f"{inv['name']}_{level}" / name)

    results = pd.DataFrame(rows)
    results.to_csv(cfg.RESULTS_DIR / "synthetic_results.csv", index=False)

    for inv in cfg.INVERTERS:
        table = (results[results.inverter == inv["name"]]
                 .pivot(index="filter", columns="level", values="f1")[["low", "medium", "high"]]
                 .sort_values("medium", ascending=False))
        print(inv["name"])
        print(table.round(3).to_string())
        print()


if __name__ == "__main__":
    main()
