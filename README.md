# Benchmarking Anomaly Detection Filters for Photovoltaic SCADA Data

Code and configuration accompanying the paper *Benchmarking Anomaly Detection
Filters for Photovoltaic SCADA Data*. The benchmark compares sixteen
statistical, robust-regression, and machine-learning filtering methods on a
physics-informed synthetic dataset with per-sample ground-truth labels, and on
a real operational dataset.

## Overview

A synthetic PV plant is simulated with `pvlib` for three inverter
configurations and three disturbance levels defined by their power
signal-to-noise ratio (SNR). Each sample receives a ground-truth label derived
from the deviation between measured power and a physically modeled reference.
The sixteen filters are then applied under known ground truth (synthetic) and
under operational conditions without labels (real data).

## Repository structure

    config.py          Experiment constants (site, model, labelling, seeds).
    generator.py       Synthetic meteorology, plant model, noise, and labels.
    filters.py         The sixteen filtering methods.
    parameters.py      Selected filter parameters per disturbance level.
    evaluation.py      Precision, recall, F1, and related metrics.
    plotting.py        Scatter plots (synthetic and normalized real data).
    run_synthetic.py   Full synthetic benchmark (3 inverters x 3 levels x 16 filters).
    run_real.py        Applies the filters to an operational dataset.

## Installation

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

Synthetic benchmark. Writes results and figures to `results/`:

    python run_synthetic.py

The disturbance scale for each inverter and level is calibrated to the target
SNR values in `config.py`; the resulting scales are saved to
`results/noise_scales.csv`.

Real dataset. The operational data are subject to a non-disclosure agreement
and are not distributed. The script expects a parquet file with a datetime
index and columns `g` (POA irradiance, W/m^2), `Potencia` (AC power, kW), and
`panT` (module temperature, C):

    python run_real.py path/to/data.parquet

Values in the real-data figures are normalized (irradiance by 1000 W/m^2, power
by its 99th percentile); the filters are applied to the original measurements.

## Parameter selection

Filter parameters were selected on the intermediate-capacity inverter
(`INV_1700`) independently for each disturbance level and applied unchanged to
the other inverters. The selected values are listed in `parameters.py`.

## Reproducibility

All randomness is controlled by a fixed seed (`config.RANDOM_SEED`). The same
meteorological realization and seed are used across disturbance levels for each
inverter, so performance differences reflect disturbance intensity rather than
weather variability.
