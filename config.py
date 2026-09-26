from pathlib import Path

# Site (single-axis tracker utility-scale plant)
LATITUDE, LONGITUDE, ALTITUDE = -15.825486, -43.444673, 530
TIMEZONE = "America/Sao_Paulo"
YEAR = 2025

# Tracker geometry
TRACKER_AXIS_TILT, TRACKER_AXIS_AZIMUTH, TRACKER_MAX_ANGLE = 0, 180, 55
GCR, ALBEDO = 0.327, 0.2

# Electrical model
PANEL_TEMP_DECAY = 0.0036
DC_STD_LOSSES = 0.02
INVERTER_NOM_EFF = 0.96

# Ground-truth labelling
FAULT_TOL_ABS_FRAC = 0.045
FAULT_TOL_REL = 0.05
G_PHYS_MAX = 1150

RANDOM_SEED = 42

INVERTERS = [
    {"name": "INV_0200", "dc_nominal": 268, "inverter_ac_power": 200},
    {"name": "INV_1700", "dc_nominal": 2280, "inverter_ac_power": 1700},
    {"name": "INV_5100", "dc_nominal": 6835, "inverter_ac_power": 5100},
]

# Target power SNR (dB) defining each disturbance level
TARGET_SNR = {"low": 25.8, "medium": 11.4, "high": 7.7}

# Inverter used to tune the filter parameters
TUNING_INVERTER = "INV_1700"

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

# Normalization references for anonymized plotting of the real dataset
G_REF = 1000.0
P_REF_QUANTILE = 0.99
