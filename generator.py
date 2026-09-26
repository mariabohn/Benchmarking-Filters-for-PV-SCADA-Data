import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from scipy.signal import lfilter

import config as cfg


def ar1(n, phi, rng):
    return lfilter([np.sqrt(1 - phi ** 2)], [1.0, -phi], rng.normal(0, 1, n))


def generate_meteo(year=cfg.YEAR):
    loc = Location(cfg.LATITUDE, cfg.LONGITUDE, tz=cfg.TIMEZONE, altitude=cfg.ALTITUDE)
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45:00", freq="5min", tz=cfg.TIMEZONE)

    cs = loc.get_clearsky(times)
    sun = loc.get_solarposition(times)
    az = sun["azimuth"].apply(lambda a: a if a < 180 else a - 360)

    tracker = pvlib.tracking.singleaxis(
        apparent_zenith=sun["apparent_zenith"], solar_azimuth=az,
        axis_tilt=cfg.TRACKER_AXIS_TILT, axis_azimuth=cfg.TRACKER_AXIS_AZIMUTH,
        max_angle=cfg.TRACKER_MAX_ANGLE, backtrack=True, gcr=cfg.GCR)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracker["surface_tilt"], surface_azimuth=tracker["surface_azimuth"],
        dni=cs["dni"], ghi=cs["ghi"], dhi=cs["dhi"],
        solar_zenith=sun["apparent_zenith"], solar_azimuth=az, albedo=cfg.ALBEDO)

    m = pd.DataFrame(index=times)
    m["g_clear"] = poa["poa_global"].fillna(0).clip(lower=0)
    m["ghi_clear"] = cs["ghi"].clip(lower=0)
    doy = times.dayofyear
    hour = times.hour + times.minute / 60.0
    m["ambT"] = (28 + 5 * np.cos(2 * np.pi * (doy - 15) / 365)
                 + 10 * np.sin(np.pi * (hour - 6) / 12) * (hour > 6) * (hour < 18))
    m["panT"] = pvlib.temperature.pvsyst_cell(m["g_clear"], m["ambT"])
    m["temp_factor"] = 1 + cfg.PANEL_TEMP_DECAY * (25 - m["panT"].fillna(50))
    return m, loc


def generate_inverter_dataset(meteo, dc_nominal, inv_ac, rng, noise_scale=1.0):
    a_scale = noise_scale ** 0.7

    n = len(meteo)
    g_clear = meteo["g_clear"].to_numpy()
    ghi_clear = meteo["ghi_clear"].to_numpy()
    temp_factor = meteo["temp_factor"].to_numpy()
    panT = meteo["panT"].to_numpy()
    ambT = meteo["ambT"].to_numpy()
    daylight = g_clear > 0
    pdc0 = inv_ac / cfg.INVERTER_NOM_EFF

    z = ar1(n, 0.95, rng)
    cloud_now = z > 0.6
    depth = np.clip((z - 0.6) / 1.4, 0, 0.9)
    base = np.where(cloud_now, 1.0 - depth * rng.uniform(0.3, 1.0, n), 1.0)
    ev = (rng.random(n) < 0.05) & cloud_now
    base = np.where(ev, base * rng.uniform(0.15, 0.6, n), base)
    enh = rng.random(n) < 0.012
    base = base * np.where(enh, rng.uniform(1.05, 1.25, n), 1.0)
    ktp_point = np.where(daylight, np.clip(base, 0.05, 1.4), 1.0)

    smoothed = pd.Series(ktp_point).ewm(span=8, adjust=False).mean().to_numpy()
    ktp_plant = np.empty(n)
    ktp_plant[0] = smoothed[0]
    for i in range(1, n):
        tgt = smoothed[i]
        ktp_plant[i] = min(tgt, ktp_plant[i - 1] + 0.025) if tgt > ktp_plant[i - 1] else tgt
    ktp_plant = np.clip(ktp_plant, 0.03, 1.10)

    g_meas = g_clear * ktp_point
    ghi_meas = ghi_clear * ktp_point
    dc = dc_nominal * (g_clear * ktp_plant / 1000.0) * temp_factor * (1 - cfg.DC_STD_LOSSES)
    p = np.clip(pvlib.inverter.pvwatts(pd.Series(dc), pdc0).to_numpy(), 0, inv_ac)

    underread = (rng.random(n) < 0.025 * a_scale) & daylight
    g_meas = np.where(underread, g_meas * rng.uniform(0.05, 0.35, n), g_meas)
    overread = (rng.random(n) < 0.015 * a_scale) & daylight
    g_meas = np.where(overread, g_meas * rng.uniform(1.3, 2.2, n), g_meas)

    partial = (rng.random(n) < 0.05 * a_scale) & daylight & (~cloud_now)
    p = np.where(partial, p * rng.uniform(0.35, 0.85, n), p)

    g_meas = g_meas * (1 + 0.02 * noise_scale * ar1(n, 0.85, rng)) + rng.normal(0, 4.0 * noise_scale, n)
    ghi_meas = ghi_meas * (1 + 0.02 * noise_scale * ar1(n, 0.85, rng)) + rng.normal(0, 4.0 * noise_scale, n)
    p = p * (1 + 0.012 * noise_scale * ar1(n, 0.85, rng)) + rng.normal(0, 0.004 * inv_ac * noise_scale, n)

    spike = (rng.random(n) < 0.006 * a_scale) & daylight
    p = np.where(spike, p * rng.uniform(0.05, 0.9, n), p)
    gspike = (rng.random(n) < 0.002 * a_scale) & daylight
    g_meas = np.where(gspike, g_meas + rng.uniform(50, 250, n), g_meas)

    g_meas = np.clip(g_meas, 0, 1450)
    ghi_meas = np.clip(ghi_meas, 0, 1450)
    p = np.where(daylight, np.clip(p, 0, inv_ac), 0.0)

    dc_ideal = dc_nominal * (g_meas / 1000.0) * temp_factor * (1 - cfg.DC_STD_LOSSES)
    p_ideal = np.clip(pvlib.inverter.pvwatts(pd.Series(dc_ideal), pdc0).to_numpy(), 0, inv_ac)
    resid = np.abs(p - p_ideal)
    tol = cfg.FAULT_TOL_ABS_FRAC * inv_ac + cfg.FAULT_TOL_REL * p_ideal
    is_loss = daylight & ((resid > tol) | (g_meas > cfg.G_PHYS_MAX))

    df = pd.DataFrame({"g": g_meas, "ghi": ghi_meas, "Potencia": p,
                       "panT": panT, "ambT": ambT}, index=meteo.index)
    df["is_loss"] = is_loss
    return df


def daytime_subset(df, min_irradiance=10.0):
    return df[(df["g"] > min_irradiance) & (df["Potencia"] > 0)].copy()


def snr_db(clean, noisy):
    m = clean["g"].to_numpy() > 10
    sig = clean["Potencia"].to_numpy()[m]
    noise = (noisy["Potencia"].to_numpy() - clean["Potencia"].to_numpy())[m]
    return 10 * np.log10(np.var(sig) / (np.var(noise) + 1e-12))


def find_scale_for_snr(meteo, inv, target_db, seed, lo=0.05, hi=8.0, iters=12):
    def gen(scale):
        return generate_inverter_dataset(meteo, inv["dc_nominal"], inv["inverter_ac_power"],
                                          np.random.default_rng(seed), noise_scale=scale)
    clean = gen(0.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if snr_db(clean, gen(mid)) > target_db:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 3)
