"""Generate daily deterministic WSA + simpleBRaVDA + SURF forecasts.

This is the WSA/BRaVDA analogue of ``generate_insitusurf_forecasts.py``.
Each run assimilates the preceding Carrington rotation of ICME-cleaned OMNI
speed into the latest WSA map available at the issue time, propagates the
posterior boundary with 1-D SURF/hydro, and stores only the forecast interval.
"""

import datetime as dt
from pathlib import Path
import re
import sys

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas as pd
from sunpy.coordinates import sun

import surf.surf as surf_model
import surf.surf_analysis as sa
import surf.surf_inputs as sin

from generate_insitusurf_forecasts import prepare_omni

SIMPLE_BRAVDA = Path(__file__).resolve().parent / "simpleBRaVDA"
sys.path.insert(0, str(SIMPLE_BRAVDA))
from simpleBRaVDA import (  PERIOD_DAYS,  Settings, deaccelerate_wsa_speeds, earth_latitudes, omni_observations, simpleBRaVDA,)
from wsa import get_wsa_from_iswa, read_wsa  


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
WINDOW_START = dt.datetime(2022, 10, 2)
WINDOW_END = dt.datetime(2025, 12, 31)
FORECAST_CADENCE = dt.timedelta(days=1)
FORECAST_DAYS = 10

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "wsa_bravda_surf_forecasts"
WSA_DIRECTORY = Path(__file__).resolve().parent / "wsa_data"
# "local" recursively searches LOCAL_WSA_DIRECTORY for the nearest map;
# "download" retains the existing CCMC/ISWA download behaviour.
WSA_SOURCE = "local"
LOCAL_WSA_DIRECTORY = Path(r"C:\Users\mathe\Dropbox\Data\WSA\WSA_CCMC")
OVERWRITE_EXISTING = False

RMIN = 0.1 * u.AU
RMAX = 240 * u.solRad
DR = 1.5 * u.solRad
DT_SCALE = 4
V_MAX = 3000 * u.km / u.s
GAMMA = 1.5
SOLVER = "hydro"
BRAVDA_SETTINGS = Settings(n_longitude=128, n_ensemble=500, seed=20000)
APPLY_HUXT_TO_HYDRO_CONVERSION = True
MIN_BRAVDA_SPEED = 200 * u.km / u.s

# The BRaVDA posterior uses the HUXt 0.1-AU speed convention. Convert it to
# the hydro/Parker 0.1-AU convention through equal-speed points at 1 AU.
SPEED_LUT_1AU_MIN = 150 * u.km / u.s
SPEED_LUT_1AU_MAX = 1500 * u.km / u.s
SPEED_LUT_STEP = 1 * u.km / u.s
SPEED_LUT_OUTER_RADIUS = 215 * u.solRad

_LOCAL_WSA_INDEX = None
_SPEED_CONVERSION_LUT = None


def speed_conversion_lut():
    """Return HUXt-inner and hydro-inner speeds sharing each 1-AU speed."""
    global _SPEED_CONVERSION_LUT
    if _SPEED_CONVERSION_LUT is None:
        outer = np.arange(
            SPEED_LUT_1AU_MIN.to_value(u.km / u.s),
            SPEED_LUT_1AU_MAX.to_value(u.km / u.s)
            + SPEED_LUT_STEP.to_value(u.km / u.s) / 2,
            SPEED_LUT_STEP.to_value(u.km / u.s),
        ) * u.km / u.s
        longitude = (np.arange(len(outer)) + 0.5) * 2 * np.pi / len(outer) * u.rad
        huxt_inner, _ = sin.map_v_inwards(
            outer, SPEED_LUT_OUTER_RADIUS, longitude, RMIN
        )
        hydro_inner, _, _, _ = sin.map_v_inwards_parker(
            outer, SPEED_LUT_OUTER_RADIUS, longitude, RMIN, gamma=GAMMA
        )
        order = np.argsort(huxt_inner.to_value(u.km / u.s))
        _SPEED_CONVERSION_LUT = (
            huxt_inner[order].to(u.km / u.s),
            hydro_inner[order].to(u.km / u.s),
            outer[order].to(u.km / u.s),
        )
    return _SPEED_CONVERSION_LUT


def huxt_inner_to_hydro_inner(speed):
    """Convert HUXt 0.1-AU speeds to hydro speeds with equal 1-AU values."""
    huxt_inner, hydro_inner, _ = speed_conversion_lut()
    values = u.Quantity(speed).to_value(u.km / u.s)
    huxt_values = huxt_inner.to_value(u.km / u.s)
    if np.any(values < huxt_values[0]) or np.any(values > huxt_values[-1]):
        raise ValueError(
            "BRaVDA boundary lies outside the HUXt-to-hydro lookup table: "
            f"{values.min():.1f}-{values.max():.1f} km/s versus "
            f"{huxt_values[0]:.1f}-{huxt_values[-1]:.1f} km/s"
        )
    return np.interp(
        values,
        huxt_values,
        hydro_inner.to_value(u.km / u.s),
    ) * u.km / u.s


def local_wsa_index():
    """Return cached ``(map time, path)`` entries from the local WSA tree."""
    global _LOCAL_WSA_INDEX
    if _LOCAL_WSA_INDEX is None:
        if not LOCAL_WSA_DIRECTORY.is_dir():
            raise FileNotFoundError(f"Local WSA directory not found: {LOCAL_WSA_DIRECTORY}")
        entries = []
        for path in LOCAL_WSA_DIRECTORY.rglob("*.fits"):
            match = re.search(r"vel_(\d{12})R\d+", path.name, re.IGNORECASE)
            if match:
                entries.append((dt.datetime.strptime(match.group(1), "%Y%m%d%H%M"), path))
        if not entries:
            raise FileNotFoundError(f"No timestamped WSA FITS files below {LOCAL_WSA_DIRECTORY}")
        _LOCAL_WSA_INDEX = entries
        print(f"Indexed {len(entries)} local WSA maps below {LOCAL_WSA_DIRECTORY}")
    return _LOCAL_WSA_INDEX


def select_wsa(issue_time):
    """Select a local nearest-time map or download the latest prior map."""
    source = WSA_SOURCE.lower()
    if source == "local":
        _, path = min(local_wsa_index(), key=lambda item: abs(item[0] - issue_time))
        return path
    if source == "download":
        return get_wsa_from_iswa(issue_time, datadir=WSA_DIRECTORY)
    raise ValueError("WSA_SOURCE must be 'local' or 'download'")


def forecast_times(start, end, cadence):
    if cadence <= dt.timedelta(0) or end < start:
        raise ValueError("Require a positive cadence and WINDOW_END >= WINDOW_START")
    count = int((end - start) // cadence) + 1
    return [start + index * cadence for index in range(count)]


def omni_chunk(omni, issue_time):
    start = issue_time - dt.timedelta(days=PERIOD_DAYS)
    mask = (omni["datetime"] >= start) & (omni["datetime"] <= issue_time)
    chunk = omni.loc[mask].copy().reset_index(drop=True)
    if chunk.empty or not np.isfinite(chunk["V"]).any():
        raise ValueError(f"No usable OMNI speeds for forecast at {issue_time}")
    return chunk


def output_path(issue_time):
    directory = OUTPUT_DIRECTORY / f"{issue_time:%Y}"
    return directory / f"wsa_bravda_surf_{issue_time:%Y%m%dT%H%MZ}.csv"


def wsa_bpol_boundary(wsa_path, forecast, longitude, latitude, output_longitude):
    """Sample the signed WSA radial-field plane onto SURF's Carrington grid."""
    with fits.open(wsa_path) as hdul:
        header = hdul[0].header
        raw_br = np.asarray(hdul[0].data[0], dtype=float)
        offset = np.deg2rad(header["CARRLONG"]) % (2 * np.pi)

    lon = longitude.to_value(u.rad)
    lat = latitude.to_value(u.rad)
    source_lon = (lon + offset) % (2 * np.pi)
    order = np.argsort(source_lon)
    br_map = np.vstack([
        np.interp(lon, source_lon[order], row[order], period=2 * np.pi)
        for row in raw_br
    ])

    track_lat = earth_latitudes(forecast, longitude).to_value(u.rad)
    track_br = np.array([
        np.interp(track_lat[index], lat, br_map[:, index])
        for index in range(len(lon))
    ])
    boundary_br = np.interp(
        output_longitude.to_value(u.rad), lon, track_br, period=2 * np.pi
    )
    return np.sign(boundary_br)


def run_one_forecast(issue_time, omni):
    forecast = Time(issue_time, scale="utc")
    finish = forecast + FORECAST_DAYS * u.day

    observations = omni_observations(
        forecast,
        Time(pd.to_datetime(omni["datetime"]).to_numpy()),
        omni["V"].to_numpy() * u.km / u.s,
    )
    wsa_path = select_wsa(issue_time)
    speed_1au, longitude, latitude, map_time = read_wsa(wsa_path)
    speed_inner = deaccelerate_wsa_speeds(speed_1au, RMIN)
    boundary = simpleBRaVDA(
        forecast, observations, speed_inner, longitude, latitude,
        inner_radius=RMIN, settings=BRAVDA_SETTINGS,
    )
    boundary_values = boundary.speed.to_value(u.km / u.s)
    clamped_bin_count = int(np.count_nonzero(
        boundary_values < MIN_BRAVDA_SPEED.to_value(u.km / u.s)
    ))
    clamped_bravda_speed = np.maximum(
        boundary_values, MIN_BRAVDA_SPEED.to_value(u.km / u.s)
    ) * u.km / u.s
    if APPLY_HUXT_TO_HYDRO_CONVERSION:
        surf_boundary_speed = huxt_inner_to_hydro_inner(clamped_bravda_speed)
    else:
        surf_boundary_speed = clamped_bravda_speed
    b_boundary = wsa_bpol_boundary(
        wsa_path, forecast, longitude, latitude, boundary.longitude
    )

    cr_fraction = float(sun.carrington_rotation_number(forecast))
    cr = int(np.floor(cr_fraction))
    cr_lon = 2 * np.pi * (1 - cr_fraction + cr) * u.rad
    run_cr = cr - 1
    run_start = sun.carrington_rotation_time(
        run_cr + (2 * np.pi - cr_lon.to_value(u.rad)) / (2 * np.pi)
    )
    simulation_time = (finish - run_start).to(u.day)
    latitude_times = run_start + np.linspace(0, simulation_time.value, 129) * u.day

    model = surf_model.SURF(
        v_boundary=surf_boundary_speed,
        b_boundary=b_boundary,
        cr_num=run_cr,
        cr_lon_init=cr_lon,
        latitude=np.mean(sun.B0(latitude_times)),
        r_min=RMIN.to(u.solRad),
        r_max=RMAX,
        dr=DR,
        v_max=V_MAX,
        simtime=simulation_time,
        lon_out=0.0 * u.rad,
        dt_scale=DT_SCALE,
        solver=SOLVER,
        gamma=GAMMA,
        track_cmes=False,
        parallel=False,
    )
    model.solve([])

    earth = sa.get_observer_timeseries(model, observer="Earth", suppress_warning=True)
    mask = (
        (earth["time"] >= pd.Timestamp(issue_time))
        & (earth["time"] <= pd.Timestamp(finish.datetime))
    )
    result = earth.loc[mask].reset_index(drop=True)
    details = {
        "wsa_path": str(wsa_path),
        "wsa_source": WSA_SOURCE.lower(),
        "wsa_map_time": map_time,
        "optimizer_success": boundary.optimizer.success,
        "initial_cost": boundary.initial_cost,
        "final_cost": boundary.final_cost,
        "bravda_huxt_boundary_mean": boundary.speed.mean().to_value(u.km / u.s),
        "bravda_clamped_boundary_mean": clamped_bravda_speed.mean().to_value(u.km / u.s),
        "bravda_minimum_speed": MIN_BRAVDA_SPEED.to_value(u.km / u.s),
        "bravda_clamped_bin_count": clamped_bin_count,
        "huxt_to_hydro_conversion_applied": APPLY_HUXT_TO_HYDRO_CONVERSION,
        "surf_boundary_mean": surf_boundary_speed.mean().to_value(u.km / u.s),
        "mean_huxt_to_hydro_factor": (
            surf_boundary_speed.mean() / clamped_bravda_speed.mean()
        ).to_value(u.dimensionless_unscaled),
        "bpol_positive_bins": int(np.count_nonzero(b_boundary > 0)),
        "bpol_negative_bins": int(np.count_nonzero(b_boundary < 0)),
        "bpol_zero_bins": int(np.count_nonzero(b_boundary == 0)),
        "run_start": run_start,
    }
    return result, details


def description(issue_time, omni, details):
    end = issue_time + dt.timedelta(days=FORECAST_DAYS)
    return [
        "WSA + simpleBRaVDA + SURF deterministic forecast Earth time series",
        f"forecast_issue_time_utc: {issue_time:%Y-%m-%dT%H:%M:%SZ}",
        f"forecast_end_time_utc: {end:%Y-%m-%dT%H:%M:%SZ}",
        f"forecast_horizon_days: {FORECAST_DAYS}",
        f"forecast_generation_cadence_hours: {FORECAST_CADENCE.total_seconds() / 3600:g}",
        "run_type: single deterministic 1-D run (not an ensemble forecast)",
        "cone_cmes_included: none",
        f"wsa_file: {details['wsa_path']}",
        f"wsa_source: {details['wsa_source']}",
        f"wsa_map_time_utc: {details['wsa_map_time'].isot}",
        "wsa_speed_calibration: deaccelerated from 1 AU to the SURF inner boundary",
        f"bravda_ensemble_members: {BRAVDA_SETTINGS.n_ensemble}",
        f"bravda_seed: {BRAVDA_SETTINGS.seed}",
        f"bravda_optimizer_success: {details['optimizer_success']}",
        f"bravda_initial_cost: {details['initial_cost']:.12g}",
        f"bravda_final_cost: {details['final_cost']:.12g}",
        "bravda_boundary_speed_convention: HUXt residual-acceleration profile at 0.1 AU",
        f"bravda_minimum_speed_km_per_s: {details['bravda_minimum_speed']:.6g}",
        "bravda_minimum_speed_applied_before_huxt_to_hydro_conversion: True",
        f"bravda_clamped_longitude_bins: {details['bravda_clamped_bin_count']}",
        f"bravda_preclamp_boundary_mean_km_per_s: {details['bravda_huxt_boundary_mean']:.6g}",
        f"bravda_postclamp_boundary_mean_km_per_s: {details['bravda_clamped_boundary_mean']:.6g}",
        f"huxt_to_hydro_conversion_applied: {details['huxt_to_hydro_conversion_applied']}",
        (
            "surf_boundary_conversion: HUXt-to-hydro lookup table matched by common "
            "speed at 215 solar radii"
            if details["huxt_to_hydro_conversion_applied"]
            else "surf_boundary_conversion: none; raw SimpleBRaVDA HUXt-convention speeds used"
        ),
        f"surf_boundary_mean_km_per_s: {details['surf_boundary_mean']:.6g}",
        f"mean_huxt_to_hydro_factor: {details['mean_huxt_to_hydro_factor']:.8g}",
        "bravda_observations: hourly OMNI speed from the preceding Carrington rotation",
        f"omni_chunk_start_utc: {pd.Timestamp(omni['datetime'].min()):%Y-%m-%dT%H:%M:%SZ}",
        f"omni_chunk_end_utc: {pd.Timestamp(omni['datetime'].max()):%Y-%m-%dT%H:%M:%SZ}",
        "omni_icme_removal: same preprocessed DONKI-cleaned series as generate_insitusurf_forecasts.py",
        f"surf_run_start_utc: {details['run_start'].isot}",
        f"solver: {SOLVER}",
        f"gamma: {GAMMA:g}",
        f"rmin_solar_radii: {RMIN.to_value(u.solRad):g}",
        f"rmax_solar_radii: {RMAX.to_value(u.solRad):g}",
        f"dr_solar_radii: {DR.to_value(u.solRad):g}",
        "magnetic_field_boundary_included: True",
        "magnetic_field_boundary_source: WSA signed radial-field map sampled along the Earth-latitude track",
        f"wsa_bpol_positive_bins: {details['bpol_positive_bins']}",
        f"wsa_bpol_negative_bins: {details['bpol_negative_bins']}",
        f"wsa_bpol_zero_bins: {details['bpol_zero_bins']}",
        "data_below: forecast interval only; reconstruction/spinup samples excluded",
    ]


def store_forecast(path, issue_time, omni, series, details):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        for line in description(issue_time, omni, details):
            stream.write(f"# {line}\n")
        series.to_csv(stream, index=False)
    temporary.replace(path)


def main():
    issue_times = forecast_times(WINDOW_START, WINDOW_END, FORECAST_CADENCE)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    all_omni = prepare_omni(issue_times[0], issue_times[-1])

    for number, issue_time in enumerate(issue_times, start=1):
        path = output_path(issue_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not OVERWRITE_EXISTING:
            print(f"[{number}/{len(issue_times)}] Skipping existing {path.name}")
            continue
        print(f"[{number}/{len(issue_times)}] Forecast at {issue_time}")
        chunk = omni_chunk(all_omni, issue_time)
        series, details = run_one_forecast(issue_time, chunk)
        store_forecast(path, issue_time, chunk, series, details)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
