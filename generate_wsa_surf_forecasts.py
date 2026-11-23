"""Generate daily deterministic WSA + SURF forecasts (no data assimilation).

The no-DA baseline for ``generate_wsa_bravda_surf_forecasts.py``: same WSA map
selection (newest map at or before the issue time), Earth-latitude sampling,
magnetic-polarity boundary, spin-up and SURF settings, but the WSA speeds are
used directly. Each forecast is run with the HUXt and hydro solvers and both
are stored in one CSV (solver-suffixed columns).

WSA speeds are calibrated to 1 AU, so they are mapped in to RMIN separately for
each solver with SURF's own profile for that solver: the HUXt residual-
acceleration law for HUXt and the Parker solution for hydro. As in SURF's
WSA input routines, only the speed is mapped; longitudes are kept.
"""

import datetime as dt
from pathlib import Path

import astropy.units as u
from astropy.time import Time
import numpy as np

import surf.surf_inputs as sin

from generate_insitusurf_forecasts import stored_header
from generate_wsa_bravda_surf_forecasts import (
    DR, FORECAST_DAYS, GAMMA, RMAX, RMIN, SOLVERS, WSA_MAX_AGE, WSA_SELECTION,
    WSA_SOURCE, WSA_VERSIONS, earth_latitudes, forecast_times, iswa_url, read_wsa,
    run_surf_solvers, select_wsa, wsa_bpol_boundary,
)


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

WINDOW_START = dt.datetime(2019, 1, 1)
WINDOW_END = dt.datetime(2026, 1, 1)
FORECAST_CADENCE = dt.timedelta(days=1)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "wsa_surf_forecasts"
OVERWRITE_EXISTING = False

N_LONGITUDE = 128                       # SURF boundary grid, as for BRaVDA
WSA_CALIBRATION_RADIUS = 215 * u.solRad  # radius the WSA speeds represent


def output_path(issue_time):
    directory = OUTPUT_DIRECTORY / f"{issue_time:%Y}"
    return directory / f"wsa_surf_{issue_time:%Y%m%dT%H%MZ}.csv"


def earth_track_speed(speed_map, longitude, latitude, forecast, grid):
    """Sample the WSA speed map along Earth's latitude track onto ``grid``.

    The track is the latitude Earth had when each longitude faced it over the
    preceding rotation, as used for the BRaVDA prior and the polarity boundary.
    """
    lon = longitude.to_value(u.rad)
    lat = latitude.to_value(u.rad)
    speed = speed_map.to_value(u.km / u.s)
    track_lat = earth_latitudes(forecast, longitude).to_value(u.rad)
    track = np.array([np.interp(track_lat[i], lat, speed[:, i]) for i in range(len(lon))])
    return np.interp(grid.to_value(u.rad), lon, track, period=2 * np.pi) * u.km / u.s


def inner_boundary(speed_1au, grid, solver):
    """Map 1-AU-calibrated speeds in to RMIN with ``solver``'s profile."""
    if solver == "huxt":
        speed, _ = sin.map_v_inwards(speed_1au, WSA_CALIBRATION_RADIUS, grid, RMIN)
    elif solver == "hydro":
        speed = sin.map_v_inwards_parker(
            speed_1au, WSA_CALIBRATION_RADIUS, grid, RMIN, gamma=GAMMA
        )[0]
    else:
        raise ValueError(f"No WSA inward mapping defined for solver {solver!r}")
    speed = u.Quantity(speed).to(u.km / u.s)
    if not np.all(np.isfinite(speed.value)):
        raise ValueError(f"Non-finite {solver} inner-boundary speeds")
    return speed


def run_one_forecast(issue_time, wsa_path=None):
    forecast = Time(issue_time, scale="utc")
    if wsa_path is None:
        wsa_path = select_wsa(issue_time)
    speed_1au, longitude, latitude, map_time = read_wsa(wsa_path)
    if map_time > forecast:
        raise ValueError(
            f"{wsa_path.name}: FITS map time {map_time.isot} is after the issue time"
        )

    grid = (np.arange(N_LONGITUDE) + 0.5) * 2 * np.pi / N_LONGITUDE * u.rad
    track_1au = earth_track_speed(speed_1au, longitude, latitude, forecast, grid)
    boundary_speeds = {solver: inner_boundary(track_1au, grid, solver) for solver in SOLVERS}
    b_boundary = wsa_bpol_boundary(wsa_path, forecast, longitude, latitude, grid)

    series, run_start = run_surf_solvers(issue_time, boundary_speeds, b_boundary)
    details = {
        "wsa_path": str(wsa_path),
        "wsa_url": iswa_url(wsa_path),
        "wsa_version": Path(wsa_path).parent.name,
        "wsa_map_time": map_time,
        "track_mean_1au": track_1au.mean().to_value(u.km / u.s),
        "boundary_means": {
            solver: speed.mean().to_value(u.km / u.s)
            for solver, speed in boundary_speeds.items()
        },
        "bpol_positive_bins": int(np.count_nonzero(b_boundary > 0)),
        "bpol_negative_bins": int(np.count_nonzero(b_boundary < 0)),
        "bpol_zero_bins": int(np.count_nonzero(b_boundary == 0)),
        "run_start": run_start,
    }
    return series, details


def required_header():
    """Header lines identifying files made with the current method."""
    return [
        "WSA + SURF deterministic forecast Earth time series (no data assimilation)",
        f"solvers: {', '.join(SOLVERS)}",
        WSA_SELECTION,
    ]


BOUNDARY_METHOD = {
    "huxt": "HUXt residual-acceleration profile (surf_inputs.map_v_inwards)",
    "hydro": "Parker solution (surf_inputs.map_v_inwards_parker)",
}


def description(issue_time, details):
    end = issue_time + dt.timedelta(days=FORECAST_DAYS)
    lines = [
        *required_header(),
        "columns: shared time/position columns, then solver-suffixed outputs (e.g. vsw_huxt, vsw_hydro)",
        f"forecast_issue_time_utc: {issue_time:%Y-%m-%dT%H:%M:%SZ}",
        f"forecast_end_time_utc: {end:%Y-%m-%dT%H:%M:%SZ}",
        f"forecast_horizon_days: {FORECAST_DAYS}",
        f"forecast_generation_cadence_hours: {FORECAST_CADENCE.total_seconds() / 3600:g}",
        "run_type: single deterministic 1-D run per solver (not an ensemble forecast)",
        "cone_cmes_included: none",
        f"wsa_file: {details['wsa_path']}",
        f"wsa_ccmc_url: {details['wsa_url']}",
        f"wsa_version: {details['wsa_version']}",
        f"wsa_source: {WSA_SOURCE.lower()}",
        f"wsa_map_time_utc: {details['wsa_map_time'].isot}",
        f"wsa_version_preference: {', '.join(WSA_VERSIONS)}",
        f"wsa_max_age_days: {WSA_MAX_AGE.total_seconds() / 86400:g}",
        "wsa_sampling: speed and polarity along Earth's latitude track over the preceding rotation",
        f"wsa_speed_calibration_radius_solar_radii: {WSA_CALIBRATION_RADIUS.to_value(u.solRad):g}",
        f"wsa_track_mean_1au_km_per_s: {details['track_mean_1au']:.6g}",
        "wsa_inward_mapping: speed only, longitudes kept",
    ]
    for solver in SOLVERS:
        lines += [
            f"surf_boundary_mapping_{solver}: {BOUNDARY_METHOD[solver]}",
            f"surf_boundary_mean_km_per_s_{solver}: {details['boundary_means'][solver]:.6g}",
        ]
    lines += [
        f"surf_run_start_utc: {details['run_start'].isot}",
        f"gamma: {GAMMA:g}",
        f"rmin_solar_radii: {RMIN.to_value(u.solRad):g}",
        f"rmax_solar_radii: {RMAX.to_value(u.solRad):g}",
        f"dr_solar_radii: {DR.to_value(u.solRad):g}",
        f"boundary_longitude_bins: {N_LONGITUDE}",
        "magnetic_field_boundary_included: True",
        f"wsa_bpol_positive_bins: {details['bpol_positive_bins']}",
        f"wsa_bpol_negative_bins: {details['bpol_negative_bins']}",
        f"wsa_bpol_zero_bins: {details['bpol_zero_bins']}",
        "data_below: forecast interval only; reconstruction/spinup samples excluded",
    ]
    return lines


def store_forecast(path, issue_time, series, details):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        for line in description(issue_time, details):
            stream.write(f"# {line}\n")
        series.to_csv(stream, index=False)
    temporary.replace(path)


def main():
    issue_times = forecast_times(WINDOW_START, WINDOW_END, FORECAST_CADENCE)
    for number, issue_time in enumerate(issue_times, start=1):
        path = output_path(issue_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        if (not OVERWRITE_EXISTING and path.exists()
                and set(required_header()) <= stored_header(path)):
            print(f"[{number}/{len(issue_times)}] Skipping existing {path.name}")
            continue
        print(f"[{number}/{len(issue_times)}] Forecast at {issue_time}")
        try:
            wsa_path = select_wsa(issue_time)
        except FileNotFoundError as error:
            print(f"Skipping {issue_time}: {error}")
            continue
        print(f"Using {wsa_path}")
        series, details = run_one_forecast(issue_time, wsa_path)
        store_forecast(path, issue_time, series, details)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
