"""Generate deterministic 1-D InSitu-SURF hindcasts with DONKI cone CMEs.

Identical to generate_insitusurf_forecasts.py (same ICME-cleaned OMNI
boundary and model settings) except that:

* each run starts CME_BUFFER_DAYS before the issue time and runs to
  FORECAST_DAYS after it; only the forecast interval is stored;
* every DONKI cone CME (CMEAnalysis, most-accurate leading-edge fits) that
  reaches 21.5 solar radii in the CME_BUFFER_DAYS up to and including the
  issue time is inserted, with SURF's default ConeCME properties.

DONKI analyses are downloaded once for the complete window, before any runs.
"""

import datetime as dt
import json
from pathlib import Path
from urllib.request import urlopen

import astropy.units as u
import numpy as np
import pandas as pd

import surf.surf as surf_model
import surf.surf_analysis as sa
import surf.surf_inputs as sin
import surf.surf_insitu as sinsit

import generate_insitusurf_forecasts as base


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
WINDOW_START = base.WINDOW_START
WINDOW_END = base.WINDOW_END
FORECAST_CADENCE = base.FORECAST_CADENCE
FORECAST_DAYS = base.FORECAST_DAYS

# The run starts this long before the issue time, and CMEs reaching 21.5 rS
# in this interval (up to the issue time) are included.
CME_BUFFER_DAYS = 5

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "insitu_cone_surf"
OVERWRITE_EXISTING = False

# DONKI CMEAnalysis query (the defaults of surf_inputs.get_DONKI_coneCMEs).
DONKI_CME_MOST_ACCURATE_ONLY = "true"
DONKI_CME_CATALOG = "ALL"
DONKI_CME_FEATURE = "LE"
DONKI_CME_CHUNK_DAYS = 30  # the API rejects requests much over 60 days
CME_ANALYSIS_URL = "https://ccmc.gsfc.nasa.gov/DONKI-API/get/CMEAnalysis"

# CME properties not set by DONKI take SURF's ConeCME defaults, as applied by
# surf_inputs.cone_dict_to_cme_list (thickness 0, initial height 21.5 rS).
_DEFAULT_CME = surf_model.ConeCME()


def download_donki_cmes(start, end):
    """All DONKI cone-CME analyses between two dates, as one DataFrame.

    Requests are made in short chunks; analyses returned by two neighbouring
    chunks are kept once.
    """
    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + dt.timedelta(days=DONKI_CME_CHUNK_DAYS), end)
        url = (
            f"{CME_ANALYSIS_URL}?startDate={chunk_start:%Y-%m-%d}"
            f"&endDate={chunk_end:%Y-%m-%d}"
            f"&mostAccurateOnly={DONKI_CME_MOST_ACCURATE_ONLY}"
            f"&feature={DONKI_CME_FEATURE}&catalog={DONKI_CME_CATALOG}"
        )
        print(f"Downloading DONKI CMEs: {chunk_start:%Y-%m-%d} through {chunk_end:%Y-%m-%d}")
        with urlopen(url) as response:
            payload = response.read().decode("utf-8")
        frames.append(pd.DataFrame(json.loads(payload) if payload.strip() else []))
        chunk_start = chunk_end

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["time21_5"])
    cmes = pd.concat(frames, ignore_index=True)
    cmes = cmes.dropna(subset=["time21_5", "latitude", "longitude", "halfAngle", "speed"])
    cmes = cmes.drop_duplicates(subset=["link"]).copy()
    cmes["time21_5"] = pd.to_datetime(cmes["time21_5"], format="%Y-%m-%dT%H:%MZ")
    return cmes.sort_values("time21_5").reset_index(drop=True)


def cmes_for_forecast(all_cmes, issue_time):
    """CMEs reaching 21.5 rS in the buffer interval up to the issue time."""
    start = pd.Timestamp(issue_time - dt.timedelta(days=CME_BUFFER_DAYS))
    mask = (all_cmes["time21_5"] >= start) & (all_cmes["time21_5"] <= pd.Timestamp(issue_time))
    return all_cmes.loc[mask].reset_index(drop=True)


def cone_cme_list(model, cmes):
    """ConeCME objects for one model, via SURF's DONKI conversion."""
    if cmes.empty:
        return []
    cme_params = {
        number: {
            "ldates": f"{row.time21_5:%Y-%m-%dT%H:%M}",
            "lat": row.latitude,
            "lon": row.longitude,
            "rmajor": row.halfAngle,
            "vcld": row.speed,
        }
        for number, row in enumerate(cmes.itertuples(), start=1)
    }
    return sin.cone_dict_to_cme_list(model, cme_params)


def output_path(issue_time):
    annual_directory = OUTPUT_DIRECTORY / f"{issue_time:%Y}"
    return annual_directory / f"insitu_cone_surf_{issue_time:%Y%m%dT%H%MZ}.csv"


def required_header():
    """Header lines identifying files made with the current method."""
    return [
        *base.required_header(),
        f"cme_buffer_days: {CME_BUFFER_DAYS}",
        (
            f"cone_cmes: DONKI CMEAnalysis (mostAccurateOnly={DONKI_CME_MOST_ACCURATE_ONLY}, "
            f"feature={DONKI_CME_FEATURE}, catalog={DONKI_CME_CATALOG}) reaching 21.5 rS "
            f"in the {CME_BUFFER_DAYS} days up to the issue time"
        ),
    ]


def model_v_max(cmes):
    """Model speed limit: the standard one, raised if a CME is faster.

    SURF rejects CMEs faster than v_max (its CFL limit), so for such runs the
    limit is raised to the next 100 km/s above the fastest CME.
    """
    standard = base.V_MAX.to_value(u.km / u.s)
    fastest = cmes["speed"].max() if not cmes.empty else 0.0
    if fastest < standard:
        return base.V_MAX
    return (np.floor(fastest / 100.0) + 1) * 100.0 * u.km / u.s


def run_one_forecast(issue_time, omni_chunk, cmes):
    """One deterministic 1-D forecast per solver, with cone CMEs."""
    forecast_end = issue_time + dt.timedelta(days=FORECAST_DAYS)
    series_by_solver = {}
    for solver in base.SOLVERS:
        model = sinsit.omniSURF_forecast(
            issue_time,
            simtime=(CME_BUFFER_DAYS + FORECAST_DAYS) * u.day,
            buffertime=CME_BUFFER_DAYS * u.day,
            rmin=base.RMIN,
            rmax=base.RMAX,
            dr=base.DR,
            nlon=base.NLON,
            v_max=model_v_max(cmes),
            dt_scale=base.DT_SCALE,
            solver=solver,
            gamma=base.GAMMA,
            run_2d=False,
            track_cmes=False,
            include_b_boundary=base.INCLUDE_B_BOUNDARY,
            icme_list=base.ICME_LIST,
            omni_input=omni_chunk,
        )
        model.solve(cone_cme_list(model, cmes))

        earth_series = sa.get_observer_timeseries(model, observer="Earth")
        forecast_mask = (
            (earth_series["time"] >= pd.Timestamp(issue_time))
            & (earth_series["time"] <= pd.Timestamp(forecast_end))
        )
        series_by_solver[solver] = earth_series.loc[forecast_mask]
    return base.combine_solver_series(series_by_solver)


def cme_description(issue_time, cmes):
    """Header lines describing the CME settings and each included CME."""
    run_start = issue_time - dt.timedelta(days=CME_BUFFER_DAYS)
    lines = [
        f"surf_run_start_utc: {run_start:%Y-%m-%dT%H:%M:%SZ}",
        f"spinup_buffer_days: {CME_BUFFER_DAYS}",
        (
            f"cme_selection_window_utc: {run_start:%Y-%m-%dT%H:%M:%SZ} to "
            f"{issue_time:%Y-%m-%dT%H:%M:%SZ} (time at 21.5 rS)"
        ),
        "cme_conversion: surf_inputs.cone_dict_to_cme_list (full width = 2 x DONKI halfAngle)",
        (
            "cme_default_properties: "
            f"initial_height_solar_radii=21.5, thickness_solar_radii=0, "
            f"profile_type={_DEFAULT_CME.profile_type}, "
            f"cme_fixed_duration={_DEFAULT_CME.cme_fixed_duration}, "
            f"fixed_duration_hours={_DEFAULT_CME.fixed_duration.to_value(u.hour):g}, "
            f"cme_expansion={_DEFAULT_CME.cme_expansion}, "
            f"density_per_cm3={(_DEFAULT_CME.cme_density / surf_model.m_p).to_value(u.cm ** -3):g}, "
            f"temperature_K={_DEFAULT_CME.cme_temperature.to_value(u.K):g}"
        ),
        "cme_submission_time_filter: none (analyses submitted after the issue time are included)",
        f"n_cone_cmes: {len(cmes)}",
    ]
    for number, row in enumerate(cmes.itertuples(), start=1):
        lines.append(
            f"cme_{number:02d}: id={row.associatedCMEID}, "
            f"time21_5={row.time21_5:%Y-%m-%dT%H:%MZ}, lat_deg={row.latitude:g}, "
            f"lon_deg={row.longitude:g}, half_angle_deg={row.halfAngle:g}, "
            f"speed_km_per_s={row.speed:g}, type={row.type}, "
            f"submitted={row.submissionTime}"
        )
    return lines


def description(issue_time, omni_chunk, cmes):
    """InSitu-SURF header lines, adapted for cone CMEs and the run buffer."""
    replaced = ("spinup_buffer_days:", "cone_cmes_included:")
    lines = [*required_header()]
    for line in base.run_description(issue_time, omni_chunk):
        if line == "InSitu-SURF deterministic forecast Earth time series":
            lines.insert(0, "InSitu-SURF + DONKI cone CMEs deterministic forecast Earth time series")
        elif line.startswith(replaced) or line in lines:
            continue
        elif line.startswith("v_max_km_per_s:"):
            v_max = model_v_max(cmes).to_value(u.km / u.s)
            note = "" if v_max == base.V_MAX.to_value(u.km / u.s) else \
                " (raised from the standard limit to exceed the fastest CME)"
            lines.append(f"v_max_km_per_s: {v_max:g}{note}")
        elif line.startswith("data_below:"):
            lines.extend(cme_description(issue_time, cmes))
            lines.append(line)
        else:
            lines.append(line)
    return lines


def store_forecast(path, issue_time, omni_chunk, cmes, forecast_series):
    """Atomically store metadata and the forecast Earth time series as CSV."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        for line in description(issue_time, omni_chunk, cmes):
            stream.write(f"# {line}\n")
        forecast_series.to_csv(stream, index=False)
    temporary_path.replace(path)


def main():
    issue_times = base.forecast_times(WINDOW_START, WINDOW_END, FORECAST_CADENCE)
    if not issue_times:
        return

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    omni = base.prepare_omni(issue_times[0], issue_times[-1])
    # DONKI is queried by CME start time; start early so CMEs that reach
    # 21.5 rS inside the first buffer are not missed.
    all_cmes = download_donki_cmes(
        issue_times[0] - dt.timedelta(days=CME_BUFFER_DAYS + 2),
        issue_times[-1] + dt.timedelta(days=1),
    )
    print(f"Downloaded {len(all_cmes)} unique DONKI cone CMEs")

    for run_number, issue_time in enumerate(issue_times, start=1):
        path = output_path(issue_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        if (not OVERWRITE_EXISTING and path.exists()
                and set(required_header()) <= base.stored_header(path)):
            print(f"[{run_number}/{len(issue_times)}] Skipping existing {path.name}")
            continue

        cmes = cmes_for_forecast(all_cmes, issue_time)
        print(f"[{run_number}/{len(issue_times)}] Forecast at {issue_time} "
              f"with {len(cmes)} cone CMEs")
        chunk = base.omni_chunk_for_forecast(omni, issue_time)
        forecast_series = run_one_forecast(issue_time, chunk, cmes)
        store_forecast(path, issue_time, chunk, cmes, forecast_series)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
