"""Generate a sequence of deterministic 1-D InSitu-SURF forecasts.

The OMNI hourly data are downloaded once for the complete experiment, then a
separate look-back chunk is passed to each forecast.  This is deliberately a
single-run workflow: no ambient or CME ensemble members are generated.
"""

import datetime as dt
from pathlib import Path

import astropy.units as u
import pandas as pd

import surf.surf_analysis as sa
import surf.surf_insitu as sinsit


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
# Naive datetimes are interpreted as UTC by this script and by SURF.
WINDOW_START = dt.datetime(2019, 1, 1, 0, 0)
WINDOW_END = dt.datetime(2019, 12, 31, 0, 0)

FORECAST_CADENCE = dt.timedelta(days=1)
FORECAST_DAYS = 10
BUFFER_DAYS = 0

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "insitusurf_forecasts"
OVERWRITE_EXISTING = False

# Model settings. run_2d=False below makes every forecast a single 1-D run.
SOLVER = "hydro"
RMIN = 21.5 * u.solRad
RMAX = 240.0 * u.solRad
DR = 1.5 * u.solRad
NLON = 128
DT_SCALE = 4
V_MAX = 3000.0 * u.km / u.s
GAMMA = 1.5
INCLUDE_B_BOUNDARY = True

# ICMEs are removed from the OMNI boundary before forecasting. Set this to
# None to retain them, or to another catalogue accepted by removeICMEs().
ICME_LIST = "DONKI"
PRE_ICME_BUFFER_DAYS = 0.2
POST_ICME_BUFFER_DAYS = 1.0
DONKI_MIN_QUALITY = 1
DONKI_CHUNK_DAYS = 60
DONKI_ASSUMED_ICME_DURATION_DAYS = 1.5

# omniSURF_forecast uses the preceding Carrington rotation. 28 days provides
# a small margin around its 27.27-day look-back requirement.
OMNI_LOOKBACK = dt.timedelta(days=28)


def forecast_times(start, end, cadence):
    """Return forecast issue times, including *end* when it is on the grid."""
    if cadence <= dt.timedelta(0):
        raise ValueError("FORECAST_CADENCE must be positive")
    if end < start:
        raise ValueError("WINDOW_END must not be earlier than WINDOW_START")

    times = []
    current = start
    while current <= end:
        times.append(current)
        current += cadence
    return times


def download_donki_icmes(start, end):
    """Download a long DONKI interval as API-safe chunks, before any runs."""
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + dt.timedelta(days=DONKI_CHUNK_DAYS), end)
        print(f"Downloading DONKI ICMEs: {chunk_start} through {chunk_end}")
        try:
            chunk = sinsit.get_DONKI_ICMEs(
                chunk_start,
                chunk_end,
                ICME_duration=DONKI_ASSUMED_ICME_DURATION_DAYS * u.day,
                min_quality=DONKI_MIN_QUALITY,
            )
        except KeyError as error:
            # SURF's DONKI reader currently indexes the ``location`` column
            # unconditionally. An API response of [] creates a DataFrame with
            # no columns, so that valid "no events" response raises here.
            if error.args != ("location",):
                raise
            print("No DONKI ICMEs returned for this interval")
            chunk = pd.DataFrame(columns=["Shock_time", "ICME_end"])
        chunks.append(chunk)
        chunk_start = chunk_end

    nonempty_chunks = [chunk for chunk in chunks if not chunk.empty]
    if not nonempty_chunks:
        return pd.DataFrame(columns=["Shock_time", "ICME_end"])

    # Consecutive API requests share an endpoint date, so remove any event
    # returned in both neighbouring chunks.
    return (
        pd.concat(nonempty_chunks, ignore_index=True)
        .drop_duplicates(subset=["Shock_time", "ICME_end"])
        .sort_values("Shock_time")
        .reset_index(drop=True)
    )


def remove_downloaded_icmes(omni, icmes):
    """Apply one pre-downloaded ICME catalogue to the complete OMNI frame."""
    if icmes.empty:
        return omni.copy()

    cleaned = omni.copy()
    pre_buffer = pd.Timedelta(days=PRE_ICME_BUFFER_DAYS)
    post_buffer = pd.Timedelta(days=POST_ICME_BUFFER_DAYS)
    for event_number, event in icmes.iterrows():
        removal_start = pd.Timestamp(event["Shock_time"]) - pre_buffer
        removal_end = pd.Timestamp(event["ICME_end"]) + post_buffer
        mask = (
            (cleaned["datetime"] >= removal_start)
            & (cleaned["datetime"] <= removal_end)
        )
        if mask.any():
            cleaned.loc[mask, ["V", "BX_GSE"]] = float("nan")
            print(f"Removing DONKI ICME {event_number + 1}/{len(icmes)}")

    # Match removeICMEs(..., interp_gaps=True), but interpolate once over the
    # complete experiment rather than independently at catalogue boundaries.
    cleaned = cleaned.set_index("datetime")
    cleaned[["V", "BX_GSE"]] = (
        cleaned[["V", "BX_GSE"]].interpolate(method="time").ffill().bfill()
    )
    return cleaned.reset_index()


def prepare_omni(first_forecast, last_forecast):
    """Download all OMNI and ICME inputs before any model run starts."""
    download_start = first_forecast - OMNI_LOOKBACK
    download_end = last_forecast
    print(f"Downloading one OMNI block: {download_start} through {download_end}")
    omni = sinsit.get_omni(download_start, download_end)

    if ICME_LIST == "DONKI":
        # Match SURF's normal catalogue padding while keeping every DONKI
        # request at or below its 60-day API limit.
        catalogue_start = download_start - dt.timedelta(days=5)
        catalogue_end = download_end + dt.timedelta(days=5)
        icmes = download_donki_icmes(catalogue_start, catalogue_end)
        print(f"Downloaded {len(icmes)} unique DONKI ICMEs")
        omni = remove_downloaded_icmes(omni, icmes)
    elif ICME_LIST is not None:
        omni = sinsit.removeICMEs(
            omni,
            icme_list=ICME_LIST,
            pre_icme_buffer=PRE_ICME_BUFFER_DAYS,
            post_icme_buffer=POST_ICME_BUFFER_DAYS,
        )

    # omniSURF_forecast fills ordinary data gaps after receiving each chunk.
    return omni


def omni_chunk_for_forecast(omni, issue_time):
    """Select only the OMNI data needed by one forecast issue time."""
    start = issue_time - OMNI_LOOKBACK
    mask = (omni["datetime"] >= start) & (omni["datetime"] <= issue_time)
    chunk = omni.loc[mask].copy().reset_index(drop=True)
    if chunk.empty:
        raise ValueError(f"No OMNI data available for forecast at {issue_time}")
    return chunk


def output_path(issue_time):
    annual_directory = OUTPUT_DIRECTORY / f"{issue_time:%Y}"
    return annual_directory / f"insitusurf_{issue_time:%Y%m%dT%H%MZ}.csv"


def run_one_forecast(issue_time, omni_chunk):
    """Initialize, solve, and return one deterministic 1-D forecast."""
    total_simulation_days = BUFFER_DAYS + FORECAST_DAYS
    model = sinsit.omniSURF_forecast(
        issue_time,
        simtime=total_simulation_days * u.day,
        buffertime=BUFFER_DAYS * u.day,
        rmin=RMIN,
        rmax=RMAX,
        dr=DR,
        nlon=NLON,
        v_max=V_MAX,
        dt_scale=DT_SCALE,
        solver=SOLVER,
        gamma=GAMMA,
        run_2d=False,
        track_cmes=False,
        include_b_boundary=INCLUDE_B_BOUNDARY,
        icme_list=ICME_LIST,
        omni_input=omni_chunk,
    )
    model.solve([])

    earth_series = sa.get_observer_timeseries(model, observer="Earth")
    forecast_end = issue_time + dt.timedelta(days=FORECAST_DAYS)
    forecast_mask = (
        (earth_series["time"] >= pd.Timestamp(issue_time))
        & (earth_series["time"] <= pd.Timestamp(forecast_end))
    )
    forecast_series = earth_series.loc[forecast_mask].reset_index(drop=True)
    return forecast_series


def run_description(issue_time, omni_chunk):
    """Return human-readable comment lines describing one stored run."""
    forecast_end = issue_time + dt.timedelta(days=FORECAST_DAYS)
    cadence_hours = FORECAST_CADENCE.total_seconds() / 3600.0
    return [
        "InSitu-SURF deterministic forecast Earth time series",
        f"forecast_issue_time_utc: {issue_time:%Y-%m-%dT%H:%M:%SZ}",
        f"forecast_end_time_utc: {forecast_end:%Y-%m-%dT%H:%M:%SZ}",
        f"forecast_horizon_days: {FORECAST_DAYS}",
        f"forecast_generation_cadence_hours: {cadence_hours:g}",
        f"spinup_buffer_days: {BUFFER_DAYS}",
        "run_type: single deterministic 1-D run (not an ensemble)",
        "cone_cmes_included: none",
        f"solver: {SOLVER}",
        f"gamma: {GAMMA:g}",
        f"rmin_solar_radii: {RMIN.to_value(u.solRad):g}",
        f"rmax_solar_radii: {RMAX.to_value(u.solRad):g}",
        f"dr_solar_radii: {DR.to_value(u.solRad):g}",
        f"nlon_boundary_grid: {NLON}",
        f"dt_scale: {DT_SCALE}",
        f"v_max_km_per_s: {V_MAX.to_value(u.km / u.s):g}",
        f"magnetic_field_boundary_included: {INCLUDE_B_BOUNDARY}",
        "ambient_boundary_source: hourly OMNI data from the preceding solar rotation",
        f"omni_chunk_start_utc: {pd.Timestamp(omni_chunk['datetime'].min()):%Y-%m-%dT%H:%M:%SZ}",
        f"omni_chunk_end_utc: {pd.Timestamp(omni_chunk['datetime'].max()):%Y-%m-%dT%H:%M:%SZ}",
        f"icme_catalogue: {ICME_LIST}",
        f"donki_minimum_quality: {DONKI_MIN_QUALITY}",
        f"donki_assumed_icme_duration_days: {DONKI_ASSUMED_ICME_DURATION_DAYS:g}",
        f"icme_pre_shock_buffer_days: {PRE_ICME_BUFFER_DAYS:g}",
        f"icme_post_end_buffer_days: {POST_ICME_BUFFER_DAYS:g}",
        (
            "icme_removal: DONKI shock-to-assumed-end intervals, including the stated "
            "buffers, were removed from OMNI V and BX_GSE before any model runs; "
            "removed samples were time-interpolated once across the complete OMNI dataset"
        ),
        (
            f"donki_download: completed before any model runs in chunks of at most "
            f"{DONKI_CHUNK_DAYS} days; duplicate boundary events removed"
        ),
        "data_below: forecast interval only; model spinup samples excluded",
    ]


def store_forecast(path, issue_time, omni_chunk, forecast_series):
    """Atomically store metadata and the forecast Earth time series as CSV."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        for line in run_description(issue_time, omni_chunk):
            stream.write(f"# {line}\n")
        forecast_series.to_csv(stream, index=False)
    temporary_path.replace(path)


def main():
    issue_times = forecast_times(WINDOW_START, WINDOW_END, FORECAST_CADENCE)
    if not issue_times:
        return

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    omni = prepare_omni(issue_times[0], issue_times[-1])

    for run_number, issue_time in enumerate(issue_times, start=1):
        path = output_path(issue_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not OVERWRITE_EXISTING:
            print(f"[{run_number}/{len(issue_times)}] Skipping existing {path.name}")
            continue

        print(f"[{run_number}/{len(issue_times)}] Forecast at {issue_time}")
        chunk = omni_chunk_for_forecast(omni, issue_time)
        forecast_series = run_one_forecast(issue_time, chunk)
        store_forecast(path, issue_time, chunk, forecast_series)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
