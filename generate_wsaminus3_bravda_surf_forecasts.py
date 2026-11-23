"""Generate WSA-BRaVDA-SURF forecasts from WSA maps 3 days older than usual.

Identical to ``generate_wsa_bravda_surf_forecasts.py`` except for the WSA map:
each forecast uses the newest map available at (issue time - WSA_LAG) rather
than at the issue time. The OMNI assimilation window (preceding Carrington
rotation up to the issue time), the SURF set-up and the forecast interval are
unchanged, so differences from the standard runs isolate the effect of WSA map
age. Output goes to wsaminus3_bravda_surf_forecasts/<year>/.
"""

import datetime as dt
from pathlib import Path

import generate_wsa_bravda_surf_forecasts as base
from generate_insitusurf_forecasts import prepare_omni, stored_header


# ---------------------------------------------------------------------------
# User settings (all other settings are taken from the standard generator)
# ---------------------------------------------------------------------------
WINDOW_START = base.WINDOW_START
WINDOW_END = base.WINDOW_END
FORECAST_CADENCE = base.FORECAST_CADENCE
WSA_LAG = dt.timedelta(days=3)
OVERWRITE_EXISTING = False
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "wsaminus3_bravda_surf_forecasts"

WSA_LAG_DAYS = WSA_LAG.total_seconds() / 86400
WSA_SELECTION = (
    f"wsa_selection: newest map at or before {WSA_LAG_DAYS:g} days prior to the issue time"
)


def output_path(issue_time):
    directory = OUTPUT_DIRECTORY / f"{issue_time:%Y}"
    return directory / f"wsaminus3_bravda_surf_{issue_time:%Y%m%dT%H%MZ}.csv"


def select_lagged_wsa(issue_time):
    """Newest WSA map available WSA_LAG before ``issue_time``.

    WSA_MAX_AGE and the version preference apply relative to the lagged time.
    """
    return base.select_wsa(issue_time - WSA_LAG)


def required_header():
    """Header lines identifying files made with the current method."""
    standard = base.required_header()
    return [WSA_SELECTION if line == base.WSA_SELECTION else line for line in standard]


def description(issue_time, omni, details):
    lines = []
    for line in base.description(issue_time, omni, details):
        if line == base.WSA_SELECTION:
            lines.extend([
                WSA_SELECTION,
                f"wsa_lag_days: {WSA_LAG_DAYS:g}",
                f"wsa_selection_time_utc: {issue_time - WSA_LAG:%Y-%m-%dT%H:%M:%SZ}",
            ])
        else:
            lines.append(line)
    return lines


def store_forecast(path, issue_time, omni, series, details):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        for line in description(issue_time, omni, details):
            stream.write(f"# {line}\n")
        series.to_csv(stream, index=False)
    temporary.replace(path)


def main():
    issue_times = base.forecast_times(WINDOW_START, WINDOW_END, FORECAST_CADENCE)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    all_omni = prepare_omni(issue_times[0], issue_times[-1])

    for number, issue_time in enumerate(issue_times, start=1):
        path = output_path(issue_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        if (not OVERWRITE_EXISTING and path.exists()
                and set(required_header()) <= stored_header(path)):
            print(f"[{number}/{len(issue_times)}] Skipping existing {path.name}")
            continue
        print(f"[{number}/{len(issue_times)}] Forecast at {issue_time} "
              f"(WSA map from <= {issue_time - WSA_LAG})")
        try:
            wsa_path = select_lagged_wsa(issue_time)
        except FileNotFoundError as error:
            print(f"Skipping {issue_time}: {error}")
            continue
        print(f"Using {wsa_path}")
        chunk = base.omni_chunk(all_omni, issue_time)
        series, details = base.run_one_forecast(issue_time, chunk, wsa_path)
        store_forecast(path, issue_time, chunk, series, details)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
