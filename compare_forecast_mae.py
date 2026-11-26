"""Compare InSitu-SURF (with and without DONKI cone CMEs), WSA-BRaVDA-SURF and
WSA-SURF speed forecast MAE against OMNI.

Each forecast CSV holds a HUXt and a hydro run (vsw_huxt, vsw_hydro); each
solver is evaluated as a separate forecast system. Files written by earlier
versions of the generators (single solver, or nearest-time WSA) are ignored.

High-speed streams are also verified as events: hits, misses and false alarms
by lead day, with arrival-time and peak-speed errors for the hits.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import surf.surf_insitu as sinsit

from generate_insitusurf_forecasts import (
    OMNI_LOOKBACK, prepare_omni, required_header as insitu_header, stored_header,
)
from generate_insitu_cone_surf_forecasts import required_header as cone_header
from generate_wsa_bravda_surf_forecasts import required_header as bravda_header
from generate_wsa_surf_forecasts import required_header as wsa_header


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
# Use None for both dates to evaluate every available forecast.
START_DATE = None #"2020-06-01"#                # e.g. "2020-01-01"
END_DATE = None  #"2025-07-01"#                 # e.g. "2020-12-31" (inclusive)
LEAD_DAYS = range(1, 11)          # day 1 is lead time [0, 24) hours

USE_INSITU_SURF = True
USE_INSITU_CONE_SURF = True
USE_BRAVDA_SURF = True
USE_WSA_SURF = True
USE_27_DAY_RECURRENCE = True
REQUIRE_COMMON_FORECAST_DATES = True
RECURRENCE_DAYS = 27.2753
DISTRIBUTION_LEAD_DAY = 4
YEARLY_LEAD_DAY = 4              # lead day for the MAE-by-year figure
MIN_FORECASTS_PER_YEAR = 30       # years with fewer issue dates are dropped

# High-speed stream (HSS) event verification by lead day, following
# HUXt_insitu/code/hse.py (Owens et al., 2005). Observed events come from
# OMNI with no ICME removal.
USE_HSS_ANALYSIS = True
# Repeat the HSS analysis against OMNI with DONKI ICMEs removed (the same
# cleaning the InSitu-SURF and BRaVDA inputs use). MAE always uses raw OMNI.
HSS_ALSO_ICME_REMOVED = True
HSS_SMOOTHING_HOURS = 8           # running mean applied before detection
HSS_RISE_THRESHOLD = 100          # km/s rise needed to flag an event...
HSS_RISE_HOURS = 48               # ...within this many hours
HSS_MIN_SEPARATION_HOURS = 24     # closer events are merged
HSS_MATCH_WINDOW_HOURS = 72       # max-gradient-time matching tolerance
HSS_TIMING = "max_gradient"              # timing error from 'start', 'peak' or 'max_gradient'

SOLVERS = ("hydro",)        # evaluated separately; drop one to skip it (keep the comma)

# name: (directory, file pattern, enabled, colour, required header lines)
SOURCES = {
    "InSitu-SURF": (
        Path("insitusurf_forecasts"),
        "insitusurf_*.csv",
        USE_INSITU_SURF,
        "blue",
        set(insitu_header()),
    ),
    "InSitu-Cone-SURF": (
        Path("insitu_cone_surf"),
        "insitu_cone_surf_*.csv",
        USE_INSITU_CONE_SURF,
        "purple",
        set(cone_header()),
    ),
    "WSA-BRaVDA-SURF": (
        Path("wsa_bravda_surf_forecasts"),
        "wsa_bravda_surf_*.csv",
        USE_BRAVDA_SURF,
        "red",
        set(bravda_header()),
    ),
    "WSA-SURF": (
        Path("wsa_surf_forecasts"),
        "wsa_surf_*.csv",
        USE_WSA_SURF,
        "green",
        set(wsa_header()),
    ),
}
SOLVER_LINESTYLES = {"hydro": "-", "huxt": "--"}
SOLVER_MARKERS = {"hydro": "o", "huxt": "^"}
BRAVDA = "WSA-BRaVDA-SURF"
RECURRENCE = "27.27-day recurrence"
plt.rcParams.update({"font.size": 14, "legend.fontsize": 12})


def issue_time_from_name(path):
    match = re.search(r"_(\d{8}T\d{4})Z\.csv$", path.name)
    return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M") if match else None


def find_forecasts(root, pattern, required):
    forecasts = {}
    outdated = 0
    if not root.exists():
        return forecasts
    start = pd.Timestamp(START_DATE) if START_DATE else None
    # Make an inclusive date-only END_DATE include the entire final day.
    end = pd.Timestamp(END_DATE) + pd.Timedelta(days=1) if END_DATE else None
    for path in root.rglob(pattern):
        issue = issue_time_from_name(path)
        if issue is not None and (start is None or issue >= start) and (end is None or issue < end):
            if required <= stored_header(path):
                forecasts[issue] = path
            else:
                outdated += 1
    if outdated:
        print(f"Ignoring {outdated} outdated files in {root}; rerun the generator")
    return forecasts


def nanoseconds(times):
    """Integer nanoseconds, independent of the datetime64 resolution."""
    return times.astype("datetime64[ns]").astype("int64").to_numpy()


def model_label(source, solver):
    return f"{source} ({solver})"


def series_style(name):
    """Colour by forecast system (black/red/blue), linestyle by solver."""
    if name == "OMNI observed":
        return "black", "-"
    if name == RECURRENCE:
        return "black", ":"
    for source, (_, _, _, colour, _) in SOURCES.items():
        for solver in SOLVERS:
            if name == model_label(source, solver):
                return colour, SOLVER_LINESTYLES[solver]
    raise KeyError(name)


def main():
    if not LEAD_DAYS or min(LEAD_DAYS) < 1:
        raise ValueError("LEAD_DAYS must contain positive integers")
    available = {
        name: find_forecasts(root, pattern, required)
        for name, (root, pattern, enabled, _, required) in SOURCES.items()
        if enabled
    }
    available = {name: files for name, files in available.items() if files}
    if not available:
        raise FileNotFoundError("No enabled forecast files were found in the requested period")

    date_sets = [set(files) for files in available.values()]
    if REQUIRE_COMMON_FORECAST_DATES and len(date_sets) > 1:
        selected_dates = set.intersection(*date_sets)
        if not selected_dates:
            raise ValueError("The enabled forecast systems have no common issue dates")
        available = {
            name: {date: files[date] for date in selected_dates}
            for name, files in available.items()
        }

    all_dates = sorted({date for files in available.values() for date in files})
    max_lead = max(LEAD_DAYS)
    omni_start = all_dates[0] - pd.Timedelta(days=RECURRENCE_DAYS)
    print(
        f"Downloading OMNI once for {omni_start:%Y-%m-%d} through "
        f"{all_dates[-1] + pd.Timedelta(days=max_lead):%Y-%m-%d}"
    )
    omni = sinsit.get_omni(
        omni_start.to_pydatetime(),
        (all_dates[-1] + pd.Timedelta(days=max_lead)).to_pydatetime(),
    )
    omni = omni[["datetime", "V"]].copy()
    omni["datetime"] = pd.to_datetime(omni["datetime"])
    omni = omni.sort_values("datetime")

    errors = []
    speed_distributions = []

    def score(model_name, issue, verification):
        """Add lead-day absolute errors and the distribution-day speeds."""
        lead_hours = (verification["datetime"] - issue).dt.total_seconds() / 3600
        absolute_error = (verification["forecast_v"] - verification["V"]).abs()
        in_distribution_day = (
            (lead_hours >= 24 * (DISTRIBUTION_LEAD_DAY - 1))
            & (lead_hours < 24 * DISTRIBUTION_LEAD_DAY)
        )
        speed_distributions.extend(
            (model_name, value)
            for value in verification.loc[in_distribution_day, "forecast_v"].to_numpy()
        )
        for lead_day in LEAD_DAYS:
            in_lead_day = (lead_hours >= 24 * (lead_day - 1)) & (lead_hours < 24 * lead_day)
            errors.extend(
                (model_name, issue, lead_day, value)
                for value in absolute_error[in_lead_day].to_numpy()
            )

    # Fixed-lead-day forecast series on the OMNI time grid, for the stream
    # analysis: row lead_day - 1 holds that lead day of every forecast, joined
    # end to end.
    hourly = omni.drop_duplicates("datetime").set_index("datetime")["V"]
    grid = pd.date_range(hourly.index[0], hourly.index[-1],
                         freq=hourly.index.to_series().diff().median())
    observed_speed = hourly.reindex(grid).to_numpy()
    grid_ns = nanoseconds(grid)
    step_ns = grid_ns[1] - grid_ns[0]
    day_ns = 86_400 * 10**9
    stitched = {}

    def stitch(model_name, issue, time_ns, speed):
        """Add one forecast to the fixed-lead-day series of its model."""
        series = stitched.setdefault(
            model_name, np.full((max_lead, len(grid)), np.nan)
        )
        issue_ns = issue.as_unit("ns").value
        first, stop = np.searchsorted(
            grid_ns, [issue_ns, issue_ns + max_lead * day_ns]
        )
        sample_ns = grid_ns[first:stop]
        covered = (
            (sample_ns >= time_ns[0] - step_ns) & (sample_ns <= time_ns[-1] + step_ns)
        )
        series[
            (sample_ns[covered] - issue_ns) // day_ns,
            np.arange(first, stop)[covered],
        ] = np.interp(sample_ns[covered], time_ns, speed)

    valid_omni = omni.loc[omni["V"].notna()]
    for source_name, files in available.items():
        print(f"Evaluating {len(files)} {source_name} forecasts")
        for issue, path in sorted(files.items()):
            frame = pd.read_csv(path, comment="#", parse_dates=["time"])
            for solver in SOLVERS:
                forecast = frame[["time", f"vsw_{solver}"]].dropna().sort_values("time")
                if forecast.empty:
                    continue
                stitch(model_label(source_name, solver), issue,
                       nanoseconds(forecast["time"]),
                       forecast[f"vsw_{solver}"].to_numpy())
                verification = valid_omni.loc[
                    (valid_omni["datetime"] >= forecast["time"].min())
                    & (valid_omni["datetime"] <= forecast["time"].max())
                ].copy()
                if verification.empty:
                    continue
                verification["forecast_v"] = np.interp(
                    nanoseconds(verification["datetime"]),
                    nanoseconds(forecast["time"]),
                    forecast[f"vsw_{solver}"],
                )
                score(model_label(source_name, solver), issue, verification)

    # Apply the recurrence baseline to exactly the same issue dates selected
    # above (the intersection when REQUIRE_COMMON_FORECAST_DATES is True).
    if USE_27_DAY_RECURRENCE:
        print(f"Evaluating {len(all_dates)} 27.27-day recurrence forecasts")
        source_time = nanoseconds(valid_omni["datetime"])
        source_speed = valid_omni["V"].to_numpy()
        recurrence = pd.Timedelta(days=RECURRENCE_DAYS)
        recurrence_speed = np.interp(
            grid_ns - recurrence.as_unit("ns").value, source_time, source_speed
        )

        for issue in all_dates:
            stitch(RECURRENCE, issue, grid_ns, recurrence_speed)
            verification = valid_omni.loc[
                (valid_omni["datetime"] >= issue)
                & (valid_omni["datetime"] < issue + pd.Timedelta(days=max_lead))
            ].copy()
            if verification.empty:
                continue
            verification["forecast_v"] = np.interp(
                nanoseconds(verification["datetime"] - recurrence),
                source_time,
                source_speed,
            )
            score(RECURRENCE, issue, verification)

    samples = pd.DataFrame(
        errors, columns=["model", "issue_time", "lead_day", "absolute_error_km_s"]
    )
    if samples.empty:
        raise ValueError("No forecast/OMNI verification pairs were available")

    summary = (
        samples.groupby(["model", "lead_day"])["absolute_error_km_s"]
        .agg(mae_km_s="mean", sample_count="size", standard_error="sem")
        .reset_index()
    )
    forecast_counts = (
        samples.groupby(["model", "lead_day"])["issue_time"]
        .nunique().rename("forecast_count").reset_index()
    )
    summary = summary.merge(forecast_counts, on=["model", "lead_day"])

    figure, (axis, distribution_axis) = plt.subplots(1, 2, figsize=(16, 5))
    for model_name, group in summary.groupby("model"):
        if model_name == RECURRENCE:
            continue
        color, linestyle = series_style(model_name)
        axis.plot(group["lead_day"], group["mae_km_s"], marker="o",
                  linestyle=linestyle, color=color, label=model_name)
    # Shade the no-skill region: MAE worse than the 27.27-day recurrence.
    recurrence_mae = summary.loc[summary["model"] == RECURRENCE]
    if not recurrence_mae.empty:
        top = 1.05 * summary["mae_km_s"].max()
        axis.fill_between(
            recurrence_mae["lead_day"], recurrence_mae["mae_km_s"], top,
            color="gray", alpha=0.3, linewidth=0, label=RECURRENCE, zorder=0,
        )
        axis.set_ylim(top=top)
    axis.set(
        xlabel="Forecast lead day",
        ylabel="Speed MAE [km/s]",
        xticks=list(LEAD_DAYS),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    print(summary.to_string(index=False))

    # Add OMNI once per selected issue date, rather than once per forecast
    # system, so its distribution is not duplicated when comparing two models.
    distribution_lower = pd.Timedelta(days=DISTRIBUTION_LEAD_DAY - 1)
    distribution_upper = pd.Timedelta(days=DISTRIBUTION_LEAD_DAY)
    for issue in all_dates:
        observed = omni.loc[
            (omni["datetime"] >= issue + distribution_lower)
            & (omni["datetime"] < issue + distribution_upper)
            & omni["V"].notna(),
            "V",
        ]
        speed_distributions.extend(
            ("OMNI observed", value) for value in observed.to_numpy()
        )

    distributions = pd.DataFrame(speed_distributions, columns=["series", "speed"])
    distributions = distributions.replace([np.inf, -np.inf], np.nan).dropna()
    if not distributions.empty:
        lower = max(0, distributions["speed"].quantile(0.005) - 25)
        upper = distributions["speed"].quantile(0.995) + 25
        bins = np.linspace(lower, upper, 31)
        for series, values in distributions.groupby("series"):
            if series == "OMNI observed":
                distribution_axis.hist(
                    values["speed"], bins=bins, density=True,
                    histtype="stepfilled", color="gray", alpha=0.4,
                    linewidth=0, zorder=0, label=f"{series} (n={len(values)})",
                )
                continue
            color, linestyle = series_style(series)
            distribution_axis.hist(
                values["speed"], bins=bins, density=True, histtype="step",
                linewidth=2, color=color, linestyle=linestyle,
                label=f"{series} (n={len(values)})",
            )
        distribution_axis.set(
            xlabel="Solar-wind speed [km/s]",
            ylabel=f"Probability density (lead day {DISTRIBUTION_LEAD_DAY})",
        )
        distribution_axis.grid(alpha=0.25)
        distribution_axis.legend()
    else:
        distribution_axis.set_visible(False)
    figure.tight_layout()

    # MAE at a single lead day, split by the year of the forecast issue time.
    yearly = samples.loc[samples["lead_day"] == YEARLY_LEAD_DAY].copy()
    if not yearly.empty:
        yearly["year"] = yearly["issue_time"].dt.year
        # Drop sparsely sampled years (e.g. a partial current year).
        issues_per_year = yearly.groupby("year")["issue_time"].nunique()
        sparse_years = issues_per_year[issues_per_year < MIN_FORECASTS_PER_YEAR]
        if not sparse_years.empty:
            print(f"Excluding years with < {MIN_FORECASTS_PER_YEAR} forecasts: "
                  + ", ".join(f"{year} ({n})" for year, n in sparse_years.items()))
            yearly = yearly.loc[~yearly["year"].isin(sparse_years.index)]
        yearly_mae = (
            yearly.groupby(["model", "year"])["absolute_error_km_s"]
            .agg(mae_km_s="mean", sample_count="size").reset_index()
        )
        year_figure, (year_axis, diff_axis) = plt.subplots(
            2, 1, figsize=(8, 9), sharex=True
        )
        for model_name, group in yearly_mae.groupby("model"):
            if model_name == RECURRENCE:
                continue
            color, linestyle = series_style(model_name)
            year_axis.plot(group["year"], group["mae_km_s"], marker="o",
                           linestyle=linestyle, color=color, label=model_name)
        recurrence_year = yearly_mae.loc[yearly_mae["model"] == RECURRENCE]
        if not recurrence_year.empty:
            top = 1.05 * yearly_mae["mae_km_s"].max()
            year_axis.fill_between(
                recurrence_year["year"], recurrence_year["mae_km_s"], top,
                color="gray", alpha=0.3, linewidth=0, label=RECURRENCE, zorder=0,
            )
            year_axis.set_ylim(top=top)
        year_axis.set(ylabel=f"Speed MAE, lead day {YEARLY_LEAD_DAY} [km/s]")
        year_axis.grid(alpha=0.25)
        year_axis.legend()

        # Recurrence minus WSA-SURF MAE, and InSitu-SURF minus WSA-BRaVDA-SURF
        # MAE (per solver). Positive means the second model had lower MAE.
        mae_table = yearly_mae.pivot(index="year", columns="model", values="mae_km_s")
        if yearly_mae.empty:
            mae_table = pd.DataFrame()
        for solver in SOLVERS:
            wsa = model_label("WSA-SURF", solver)
            if wsa in mae_table and RECURRENCE in mae_table:
                color, linestyle = series_style(wsa)
                difference = (mae_table[RECURRENCE] - mae_table[wsa]).dropna()
                diff_axis.plot(difference.index, difference, marker="o",
                               linestyle=linestyle, color=color,
                               label=f"Recurrence − {wsa}")
            insitu = model_label("InSitu-SURF", solver)
            bravda = model_label(BRAVDA, solver)
            if insitu in mae_table and bravda in mae_table:
                difference = (mae_table[insitu] - mae_table[bravda]).dropna()
                diff_axis.plot(
                    difference.index, difference, marker="o",
                    linestyle=SOLVER_LINESTYLES[solver], color="purple",
                    label=f"InSitu-SURF − WSA-BRaVDA-SURF ({solver})",
                )
        diff_axis.axhline(0, color="black", linewidth=1)
        diff_axis.set(
            xlabel="Forecast issue year",
            ylabel="MAE difference [km/s]",
            xticks=sorted(yearly_mae["year"].unique()),
        )
        diff_axis.grid(alpha=0.25)
        if diff_axis.get_legend_handles_labels()[0]:
            diff_axis.legend()
        year_figure.tight_layout()
        print(yearly_mae.to_string(index=False))
    else:
        print(f"No lead day {YEARLY_LEAD_DAY} samples; add it to LEAD_DAYS")

    if USE_HSS_ANALYSIS:
        print("\nHSS verification against OMNI (no ICME removal)")
        high_speed_stream_analysis(grid, observed_speed, stitched,
                                   "Observed: OMNI (no ICME removal)")
        if HSS_ALSO_ICME_REMOVED:
            # prepare_omni downloads from first_forecast - OMNI_LOOKBACK.
            cleaned = prepare_omni((grid[0] + OMNI_LOOKBACK).to_pydatetime(),
                                   grid[-1].to_pydatetime())
            cleaned_speed = (
                cleaned.drop_duplicates("datetime").set_index("datetime")["V"]
                .reindex(grid).to_numpy(copy=True)
            )
            # Keep raw OMNI data gaps as gaps; cleaning fills them by interpolation.
            cleaned_speed[~np.isfinite(observed_speed)] = np.nan
            print("\nHSS verification against OMNI with DONKI ICMEs removed")
            high_speed_stream_analysis(grid, cleaned_speed, stitched,
                                       "Observed: OMNI with DONKI ICMEs removed")
    plt.show()


def detect_hss_events(times, speed):
    """High-speed-stream events in an hourly speed series (after hse.py).

    The series is smoothed over HSS_SMOOTHING_HOURS. An event is flagged where
    the speed rises by at least HSS_RISE_THRESHOLD within HSS_RISE_HOURS. It
    starts at the minimum in that window and ends when the speed stops rising;
    the peak is the maximum in between. Events separated by less than
    HSS_MIN_SEPARATION_HOURS are merged.

    Returns a DataFrame of events: start, end, peak and max_gradient times,
    and peak_speed.
    """
    dt_hours = (times[1] - times[0]) / pd.Timedelta(hours=1)
    v = (
        pd.Series(speed)
        .rolling(max(1, round(HSS_SMOOTHING_HOURS / dt_hours)), center=True, min_periods=1)
        .mean().to_numpy()
    )
    step = round(HSS_RISE_HOURS / dt_hours)

    def max_gradient(start, end):
        gradients = np.diff(v[start:end + 1])
        if not np.isfinite(gradients).any():
            return start
        return start + int(np.nanargmax(gradients))

    events = []  # [start, end, peak, max_gradient] sample indices
    i = 0
    while i < len(v) - step:
        if v[i + step] - v[i] >= HSS_RISE_THRESHOLD:
            start = i + int(np.nanargmin(v[i:i + step + 1]))
            end = i + step
            while end + 1 < len(v) and v[end + 1] >= v[end]:
                end += 1
            peak = start + int(np.nanargmax(v[start:end + 1]))
            events.append([start, end, peak, max_gradient(start, peak)])
            i = end + 1
        else:
            i += 1

    merged = []
    for event in events:
        if merged and (event[0] - merged[-1][1]) * dt_hours < HSS_MIN_SEPARATION_HOURS:
            start, end = merged[-1][0], event[1]
            peak = start + int(np.nanargmax(v[start:end + 1]))
            merged[-1] = [start, end, peak, max_gradient(start, end)]
        else:
            merged.append(event)

    columns = ["start", "end", "peak", "max_gradient"]
    indices = np.array(merged, dtype=int).reshape(-1, 4)
    result = pd.DataFrame({name: times[indices[:, k]] for k, name in enumerate(columns)})
    result["peak_speed"] = v[indices[:, 2]]
    return result


def match_hss_events(observed, forecast):
    """One-to-one matching of events by max-gradient time (after hse.py).

    Each observed event takes the nearest forecast event within
    HSS_MATCH_WINDOW_HOURS; conflicts go to the closest pair. Returns
    (observed index, forecast index) pairs.
    """
    if observed.empty or forecast.empty:
        return []
    separation = np.abs(
        (observed["max_gradient"].to_numpy()[:, None]
         - forecast["max_gradient"].to_numpy()[None, :]) / pd.Timedelta(hours=1)
    )
    candidates = []
    for i, row in enumerate(separation):
        j = int(np.argmin(row))
        if row[j] <= HSS_MATCH_WINDOW_HOURS:
            candidates.append((row[j], i, j))
    pairs, used = [], set()
    for _, i, j in sorted(candidates):
        if j not in used:
            used.add(j)
            pairs.append((i, j))
    return pairs


def high_speed_stream_analysis(grid, observed_speed, stitched, title=None):
    """Event-based verification of high-speed streams, by forecast lead day.

    ``stitched[model][lead_day - 1]`` is the forecast speed on ``grid`` taken
    from that lead day of every forecast: the series a user of that lead time
    would have seen. Observed and forecast events are found only over the
    samples where both series exist. Matched events are hits, unmatched
    observed events misses and unmatched forecast events false alarms.
    Timing errors use HSS_TIMING ('start', 'peak' or 'max_gradient').
    """
    def ratio(count, total):
        return count / total if total else np.nan

    rows = []
    for model_name, series in stitched.items():
        for lead_day in LEAD_DAYS:
            forecast_speed = series[lead_day - 1]
            common = np.isfinite(observed_speed) & np.isfinite(forecast_speed)
            observed = detect_hss_events(grid, np.where(common, observed_speed, np.nan))
            forecast = detect_hss_events(grid, np.where(common, forecast_speed, np.nan))
            pairs = match_hss_events(observed, forecast)
            i, j = np.array(pairs, dtype=int).reshape(-1, 2).T
            hits = len(pairs)
            misses = len(observed) - hits
            false_alarms = len(forecast) - hits
            # Forecast minus observed timing (hours) and peak speed.
            dt = ((forecast[HSS_TIMING].to_numpy()[j] - observed[HSS_TIMING].to_numpy()[i])
                  / pd.Timedelta(hours=1))
            dv = forecast["peak_speed"].to_numpy()[j] - observed["peak_speed"].to_numpy()[i]
            rows.append({
                "model": model_name,
                "lead_day": lead_day,
                "observed": len(observed),
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "pod": ratio(hits, hits + misses),
                "far": ratio(false_alarms, hits + false_alarms),
                "threat_score": ratio(hits, hits + misses + false_alarms),
                "dt_mean_hours": ratio(dt.sum(), hits),
                "dt_mae_hours": ratio(np.abs(dt).sum(), hits),
                "dv_peak_mean_km_s": ratio(dv.sum(), hits),
                "dv_peak_mae_km_s": ratio(np.abs(dv).sum(), hits),
            })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    panels = (
        ("pod", "Probability of detection"),
        ("far", "False alarm ratio"),
        ("threat_score", "Threat score"),
        ("dt_mae_hours", f"Mean absolute {HSS_TIMING} time error [hours]"),
    )
    for axis, (column, label) in zip(axes.ravel(), panels):
        for model_name, group in summary.groupby("model"):
            color, linestyle = series_style(model_name)
            axis.plot(group["lead_day"], group[column], linestyle=linestyle,
                      color=color, label=model_name,
                      marker=None if model_name == RECURRENCE else "o")
        axis.set(ylabel=label, xticks=list(LEAD_DAYS))
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Forecast lead day")
    # One legend above the panels, so it never covers the curves.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    legend_rows = -(-len(labels) // 5)
    if title:
        figure.suptitle(title, y=0.995 - 0.035 * legend_rows, fontsize=14)
        legend_rows += 1
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.035 * legend_rows))
    return summary


if __name__ == "__main__":
    main()
