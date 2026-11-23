"""Compare InSitu-SURF and WSA-BRaVDA-SURF speed forecast MAE against OMNI.

Also compares the standard WSA-BRaVDA-SURF runs with WSA(-3d)-BRaVDA-SURF,
which uses WSA maps from 3 days before the issue time (map-age sensitivity).

Each forecast CSV holds a HUXt and a hydro run (vsw_huxt, vsw_hydro); each
solver is evaluated as a separate forecast system. Files written by earlier
versions of the generators (single solver, or nearest-time WSA) are ignored.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import surf.surf_insitu as sinsit

from generate_insitusurf_forecasts import required_header as insitu_header, stored_header
from generate_wsa_bravda_surf_forecasts import required_header as bravda_header
from generate_wsa_surf_forecasts import required_header as wsa_header
from generate_wsaminus3_bravda_surf_forecasts import (
    required_header as wsaminus3_bravda_header,
)


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
# Use None for both dates to evaluate every available forecast.
START_DATE = None                 # e.g. "2020-01-01"
END_DATE = None                   # e.g. "2020-12-31" (inclusive)
LEAD_DAYS = range(1, 11)          # day 1 is lead time [0, 24) hours

USE_INSITU_SURF = True
USE_BRAVDA_SURF = True
USE_WSA_SURF = True
USE_WSAMINUS3_BRAVDA_SURF = True
USE_27_DAY_RECURRENCE = True
REQUIRE_COMMON_FORECAST_DATES = True
RECURRENCE_DAYS = 27.2753
DISTRIBUTION_LEAD_DAY = 4
YEARLY_LEAD_DAY = 4              # lead day for the MAE-by-year figure
MIN_FORECASTS_PER_YEAR = 30       # years with fewer issue dates are dropped
MAP_AGE_SCATTER_LEAD_DAY = 4     # lead day for the per-forecast MAE scatter

SOLVERS = ("hydro", "huxt")         # evaluated separately; drop one to skip it

# name: (directory, file pattern, enabled, colour, required header lines)
SOURCES = {
    "InSitu-SURF": (
        Path("insitusurf_forecasts"),
        "insitusurf_*.csv",
        USE_INSITU_SURF,
        "blue",
        set(insitu_header()),
    ),
    "WSA-BRaVDA-SURF": (
        Path("wsa_bravda_surf_forecasts"),
        "wsa_bravda_surf_*.csv",
        USE_BRAVDA_SURF,
        "red",
        set(bravda_header()),
    ),
    "WSA(-3d)-BRaVDA-SURF": (
        Path("wsaminus3_bravda_surf_forecasts"),
        "wsaminus3_bravda_surf_*.csv",
        USE_WSAMINUS3_BRAVDA_SURF,
        "darkorange",
        set(wsaminus3_bravda_header()),
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
LAGGED_BRAVDA = "WSA(-3d)-BRaVDA-SURF"
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

    valid_omni = omni.loc[omni["V"].notna()]
    bravda_frames = {}  # (source, issue) -> forecast speeds, for map-age plots
    for source_name, files in available.items():
        print(f"Evaluating {len(files)} {source_name} forecasts")
        for issue, path in sorted(files.items()):
            frame = pd.read_csv(path, comment="#", parse_dates=["time"])
            if source_name in (BRAVDA, LAGGED_BRAVDA):
                bravda_frames[(source_name, issue)] = frame[
                    ["time", *(f"vsw_{solver}" for solver in SOLVERS)]
                ]
            for solver in SOLVERS:
                forecast = frame[["time", f"vsw_{solver}"]].dropna().sort_values("time")
                if forecast.empty:
                    continue
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

        for issue in all_dates:
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
            lagged = model_label(LAGGED_BRAVDA, solver)
            if lagged in mae_table and bravda in mae_table:
                difference = (mae_table[lagged] - mae_table[bravda]).dropna()
                diff_axis.plot(
                    difference.index, difference, marker="o",
                    linestyle=SOLVER_LINESTYLES[solver], color="darkorange",
                    label=f"WSA(-3d) − WSA-BRaVDA-SURF ({solver})",
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

    plot_map_age_comparison(samples, bravda_frames)
    plt.show()


def plot_map_age_comparison(samples, bravda_frames):
    """Compare WSA-BRaVDA-SURF using the newest WSA map with the 3-day-old map.

    The two systems are paired by issue time, so every panel uses only
    forecasts present for both. Panels: (a) MAE by lead day; (b) paired MAE
    difference (lagged - standard; positive means the newest map is better)
    with +/-2 standard errors; (c) per-forecast MAE at MAP_AGE_SCATTER_LEAD_DAY;
    (d) how far the two forecasts differ from each other, independent of OMNI.
    """
    present = set(samples["model"])
    solvers = [
        solver for solver in SOLVERS
        if {model_label(BRAVDA, solver), model_label(LAGGED_BRAVDA, solver)} <= present
    ]
    if not solvers:
        print(f"Map-age comparison skipped: need both {BRAVDA} and {LAGGED_BRAVDA}")
        return

    # Mean absolute error per forecast and lead day, so the systems pair by issue.
    per_issue = (
        samples.groupby(["model", "issue_time", "lead_day"])["absolute_error_km_s"]
        .mean().rename("mae").reset_index()
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    mae_axis, diff_axis, scatter_axis, spread_axis = axes.ravel()
    paired_issues = set()

    for solver in solvers:
        standard = model_label(BRAVDA, solver)
        lagged = model_label(LAGGED_BRAVDA, solver)
        paired = per_issue.loc[per_issue["model"] == standard].merge(
            per_issue.loc[per_issue["model"] == lagged],
            on=["issue_time", "lead_day"], suffixes=("_standard", "_lagged"),
        )
        if paired.empty:
            continue
        paired_issues |= set(paired["issue_time"])
        linestyle = SOLVER_LINESTYLES[solver]
        n_issues = paired["issue_time"].nunique()

        by_lead = paired.groupby("lead_day")
        for column, name in (("mae_standard", standard), ("mae_lagged", lagged)):
            color, _ = series_style(name)
            means = by_lead[column].mean()
            mae_axis.plot(means.index, means, marker="o", linestyle=linestyle,
                          color=color, label=name)

        # Daily forecasts overlap in time, so these error bars are optimistic.
        paired["delta"] = paired["mae_lagged"] - paired["mae_standard"]
        stats = paired.groupby("lead_day")["delta"].agg(["mean", "sem"])
        diff_axis.errorbar(
            stats.index, stats["mean"], yerr=2 * stats["sem"], marker="o",
            linestyle=linestyle, color="darkorange", capsize=3,
            label=f"{solver} (n={n_issues} forecasts)",
        )

        day = paired.loc[paired["lead_day"] == MAP_AGE_SCATTER_LEAD_DAY]
        if not day.empty:
            standard_better = np.mean(day["mae_standard"] < day["mae_lagged"])
            scatter_axis.scatter(
                day["mae_standard"], day["mae_lagged"], s=10, alpha=0.35,
                marker=SOLVER_MARKERS[solver],
                color="red" if solver == "hydro" else "darkorange",
                label=f"{solver}: newest map better in {standard_better:.0%}",
            )

    # Recurrence MAE over the same issue dates, for reference.
    recurrence = per_issue.loc[
        (per_issue["model"] == RECURRENCE) & per_issue["issue_time"].isin(paired_issues)
    ]
    if not recurrence.empty:
        means = recurrence.groupby("lead_day")["mae"].mean()
        mae_axis.plot(means.index, means, color="black", linestyle=":",
                      label=RECURRENCE)
    mae_axis.set(xlabel="Forecast lead day", ylabel="Speed MAE [km/s]",
                 xticks=list(LEAD_DAYS), title="(a) MAE on paired issue dates")
    mae_axis.grid(alpha=0.25)
    mae_axis.legend(fontsize=10)

    diff_axis.axhline(0, color="black", linewidth=1)
    diff_axis.set(xlabel="Forecast lead day",
                  ylabel="MAE(WSA −3 d) − MAE(newest WSA) [km/s]",
                  xticks=list(LEAD_DAYS),
                  title="(b) Paired MAE difference (±2 SE)")
    diff_axis.grid(alpha=0.25)
    diff_axis.legend(fontsize=10)

    if scatter_axis.collections:
        limits = [*scatter_axis.get_xlim(), *scatter_axis.get_ylim()]
        lower, upper = max(0, min(limits)), max(limits)
        scatter_axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        scatter_axis.set(xlim=(lower, upper), ylim=(lower, upper), aspect="equal")
        scatter_axis.legend(fontsize=10, markerscale=2)
    scatter_axis.set(xlabel=f"{BRAVDA} MAE [km/s]",
                     ylabel=f"{LAGGED_BRAVDA} MAE [km/s]",
                     title=f"(c) Per-forecast MAE, lead day {MAP_AGE_SCATTER_LEAD_DAY}")
    scatter_axis.grid(alpha=0.25)

    # Divergence of the two forecasts from each other on the model time grid.
    common = sorted(
        {issue for source, issue in bravda_frames if source == BRAVDA}
        & {issue for source, issue in bravda_frames if source == LAGGED_BRAVDA}
    )
    rows = []
    for issue in common:
        merged = bravda_frames[(BRAVDA, issue)].merge(
            bravda_frames[(LAGGED_BRAVDA, issue)], on="time",
            suffixes=("_standard", "_lagged"),
        )
        lead_day = ((merged["time"] - issue).dt.total_seconds() // 86400).astype(int) + 1
        for solver in solvers:
            rows.append(pd.DataFrame({
                "solver": solver,
                "lead_day": lead_day,
                "difference": (merged[f"vsw_{solver}_lagged"]
                               - merged[f"vsw_{solver}_standard"]),
            }))
    divergence = pd.concat(rows, ignore_index=True).dropna() if rows else pd.DataFrame()
    if not divergence.empty:
        divergence = divergence.loc[divergence["lead_day"].isin(list(LEAD_DAYS))]
        for solver, group in divergence.groupby("solver"):
            linestyle = SOLVER_LINESTYLES[solver]
            by_lead = group.groupby("lead_day")["difference"]
            absolute = by_lead.apply(lambda values: values.abs().mean())
            spread_axis.plot(absolute.index, absolute, marker="o", linestyle=linestyle,
                             color="darkorange", label=f"mean |Δv| ({solver})")
            bias = by_lead.mean()
            spread_axis.plot(bias.index, bias, marker="s", linestyle=linestyle,
                             color="gray", label=f"mean Δv ({solver})")
        spread_axis.axhline(0, color="black", linewidth=1)
        spread_axis.set(xlabel="Forecast lead day",
                        ylabel="Δv = v(WSA −3 d) − v(newest WSA) [km/s]",
                        xticks=list(LEAD_DAYS),
                        title=f"(d) Forecast divergence ({len(common)} forecasts)")
        spread_axis.grid(alpha=0.25)
        spread_axis.legend(fontsize=10)
    else:
        spread_axis.set_visible(False)

    figure.suptitle("WSA-BRaVDA-SURF: newest WSA map vs map from 3 days earlier")
    figure.tight_layout()


if __name__ == "__main__":
    main()
