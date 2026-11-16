"""Compare InSitu-SURF and WSA-BRaVDA-SURF speed forecast MAE against OMNI."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import surf.surf_insitu as sinsit


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
# Use None for both dates to evaluate every available forecast.
START_DATE = None                 # e.g. "2020-01-01"
END_DATE = None                   # e.g. "2020-12-31" (inclusive)
LEAD_DAYS = range(1, 11)          # day 1 is lead time [0, 24) hours

USE_INSITU_SURF = True
USE_BRAVDA_SURF = True
USE_27_DAY_RECURRENCE = True
REQUIRE_COMMON_FORECAST_DATES = True
RECURRENCE_DAYS = 27.2753
DISTRIBUTION_LEAD_DAY = 3

SOURCES = {
    "InSitu-SURF": (
        Path("insitusurf_forecasts"),
        "insitusurf_*.csv",
        USE_INSITU_SURF,
        "C0",
    ),
    "WSA-BRaVDA-SURF": (
        Path("wsa_bravda_surf_forecasts"),
        "wsa_bravda_surf_*.csv",
        USE_BRAVDA_SURF,
        "C3",
    ),
}


def issue_time_from_name(path):
    match = re.search(r"_(\d{8}T\d{4})Z\.csv$", path.name)
    return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M") if match else None


def find_forecasts(root, pattern):
    forecasts = {}
    if not root.exists():
        return forecasts
    start = pd.Timestamp(START_DATE) if START_DATE else None
    # Make an inclusive date-only END_DATE include the entire final day.
    end = pd.Timestamp(END_DATE) + pd.Timedelta(days=1) if END_DATE else None
    for path in root.rglob(pattern):
        issue = issue_time_from_name(path)
        if issue is not None and (start is None or issue >= start) and (end is None or issue < end):
            forecasts[issue] = path
    return forecasts


def main():
    if not LEAD_DAYS or min(LEAD_DAYS) < 1:
        raise ValueError("LEAD_DAYS must contain positive integers")
    available = {
        name: find_forecasts(root, pattern)
        for name, (root, pattern, enabled, _) in SOURCES.items()
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
    for model_name, files in available.items():
        print(f"Evaluating {len(files)} {model_name} forecasts")
        for issue, path in sorted(files.items()):
            forecast = pd.read_csv(path, comment="#", usecols=["time", "vsw"])
            forecast["time"] = pd.to_datetime(forecast["time"])
            forecast = forecast.dropna().sort_values("time")
            if forecast.empty:
                continue

            verification = omni.loc[
                (omni["datetime"] >= forecast["time"].min())
                & (omni["datetime"] <= forecast["time"].max())
                & omni["V"].notna()
            ].copy()
            if verification.empty:
                continue
            verification["forecast_v"] = np.interp(
                verification["datetime"].astype("int64"),
                forecast["time"].astype("int64"),
                forecast["vsw"],
            )
            verification["lead_hours"] = (
                verification["datetime"] - issue
            ).dt.total_seconds() / 3600
            verification["absolute_error"] = abs(
                verification["forecast_v"] - verification["V"]
            )

            distribution_lower = 24 * (DISTRIBUTION_LEAD_DAY - 1)
            distribution_upper = 24 * DISTRIBUTION_LEAD_DAY
            distribution_values = verification.loc[
                (verification["lead_hours"] >= distribution_lower)
                & (verification["lead_hours"] < distribution_upper),
                "forecast_v",
            ]
            speed_distributions.extend(
                (model_name, value) for value in distribution_values.to_numpy()
            )

            for lead_day in LEAD_DAYS:
                lower = 24 * (lead_day - 1)
                upper = 24 * lead_day
                values = verification.loc[
                    (verification["lead_hours"] >= lower)
                    & (verification["lead_hours"] < upper),
                    "absolute_error",
                ]
                errors.extend(
                    (model_name, issue, lead_day, value) for value in values.to_numpy()
                )

    # Apply the recurrence baseline to exactly the same issue dates selected
    # above (the intersection when REQUIRE_COMMON_FORECAST_DATES is True).
    if USE_27_DAY_RECURRENCE:
        print(f"Evaluating {len(all_dates)} 27.27-day recurrence forecasts")
        valid_omni = omni.loc[omni["V"].notna()]
        source_time = valid_omni["datetime"].astype("int64").to_numpy()
        source_speed = valid_omni["V"].to_numpy()
        recurrence = pd.Timedelta(days=RECURRENCE_DAYS)

        for issue in all_dates:
            verification = omni.loc[
                (omni["datetime"] >= issue)
                & (omni["datetime"] < issue + pd.Timedelta(days=max_lead))
                & omni["V"].notna()
            ].copy()
            if verification.empty:
                continue
            verification["forecast_v"] = np.interp(
                (verification["datetime"] - recurrence).astype("int64"),
                source_time,
                source_speed,
            )
            verification["lead_hours"] = (
                verification["datetime"] - issue
            ).dt.total_seconds() / 3600
            verification["absolute_error"] = abs(
                verification["forecast_v"] - verification["V"]
            )

            distribution_lower = 24 * (DISTRIBUTION_LEAD_DAY - 1)
            distribution_upper = 24 * DISTRIBUTION_LEAD_DAY
            distribution_values = verification.loc[
                (verification["lead_hours"] >= distribution_lower)
                & (verification["lead_hours"] < distribution_upper),
                "forecast_v",
            ]
            speed_distributions.extend(
                ("27.27-day recurrence", value)
                for value in distribution_values.to_numpy()
            )

            for lead_day in LEAD_DAYS:
                lower = 24 * (lead_day - 1)
                upper = 24 * lead_day
                values = verification.loc[
                    (verification["lead_hours"] >= lower)
                    & (verification["lead_hours"] < upper),
                    "absolute_error",
                ]
                errors.extend(
                    ("27.27-day recurrence", issue, lead_day, value)
                    for value in values.to_numpy()
                )

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

    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, group in summary.groupby("model"):
        is_recurrence = model_name == "27.27-day recurrence"
        color = "0.25" if is_recurrence else SOURCES[model_name][3]
        linestyle = ":" if is_recurrence else "-"
        axis.plot(group["lead_day"], group["mae_km_s"], marker="o",
                  linestyle=linestyle, color=color, label=model_name)
    axis.set(
        xlabel="Forecast lead day",
        ylabel="Speed MAE [km/s]",
        xticks=list(LEAD_DAYS),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
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
        distribution_figure, distribution_axis = plt.subplots(figsize=(8, 5))
        distribution_styles = {
            "OMNI observed": ("k", "-"),
            "27.27-day recurrence": ("0.35", ":"),
            "InSitu-SURF": (SOURCES["InSitu-SURF"][3], "-"),
            "WSA-BRaVDA-SURF": (SOURCES["WSA-BRaVDA-SURF"][3], "--"),
        }
        for series, values in distributions.groupby("series"):
            color, linestyle = distribution_styles[series]
            distribution_axis.hist(
                values["speed"], bins=bins, density=True, histtype="step",
                linewidth=2, color=color, linestyle=linestyle,
                label=f"{series} (n={len(values)})",
            )
        distribution_axis.set(
            xlabel="Solar-wind speed [km/s]",
            ylabel="Probability density",
            title=f"Speed distributions at forecast lead day {DISTRIBUTION_LEAD_DAY}",
        )
        distribution_axis.grid(alpha=0.25)
        distribution_axis.legend()
        distribution_figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
