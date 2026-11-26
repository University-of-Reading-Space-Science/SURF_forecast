from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import surf.surf_insitu as sinsit

from generate_insitusurf_forecasts import download_donki_icmes




FORECAST_DATE = "2024-06-5"  # UTC; change only this line
PLOT_INSITU_SURF = True
PLOT_INSITU_CONE_SURF = True
PLOT_BRAVDA_SURF = True
PLOT_WSA_SURF = True
SOLVER_LINESTYLES = {"huxt": "--"}  # add "hydro": "-"" to show the HUXt runs too
OMNI_HISTORY_DAYS = 27  # days of OMNI shown before the forecast issue time
PLOT_DAYS_AFTER = 10  # days shown after the forecast issue time
ICME_LIST = "DONKI"  # "DONKI", "CaneRichardson", or None to hide ICME shading

issue = pd.Timestamp(FORECAST_DATE)
sources = [
    (
        PLOT_INSITU_SURF,
        "InSitu-SURF",
        Path("insitusurf_forecasts") / f"{issue:%Y}" / f"insitusurf_{issue:%Y%m%dT0000Z}.csv",
        "blue",
    ),
    (
        PLOT_INSITU_CONE_SURF,
        "InSitu-Cone-SURF",
        Path("insitu_cone_surf") / f"{issue:%Y}" / f"insitu_cone_surf_{issue:%Y%m%dT0000Z}.csv",
        "purple",
    ),
    (
        PLOT_BRAVDA_SURF,
        "WSA-BRaVDA-SURF",
        Path("wsa_bravda_surf_forecasts") / f"{issue:%Y}"
        / f"wsa_bravda_surf_{issue:%Y%m%dT0000Z}.csv",
        "red",
    ),
    (
        PLOT_WSA_SURF,
        "WSA-SURF",
        Path("wsa_surf_forecasts") / f"{issue:%Y}" / f"wsa_surf_{issue:%Y%m%dT0000Z}.csv",
        "green",
    ),
]

forecasts = []
for enabled, label, path, color in sources:
    if not enabled:
        continue
    if not path.exists():
        print(f"Skipping unavailable {label} forecast: {path}")
        continue
    frame = pd.read_csv(path, comment="#", parse_dates=["time"])
    for solver, linestyle in SOLVER_LINESTYLES.items():
        if f"vsw_{solver}" not in frame:
            print(f"{path.name} has no {solver} run; rerun the generator")
            continue
        columns = {f"{name}_{solver}": name for name in ("vsw", "n", "T", "bpol")}
        forecasts.append((
            f"{label} ({solver})",
            frame[["time", *[c for c in columns if c in frame]]].rename(columns=columns),
            color,
            linestyle,
        ))

if not forecasts:
    raise FileNotFoundError(f"No enabled forecast is available for {issue:%Y-%m-%d}")

omni_start = issue - pd.Timedelta(days=OMNI_HISTORY_DAYS)
plot_end = issue + pd.Timedelta(days=PLOT_DAYS_AFTER)
omni = sinsit.get_omni(omni_start.to_pydatetime(), plot_end.to_pydatetime())

# OMNI COHO supplies RTN BT; -BT is the equivalent GSE By direction here.
# Smooth both components before assigning Parker-spiral quadrants.
omni = omni.sort_values("datetime").reset_index(drop=True)
mag_time = pd.DatetimeIndex(omni["datetime"])
bx_gse = pd.Series(omni["BX_GSE"].to_numpy(), index=mag_time).rolling(
    "6h", center=True, min_periods=1
).mean().reset_index(drop=True)
by_raw = omni["BY_GSE"] if "BY_GSE" in omni else -omni["BT"]
by_gse = pd.Series(by_raw.to_numpy(), index=mag_time).rolling(
    "6h", center=True, min_periods=1
).mean().reset_index(drop=True)
omni["bpol"] = np.nan
finite_b = bx_gse.notna() & by_gse.notna()
omni.loc[finite_b, "bpol"] = 0
omni.loc[finite_b & (bx_gse < 0) & (by_gse > 0), "bpol"] = 1   # away
omni.loc[finite_b & (bx_gse > 0) & (by_gse < 0), "bpol"] = -1  # toward

def periods_in_window(table):
    """(start, end) of each listed ICME that overlaps the plot window."""
    if table is None or table.empty:
        return []
    return [
        (start, end)
        for start, end in zip(pd.to_datetime(table["Shock_time"]),
                              pd.to_datetime(table["ICME_end"]))
        if pd.notna(start) and pd.notna(end) and end >= omni_start and start <= plot_end
    ]


icme_label = f"ICME ({ICME_LIST})"
unlinked_periods = []  # DONKI shocks with no linked CME (not used as ICMEs)
if ICME_LIST == "DONKI":
    icme_label = "ICME (DONKI shock linked to a CME)"
    try:
        all_shocks = download_donki_icmes(omni_start.to_pydatetime(),
                                          plot_end.to_pydatetime(),
                                          require_linked_cme=False)
        icmes = download_donki_icmes(omni_start.to_pydatetime(),
                                     plot_end.to_pydatetime(),
                                     require_linked_cme=True)
        linked_times = set(pd.to_datetime(icmes["Shock_time"]))
        unlinked_periods = [
            period for period in periods_in_window(all_shocks)
            if period[0] not in linked_times
        ]
    except (OSError, ValueError) as error:
        print(f"DONKI download failed ({error}); ICMEs are not shaded")
        icmes = None
elif ICME_LIST == "CaneRichardson":
    icmes = sinsit.ICMElist()
    if pd.to_datetime(icmes["ICME_end"]).max() < plot_end:
        print("The Richardson-Cane list ends before the plot window; later ICMEs are missing")
elif ICME_LIST is None:
    icmes = None
else:
    raise ValueError("ICME_LIST must be 'DONKI', 'CaneRichardson' or None")
icme_periods = periods_in_window(icmes)

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10), layout="constrained")
for ax in axes:
    for i, (start, end) in enumerate(unlinked_periods):
        ax.axvspan(start, end, facecolor="none", edgecolor="0.6", hatch="///", lw=0,
                   label="DONKI shock, no linked CME (assumed 1.5 d)" if i == 0 else None)
    for i, (start, end) in enumerate(icme_periods):
        ax.axvspan(start, end, color="orange", alpha=0.2, lw=0,
                   label=icme_label if i == 0 else None)
panels = [
    ("vsw", "V", "Speed [km/s]"),
    ("n", "N", r"Density [cm$^{-3}$]"),
    ("T", "T", "Temperature [K]"),
]

for ax, (forecast_col, omni_col, ylabel) in zip(axes[:3], panels):
    for label, forecast, color, linestyle in forecasts:
        if forecast_col in forecast:
            ax.plot(forecast["time"], forecast[forecast_col], color=color,
                    linestyle=linestyle, label=label)
    ax.plot(omni["datetime"], omni[omni_col], color="k", alpha=0.7, label="OMNI observed")
    ax.axvline(issue, color="0.5", ls="--", lw=1)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

# Same colours/linestyles as above, so the shared legend covers this panel.
# OMNI Bpol uses Parker-spiral quadrants of the smoothed GSE field.
b_axis = axes[3]
for label, forecast, color, linestyle in forecasts:
    if "bpol" in forecast:
        b_axis.step(forecast["time"], forecast["bpol"], where="mid",
                    color=color, linestyle=linestyle)
b_axis.plot(omni["datetime"], omni["bpol"], ".", color="k", alpha=0.7)
b_axis.axvline(issue, color="0.5", ls="--", lw=1)
b_axis.set_ylabel("Bpol [−1/0/+1]")
b_axis.set_ylim(-1.2, 1.2)
b_axis.grid(alpha=0.25)

# One legend, for all panels, in the bottom panel only.
handles, labels = axes[0].get_legend_handles_labels()
b_axis.legend(handles, labels, ncol=4, fontsize="small", loc="best", framealpha=0.9)

axes[0].set_title(f"Forecast issued {issue:%Y-%m-%d}")
axes[-1].set_xlabel("Time [UTC]")
axes[-1].set_xlim(omni_start, plot_end)
for label in axes[-1].get_xticklabels():
    label.set(rotation=30, horizontalalignment="right")
plt.show()
