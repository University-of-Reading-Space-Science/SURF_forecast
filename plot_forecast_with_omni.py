from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import surf.surf_insitu as sinsit


FORECAST_DATE = "2019-03-1"  # UTC; change only this line
PLOT_INSITU_SURF = True
PLOT_BRAVDA_SURF = True

issue = pd.Timestamp(FORECAST_DATE)
sources = [
    (
        PLOT_INSITU_SURF,
        "InSitu-SURF",
        Path("insitusurf_forecasts") / f"{issue:%Y}" / f"insitusurf_{issue:%Y%m%dT0000Z}.csv",
        "C0",
        "-",
    ),
    (
        PLOT_BRAVDA_SURF,
        "WSA-BRaVDA-SURF",
        Path("wsa_bravda_surf_forecasts") / f"{issue:%Y}"
        / f"wsa_bravda_surf_{issue:%Y%m%dT0000Z}.csv",
        "C3",
        "--",
    ),
]

forecasts = []
for enabled, label, path, color, linestyle in sources:
    if not enabled:
        continue
    if not path.exists():
        print(f"Skipping unavailable {label} forecast: {path}")
        continue
    forecasts.append(
        (label, pd.read_csv(path, comment="#", parse_dates=["time"]), color, linestyle)
    )

if not forecasts:
    raise FileNotFoundError(f"No enabled forecast is available for {issue:%Y-%m-%d}")

forecast_end = max(frame["time"].max() for _, frame, _, _ in forecasts)
omni = sinsit.get_omni(issue.to_pydatetime(), forecast_end.to_pydatetime())

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

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
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
    ax.legend()

b_axis = axes[3]
for label, forecast, color, linestyle in forecasts:
    if "bpol" in forecast:
        b_axis.step(forecast["time"], forecast["bpol"], where="mid",
                    color=color, linestyle=linestyle, label=f"{label} Bpol")
b_axis.plot(omni["datetime"], omni["bpol"], ".", color="k",
            alpha=0.7, label="OMNI Parker-quadrant Bpol")
b_axis.axvline(issue, color="0.5", ls="--", lw=1)
b_axis.set_ylabel("Bpol [−1/0/+1]")
b_axis.set_ylim(-1.2, 1.2)
b_axis.grid(alpha=0.25)
b_axis.legend()

axes[0].set_title(f"Forecast issued {issue:%Y-%m-%d}")
axes[-1].set_xlabel("Time [UTC]")
fig.autofmt_xdate()
fig.tight_layout()
plt.show()
