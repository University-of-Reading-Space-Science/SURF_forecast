"""Plot InSitu-SURF and WSA-BRaVDA-SURF inner speed boundaries."""

import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sunpy.coordinates import sun

import surf.surf as surf_model
import surf.surf_analysis as sa
import surf.surf_insitu as sinsit

import generate_insitusurf_forecasts as insitu_run
import generate_wsa_bravda_surf_forecasts as bravda_run


FORECAST_DATE = "2023-09-15"  # UTC; change only this line
FORECAST_DAYS = 27


def main():
    issue = pd.Timestamp(FORECAST_DATE).to_pydatetime()

    # Use the same pre-downloaded, DONKI-cleaned OMNI input for both methods.
    omni = insitu_run.prepare_omni(issue, issue)
    omni_chunk = insitu_run.omni_chunk_for_forecast(omni, issue)

    # Construct the 1-D InSitu-SURF model.
    insitu_model = sinsit.omniSURF_forecast(
        issue,
        simtime=FORECAST_DAYS * u.day,
        buffertime=insitu_run.BUFFER_DAYS * u.day,
        rmin=insitu_run.RMIN,
        rmax=insitu_run.RMAX,
        dr=insitu_run.DR,
        nlon=insitu_run.NLON,
        v_max=insitu_run.V_MAX,
        dt_scale=insitu_run.DT_SCALE,
        solver=insitu_run.SOLVER,
        gamma=insitu_run.GAMMA,
        run_2d=False,
        track_cmes=False,
        include_b_boundary=insitu_run.INCLUDE_B_BOUNDARY,
        icme_list=insitu_run.ICME_LIST,
        omni_input=omni_chunk,
    )

    # Construct the WSA/simpleBRaVDA posterior boundary.
    forecast = Time(issue, scale="utc")
    observations = bravda_run.omni_observations(
        forecast,
        Time(pd.to_datetime(omni_chunk["datetime"]).to_numpy()),
        omni_chunk["V"].to_numpy() * u.km / u.s,
    )
    wsa_path = bravda_run.select_wsa(issue)
    speed_1au, longitude, latitude, _ = bravda_run.read_wsa(wsa_path)
    speed_inner = bravda_run.deaccelerate_wsa_speeds(speed_1au, bravda_run.RMIN)
    bravda_boundary = bravda_run.simpleBRaVDA(
        forecast,
        observations,
        speed_inner,
        longitude,
        latitude,
        inner_radius=bravda_run.RMIN,
        settings=bravda_run.BRAVDA_SETTINGS,
    )

    # Match the WSA-BRaVDA forecast generator: start one Carrington rotation
    # earlier for spin-up, solve one longitude only, then display from issue time.
    finish = forecast + FORECAST_DAYS * u.day
    cr_fraction = float(sun.carrington_rotation_number(forecast))
    cr = int(np.floor(cr_fraction))
    cr_lon = 2 * np.pi * (1 - cr_fraction + cr) * u.rad
    run_cr = cr - 1
    run_start = sun.carrington_rotation_time(
        run_cr + (2 * np.pi - cr_lon.to_value(u.rad)) / (2 * np.pi)
    )
    simulation_time = (finish - run_start).to(u.day)
    latitude_times = run_start + np.linspace(0, simulation_time.value, 129) * u.day
    b_boundary = bravda_run.wsa_bpol_boundary(
        wsa_path, forecast, longitude, latitude, bravda_boundary.longitude
    )
    insitu_mean = insitu_model._v_boundary_init_.mean().to_value(u.km / u.s)
    bravda_mean = bravda_boundary.speed.mean().to_value(u.km / u.s)
    inflation_factor = insitu_mean / bravda_mean
    inflation_percent = 100 * (inflation_factor - 1)
    inflated_bravda_speed = bravda_boundary.speed * inflation_factor
    print(
        f"BRaVDA mean={bravda_mean:.1f} km/s; "
        f"InSitu-SURF mean={insitu_mean:.1f} km/s; "
        f"inflation factor={inflation_factor:.5f} ({inflation_percent:+.2f}%)"
    )
    bravda_model = surf_model.SURF(
        v_boundary=bravda_boundary.speed,
        b_boundary=b_boundary,
        cr_num=run_cr,
        cr_lon_init=cr_lon,
        latitude=np.mean(sun.B0(latitude_times)),
        r_min=bravda_run.RMIN.to(u.solRad),
        r_max=bravda_run.RMAX,
        dr=bravda_run.DR,
        v_max=bravda_run.V_MAX,
        simtime=simulation_time,
        lon_out=0 * u.rad,
        dt_scale=bravda_run.DT_SCALE,
        solver=bravda_run.SOLVER,
        gamma=bravda_run.GAMMA,
        track_cmes=False,
        parallel=False,
    )
    inflated_bravda_model = surf_model.SURF(
        v_boundary=inflated_bravda_speed,
        b_boundary=b_boundary,
        cr_num=run_cr,
        cr_lon_init=cr_lon,
        latitude=np.mean(sun.B0(latitude_times)),
        r_min=bravda_run.RMIN.to(u.solRad),
        r_max=bravda_run.RMAX,
        dr=bravda_run.DR,
        v_max=bravda_run.V_MAX,
        simtime=simulation_time,
        lon_out=0 * u.rad,
        dt_scale=bravda_run.DT_SCALE,
        solver=bravda_run.SOLVER,
        gamma=bravda_run.GAMMA,
        track_cmes=False,
        parallel=False,
    )

    print("Solving 1-D InSitu-SURF...")
    insitu_model.solve([])
    print("Solving 1-D WSA-BRaVDA-SURF...")
    bravda_model.solve([])
    print(f"Solving 1-D inflated WSA-BRaVDA-SURF ({inflation_percent:+.2f}%)...")
    inflated_bravda_model.solve([])

    insitu_earth = sa.get_observer_timeseries(
        insitu_model, observer="Earth", suppress_warning=True
    )
    bravda_earth = sa.get_observer_timeseries(
        bravda_model, observer="Earth", suppress_warning=True
    )
    inflated_bravda_earth = sa.get_observer_timeseries(
        inflated_bravda_model, observer="Earth", suppress_warning=True
    )
    plot_start = pd.Timestamp(issue)
    plot_end = pd.Timestamp(finish.datetime)
    insitu_earth = insitu_earth.loc[
        (insitu_earth["time"] >= plot_start) & (insitu_earth["time"] <= plot_end)
    ]
    bravda_earth = bravda_earth.loc[
        (bravda_earth["time"] >= plot_start) & (bravda_earth["time"] <= plot_end)
    ]
    inflated_bravda_earth = inflated_bravda_earth.loc[
        (inflated_bravda_earth["time"] >= plot_start)
        & (inflated_bravda_earth["time"] <= plot_end)
    ]

    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2, width_ratios=(4, 1), hspace=0.32, wspace=0.05)
    boundary_axis = figure.add_subplot(grid[0, 0])
    boundary_hist_axis = figure.add_subplot(grid[0, 1], sharey=boundary_axis)
    timeseries_axis = figure.add_subplot(grid[1, 0])
    timeseries_hist_axis = figure.add_subplot(grid[1, 1], sharey=timeseries_axis)

    insitu_boundary_speed = insitu_model._v_boundary_init_.to_value(u.km / u.s)
    bravda_boundary_speed = bravda_boundary.speed.to_value(u.km / u.s)
    inflated_boundary_speed = inflated_bravda_speed.to_value(u.km / u.s)
    boundary_axis.plot(
        insitu_model.v_boundary_lons.to_value(u.deg),
        insitu_boundary_speed,
        label="InSitu-SURF inner boundary",
        color="C0",
    )
    boundary_axis.plot(
        bravda_boundary.longitude.to_value(u.deg),
        bravda_boundary_speed,
        label="WSA-BRaVDA-SURF inner boundary",
        color="C3",
    )
    boundary_axis.plot(
        bravda_boundary.longitude.to_value(u.deg),
        inflated_boundary_speed,
        label=f"WSA-BRaVDA (×{inflation_factor:.3f}; {inflation_percent:+.1f}%)",
        color="C2",
    )
    boundary_axis.set(
        xlim=(0, 360),
        xlabel="Carrington longitude [deg]",
        ylabel="Inner-boundary speed [km/s]",
        title=f"SURF inner boundaries for {issue:%Y-%m-%d}",
    )
    boundary_axis.grid(alpha=0.25)
    boundary_axis.legend()

    boundary_bins = np.linspace(
        min(insitu_boundary_speed.min(), bravda_boundary_speed.min(), inflated_boundary_speed.min()),
        max(insitu_boundary_speed.max(), bravda_boundary_speed.max(), inflated_boundary_speed.max()),
        25,
    )
    boundary_hist_axis.hist(
        insitu_boundary_speed, bins=boundary_bins, density=True,
        orientation="horizontal", histtype="step", linewidth=2, color="C0",
    )
    boundary_hist_axis.hist(
        bravda_boundary_speed, bins=boundary_bins, density=True,
        orientation="horizontal", histtype="step", linewidth=2, color="C3",
    )
    boundary_hist_axis.hist(
        inflated_boundary_speed, bins=boundary_bins, density=True,
        orientation="horizontal", histtype="step", linewidth=2, color="C2",
    )
    boundary_hist_axis.set(xlabel="Density", title="Distribution")
    boundary_hist_axis.grid(alpha=0.2)
    boundary_hist_axis.tick_params(labelleft=False)

    insitu_earth_speed = insitu_earth["vsw"].to_numpy()
    bravda_earth_speed = bravda_earth["vsw"].to_numpy()
    inflated_bravda_earth_speed = inflated_bravda_earth["vsw"].to_numpy()
    timeseries_axis.plot(
        insitu_earth["time"], insitu_earth["vsw"], color="C0",
        label="InSitu-SURF",
    )
    timeseries_axis.plot(
        bravda_earth["time"], bravda_earth["vsw"], color="C3",
        label="WSA-BRaVDA-SURF",
    )
    timeseries_axis.plot(
        inflated_bravda_earth["time"], inflated_bravda_earth["vsw"], color="C2",
        label=f"WSA-BRaVDA-SURF (×{inflation_factor:.3f})",
    )
    timeseries_axis.set(
        xlim=(plot_start, plot_end),
        xlabel="Time [UTC]",
        ylabel="Earth speed [km/s]",
        title=f"1-D SURF Earth time series: {FORECAST_DAYS} days",
    )
    timeseries_axis.grid(alpha=0.25)
    timeseries_axis.legend()

    timeseries_bins = np.linspace(
        min(insitu_earth_speed.min(), bravda_earth_speed.min(), inflated_bravda_earth_speed.min()),
        max(insitu_earth_speed.max(), bravda_earth_speed.max(), inflated_bravda_earth_speed.max()),
        25,
    )
    timeseries_hist_axis.hist(
        insitu_earth_speed, bins=timeseries_bins, density=True,
        orientation="horizontal", histtype="step", linewidth=2, color="C0",
    )
    timeseries_hist_axis.hist(
        bravda_earth_speed, bins=timeseries_bins, density=True,
        orientation="horizontal", histtype="step", linewidth=2, color="C3",
    )
    timeseries_hist_axis.hist(
        inflated_bravda_earth_speed, bins=timeseries_bins, density=True,
        orientation="horizontal", histtype="step", linewidth=2, color="C2",
    )
    timeseries_hist_axis.set(xlabel="Density", title="Distribution")
    timeseries_hist_axis.grid(alpha=0.2)
    timeseries_hist_axis.tick_params(labelleft=False)
    figure.autofmt_xdate()
    figure.tight_layout()
    print(f"WSA map: {wsa_path}")
    plt.show()


if __name__ == "__main__":
    main()
