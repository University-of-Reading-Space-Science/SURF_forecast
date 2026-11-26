import copy
import datetime
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
import surf.surf as s
import surf.surf_insitu as sinsit
import surf.surf_inputs as sin
import surf.surf_analysis as sa
from sunpy.coordinates import sun
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from scipy.ndimage import gaussian_filter1d

# User controls.
use_cache = True
forecast_time = None  # Set to a datetime for a specific run; None uses the current hour.
ambient_ensemble_size = 100 # Use 100 for production forecasts.
cme_ensemble_size = 500

cache_directory = Path(__file__).resolve().parent / 'cache'
cache_directory.mkdir(exist_ok=True)
cached_run_path = None
ensemble_cache = None
if use_cache:
    cached_runs = list(cache_directory.glob('ensemble_*.pkl'))
    if cached_runs:
        cached_run_path = max(cached_runs, key=lambda path: path.stat().st_mtime)
        with cached_run_path.open('rb') as cache_file:
            ensemble_cache = pickle.load(cache_file)
        try:
            forecast_time = ensemble_cache['forecast_time']
        except KeyError:
            raise ValueError(
                f'Cache {cached_run_path} does not contain a forecast time; regenerate it'
            ) from None
    elif forecast_time is None:
        forecast_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
elif forecast_time is None:
    forecast_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)

start_time = forecast_time - datetime.timedelta(days=10)
latitude = sun.B0(forecast_time).to(u.deg)



# Define settings shared by the boundary preparation and model setup.
solver = 'hydro'
rmin = 21.5 * u.solRad
rmax = 240.0 * u.solRad
latitude = 6.38 * u.deg
simtime = 20.0 * u.day
gamma = 1.5
cme_profile = 'sinusoidal'

ambient_speed_sigma = 25.0  # km/s
ambient_correlation_width = 10.0 * u.deg
cme_speed_sigma_fraction = 0.2
cme_width_sigma_fraction = 0.1
cme_longitude_sigma = 10.0 * u.deg
cme_latitude_sigma = 10.0 * u.deg
disturbance_tolerances = {
    'vsw': 5.0,    # km/s
    'T': 5000.0,   # K
    'n': 0.1,      # protons/cm^3
}


def perturb_cme_list(cmes, rng, perturb=True, v_max=3000.0*u.km/u.s):
    """Return CME copies with the HUXt_forecast parameter perturbations."""
    perturbed = []
    for cme in cmes:
        speed = cme.v.to_value(u.km/u.s)
        width = cme.width.to_value(u.deg)
        longitude = cme.longitude.to_value(u.deg)
        latitude = cme.latitude.to_value(u.deg)
        if perturb:
            speed += rng.normal(0.0, cme_speed_sigma_fraction * speed)
            width += rng.normal(0.0, cme_width_sigma_fraction * width)
            longitude += rng.normal(0.0, cme_longitude_sigma.to_value(u.deg))
            latitude += rng.normal(0.0, cme_latitude_sigma.to_value(u.deg))

        perturbed.append(
            s.ConeCME(
                t_launch=cme.t_launch,
                longitude=longitude*u.deg,
                latitude=np.clip(latitude, -89.9, 89.9)*u.deg,
                v=np.clip(speed, 1.0, v_max.to_value(u.km/u.s) - 1.0)*u.km/u.s,
                width=np.clip(width, 1.0, 179.0)*u.deg,
                thickness=cme.thickness,
                initial_height=cme.initial_height,
                cme_expansion=cme.cme_expansion,
                cme_fixed_duration=cme.cme_fixed_duration,
                fixed_duration=cme.fixed_duration,
                cme_density=cme.cme_density,
                cme_temperature=cme.cme_temperature,
                profile_type=cme.profile_type,
                label=cme.label,
            )
        )
    return perturbed


def disturbance_windows_from_series(cme_series, ambient_series, tolerances):
    """Return every contiguous interval distinguishable from the ambient run."""
    disturbed = np.zeros(len(cme_series), dtype=bool)
    for key, tolerance in tolerances.items():
        difference = np.abs(
            cme_series[key].to_numpy() - ambient_series[key].to_numpy()
        )
        disturbed |= np.isfinite(difference) & (difference > tolerance)

    edges = np.diff(disturbed.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    windows = []
    for start_index, stop_index in zip(starts, stops):
        start = cme_series['time'].iloc[start_index]
        end_index = min(stop_index, len(cme_series) - 1)
        end = cme_series['time'].iloc[end_index]
        windows.append((start, end))
    return windows


def tracked_cme_windows_at_earth(model):
    """Return tracked leading/trailing arrivals and leading-edge speeds at Earth."""
    windows = []
    for cme in model.cmes:
        leading = cme.compute_arrival_at_body('Earth')
        if not leading['hit']:
            windows.append((None, None, np.nan))
            continue

        trailing_edge = copy.deepcopy(cme)
        for coords in trailing_edge.coords.values():
            coords['front_id'] = 1.0 - coords['front_id']
        trailing = trailing_edge.compute_arrival_at_body('Earth')
        end = trailing['t_arrive'].to_datetime() if trailing['hit'] else None
        windows.append(
            (
                leading['t_arrive'].to_datetime(),
                end,
                leading['v'].to_value(u.km/u.s),
            )
        )
    return windows


def circle_on_sphere(longitude, latitude, angular_radius, npoints=100):
    """Return longitude/latitude coordinates for a cone outline."""
    theta = np.linspace(0.0, 2.0*np.pi, npoints)
    latitudes = np.arcsin(
        np.sin(latitude)*np.cos(angular_radius)
        + np.cos(latitude)*np.sin(angular_radius)*np.cos(theta)
    )
    longitudes = longitude + np.arctan2(
        np.sin(theta)*np.sin(angular_radius)*np.cos(latitude),
        np.cos(angular_radius) - np.sin(latitude)*np.sin(latitudes),
    )
    return longitudes, latitudes

# Prepare the selected ambient solar-wind boundary.
if forecast_time < start_time:
    raise ValueError('Forecast time must not be before the model start time')
plot_start_time = forecast_time - datetime.timedelta(days=5)
forecast_buffer = ((forecast_time - start_time).total_seconds()/86400)*u.day
omni_input = sinsit.get_SWPC_realtime(forecast_time - datetime.timedelta(days=28), forecast_time)
omni_numeric = omni_input.select_dtypes(include='number').columns
omni_input[omni_numeric] = omni_input[omni_numeric].interpolate(
    method='linear', limit_direction='both'
)
swpc_observations = omni_input.copy()
omni_input = sinsit.removeICMEs(omni_input, icme_list='DONKI', pre_icme_buffer=0.2, 
                                post_icme_buffer=1.0)
model = sinsit.omniSURF_forecast(forecast_time, simtime=simtime, buffertime=forecast_buffer, 
                                 rmin=rmin, rmax=rmax, dr=1.5*u.solRad, nlon=128, 
                                 v_max=3000.0*(u.km/u.s), dt_scale=4, solver=solver, 
                                 gamma=gamma, run_2d=False, track_cmes=True, 
                                 include_b_boundary=True, icme_list='DONKI', omni_input=omni_input)

model.latitude = latitude.to(u.rad)

# Build the list of cone CMEs injected into the simulation.
cme_list = []
donki_start_time = model.time_init.to_datetime()
donki_end_time = donki_start_time + datetime.timedelta(days=simtime.to_value(u.day))
print(f'Querying DONKI cone CMEs from {donki_start_time} through {donki_end_time}')
try:
    donki_cmes = sin.get_DONKI_cme_list(model, donki_start_time, donki_end_time, feature='LE')
except Exception as exc:
    raise RuntimeError('DONKI CME data could not be accessed') from exc
print(f'Loaded {len(donki_cmes)} DONKI cone CMEs for this run')
for donki_cme in donki_cmes:
    donki_cme.profile_type = cme_profile
    donki_cme.cme_expansion = False
    donki_cme.cme_fixed_duration = True
    donki_cme.fixed_duration = 12.0*u.hour
    donki_cme.thickness = 0.0*u.solRad
    donki_cme.initial_height = 21.5*u.solRad
    donki_cme.cme_density = (600.0/u.cm**3*const.m_p).to(u.kg/u.m**3)
    donki_cme.cme_temperature = 1000000.0*u.K
cme_list.extend(donki_cmes)

# Preserve an identical model as the template for every ensemble member.
ambient_ensemble_template = copy.deepcopy(model)

ordered_cmes = sorted(
    [cme for cme in cme_list if cme.t_launch >= 0*u.s],
    key=lambda cme: cme.t_launch.to_value(u.s),
)
n_cmes = len(ordered_cmes)
cache_tag = (
    f'v4_{forecast_time:%Y%m%dT%H%M}_a{ambient_ensemble_size}_c{cme_ensemble_size}'
    f'_v{ambient_speed_sigma:g}_cw{ambient_correlation_width.to_value(u.deg):g}'
    f'_days{simtime.to_value(u.day):g}_g{gamma:g}'
    f'_cv{cme_speed_sigma_fraction:g}_cw{cme_width_sigma_fraction:g}'
    f'_clon{cme_longitude_sigma.to_value(u.deg):g}'
    f'_clat{cme_latitude_sigma.to_value(u.deg):g}'
    f'_tol{disturbance_tolerances["vsw"]:g}'
    f'-{disturbance_tolerances["T"]:g}-{disturbance_tolerances["n"]:g}'
)
ensemble_cache_path = cache_directory / f'ensemble_{cache_tag}.pkl'

if ensemble_cache is not None:
    if 'ordered_cmes' in ensemble_cache:
        ordered_cmes = ensemble_cache['ordered_cmes']
        n_cmes = len(ordered_cmes)
    ambient_ensemble_series = ensemble_cache['ambient_ensemble_series']
    ambient_ensemble_boundaries = ensemble_cache['ambient_ensemble_boundaries']
    cme_ensemble_speed_series = ensemble_cache['cme_ensemble_speed_series']
    cme_ensemble_arrivals = ensemble_cache['cme_ensemble_arrivals']
    cme_ensemble_ends = ensemble_cache['cme_ensemble_ends']
    cme_ensemble_arrival_speeds = ensemble_cache['cme_ensemble_arrival_speeds']
    cme_ensemble_disturbance_starts = ensemble_cache['cme_ensemble_disturbance_starts']
    cme_ensemble_disturbance_ends = ensemble_cache['cme_ensemble_disturbance_ends']
    cme_ensemble_association_confidence = ensemble_cache[
        'cme_ensemble_association_confidence'
    ]
    # Older caches allocated columns from the unfiltered DONKI list even though
    # SURF removes negative-launch-time CMEs before tracking. Tracked results
    # therefore occupy the first n_cmes columns; trim the unused trailing columns.
    if cme_ensemble_arrivals.shape[1] != n_cmes:
        cme_ensemble_arrivals = cme_ensemble_arrivals[:, :n_cmes]
        cme_ensemble_ends = cme_ensemble_ends[:, :n_cmes]
        cme_ensemble_arrival_speeds = cme_ensemble_arrival_speeds[:, :n_cmes]
        cme_ensemble_disturbance_starts = cme_ensemble_disturbance_starts[:, :n_cmes]
        cme_ensemble_disturbance_ends = cme_ensemble_disturbance_ends[:, :n_cmes]
        cme_ensemble_association_confidence = (
            cme_ensemble_association_confidence[:, :n_cmes]
        )
    ambient_ensemble_size = len(ambient_ensemble_series)
    cme_ensemble_size = cme_ensemble_speed_series.shape[0]
    print(f'Loaded cached ensemble runs from {cached_run_path}')
else:
    rng = np.random.default_rng()
    correlation_width_bins = max(
        1.0,
        ambient_correlation_width.to_value(u.rad)
        / (2.0 * np.pi / ambient_ensemble_template.v_boundary.size),
    )
    ambient_ensemble_series = []
    ambient_ensemble_boundaries = []
    for ensemble_index in range(ambient_ensemble_size):
        ensemble_model = copy.deepcopy(ambient_ensemble_template)
        ensemble_model.track_cmes = False
        if ensemble_index == 0:
            # Match HUXt_forecast: member zero is the unperturbed ambient run.
            noise = np.zeros(ensemble_model.v_boundary.size)
        else:
            noise = gaussian_filter1d(
                rng.normal(size=ensemble_model.v_boundary.size),
                sigma=correlation_width_bins,
                mode='wrap',
            )
            noise -= np.mean(noise)
            noise *= ambient_speed_sigma / np.std(noise)
        perturbed_speed = np.clip(
            ensemble_model.v_boundary.to_value(u.km/u.s) + noise,
            250.0,
            ensemble_model.v_max.to_value(u.km/u.s),
        )
        ensemble_model.v_boundary = perturbed_speed * (u.km/u.s)
        number_density, temperature = s.get_density_temperature_from_velocity(
            perturbed_speed,
            ensemble_model.r[0].to_value(u.solRad),
            gamma=ensemble_model.gamma,
            cache_id=ensemble_model._density_temp_cache_id,
        )
        ensemble_model.rho_boundary = (
            number_density * u.cm**-3 * const.m_p
        ).to(u.kg/u.m**3)
        ensemble_model.temp_boundary = temperature * u.K
        ambient_ensemble_boundaries.append(
            {
                'v': ensemble_model.v_boundary.copy(),
                'rho': ensemble_model.rho_boundary.copy(),
                'T': ensemble_model.temp_boundary.copy(),
            }
        )
        s.solve_chunked(ensemble_model, [], chunk_simtime=3.0*u.day)
        ambient_ensemble_series.append(
            sa.get_observer_timeseries(ensemble_model, observer='Earth')
        )
        print(
            f'Completed perturbed ambient run {ensemble_index + 1} '
            f'of {ambient_ensemble_size}'
        )

    cme_ensemble_speed_series = []
    cme_ensemble_arrivals = np.full((cme_ensemble_size, n_cmes), None, dtype=object)
    cme_ensemble_ends = np.full((cme_ensemble_size, n_cmes), None, dtype=object)
    cme_ensemble_arrival_speeds = np.full((cme_ensemble_size, n_cmes), np.nan)
    cme_ensemble_disturbance_starts = np.full(
        (cme_ensemble_size, n_cmes), None, dtype=object
    )
    cme_ensemble_disturbance_ends = np.full(
        (cme_ensemble_size, n_cmes), None, dtype=object
    )
    cme_ensemble_association_confidence = np.full(
        (cme_ensemble_size, n_cmes), 'unassociated', dtype=object
    )

    for ensemble_index in range(cme_ensemble_size):
        # Match HUXt_forecast: the first CME run uses the unperturbed ambient
        # member; subsequent runs randomly sample the ambient ensemble.
        ambient_index = (
            0 if ensemble_index == 0 else rng.integers(0, ambient_ensemble_size)
        )
        ensemble_model = copy.deepcopy(ambient_ensemble_template)
        boundary = ambient_ensemble_boundaries[ambient_index]
        ensemble_model.v_boundary = boundary['v'].copy()
        ensemble_model.rho_boundary = boundary['rho'].copy()
        ensemble_model.temp_boundary = boundary['T'].copy()
        ensemble_model.track_cmes = True
        member_cmes = perturb_cme_list(
            ordered_cmes,
            rng,
            perturb=ensemble_index > 0,
            v_max=ensemble_model.v_max,
        )
        s.solve_chunked(
            ensemble_model,
            member_cmes,
            chunk_simtime=3.0*u.day,
            streak_carr=np.array([])*u.deg,
        )
        member_series = sa.get_observer_timeseries(ensemble_model, observer='Earth')
        cme_ensemble_speed_series.append(member_series['vsw'].to_numpy())
        disturbance_windows_member = disturbance_windows_from_series(
            member_series,
            ambient_ensemble_series[ambient_index],
            disturbance_tolerances,
        )
        tracked_windows = tracked_cme_windows_at_earth(ensemble_model)
        cme_events = [
            event_time
            for start, end, _ in tracked_windows
            for event_time in (start, end)
            if event_time is not None
        ]
        disturbance_events = [
            event_time
            for start, end in disturbance_windows_member
            for event_time in (start, end)
        ]
        for cme_index, (cme_start, cme_end, arrival_speed) in enumerate(tracked_windows):
            if cme_index >= n_cmes or cme_start is None:
                continue
            cme_ensemble_arrivals[ensemble_index, cme_index] = cme_start
            cme_ensemble_ends[ensemble_index, cme_index] = cme_end
            speed_interval_start = cme_start
            speed_interval_end = cme_end
            window_start = cme_start - datetime.timedelta(days=1)
            candidates = [
                (start, end)
                for start, end in disturbance_windows_member
                if window_start <= start < cme_start
            ]
            candidate_starts = [start for start, _ in candidates]
            other_events = [
                event_time
                for event_time in cme_events + disturbance_events
                if window_start <= event_time <= cme_start
                and event_time != cme_start
                and event_time not in candidate_starts
            ]
            if candidates:
                disturbance_start, disturbance_end = candidates[0]
                speed_interval_start = disturbance_start
                speed_interval_end = disturbance_end
                cme_ensemble_disturbance_starts[ensemble_index, cme_index] = (
                    disturbance_start
                )
                cme_ensemble_disturbance_ends[ensemble_index, cme_index] = disturbance_end
                cme_ensemble_association_confidence[ensemble_index, cme_index] = (
                    'confident'
                    if len(candidates) == 1 and not other_events
                    else 'inconfident'
                )
            if speed_interval_end is not None:
                interval_mask = (
                    (member_series['time'] >= speed_interval_start)
                    & (member_series['time'] <= speed_interval_end)
                )
                interval_speeds = member_series.loc[interval_mask, 'vsw'].to_numpy()
                if np.any(np.isfinite(interval_speeds)):
                    arrival_speed = np.nanmax(interval_speeds)
            cme_ensemble_arrival_speeds[ensemble_index, cme_index] = arrival_speed
        print(f'Completed CME run {ensemble_index + 1} of {cme_ensemble_size}')

    cme_ensemble_speed_series = np.stack(cme_ensemble_speed_series)
    ensemble_cache = {
        'forecast_time': forecast_time,
        'ordered_cmes': ordered_cmes,
        'ambient_ensemble_series': ambient_ensemble_series,
        'ambient_ensemble_boundaries': ambient_ensemble_boundaries,
        'cme_ensemble_speed_series': cme_ensemble_speed_series,
        'cme_ensemble_arrivals': cme_ensemble_arrivals,
        'cme_ensemble_ends': cme_ensemble_ends,
        'cme_ensemble_arrival_speeds': cme_ensemble_arrival_speeds,
        'cme_ensemble_disturbance_starts': cme_ensemble_disturbance_starts,
        'cme_ensemble_disturbance_ends': cme_ensemble_disturbance_ends,
        'cme_ensemble_association_confidence': cme_ensemble_association_confidence,
    }
    temporary_cache_path = ensemble_cache_path.with_suffix('.tmp')
    with temporary_cache_path.open('wb') as cache_file:
        pickle.dump(ensemble_cache, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_cache_path.replace(ensemble_cache_path)
    print(f'Saved ensemble run cache to {ensemble_cache_path}')

ensemble_times = ambient_ensemble_series[0]['time']

# Recompute interval-maximum speeds after loading as well, so legacy caches can
# be reused without retaining their older leading-edge arrival-speed metric.
for ensemble_index in range(cme_ensemble_speed_series.shape[0]):
    for cme_index in range(cme_ensemble_arrivals.shape[1]):
        interval_start = cme_ensemble_disturbance_starts[ensemble_index, cme_index]
        interval_end = cme_ensemble_disturbance_ends[ensemble_index, cme_index]
        if interval_start is None or interval_end is None:
            interval_start = cme_ensemble_arrivals[ensemble_index, cme_index]
            interval_end = cme_ensemble_ends[ensemble_index, cme_index]
        if interval_start is None or interval_end is None:
            continue
        interval_mask = (ensemble_times >= interval_start) & (ensemble_times <= interval_end)
        interval_speeds = cme_ensemble_speed_series[ensemble_index, interval_mask]
        if np.any(np.isfinite(interval_speeds)):
            cme_ensemble_arrival_speeds[ensemble_index, cme_index] = np.nanmax(
                interval_speeds
            )

# Ensemble member zero is the unperturbed run for both the ambient and CME
# ensembles, so the dashboard needs no additional deterministic model solves.
ambient_series = ambient_ensemble_series[0]
ambient_times = ambient_series['time']
unperturbed_cme_speed = cme_ensemble_speed_series[0]

# HUXt_forecast-style CME ensemble dashboard. The coronal-map panel is replaced
# by a location-only view because this forecast is driven by in-situ data.
# Okabe-Ito categorical palette, designed to remain distinguishable for the
# most common forms of colour-vision deficiency.
dashboard_colours = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#000000',  # black
]
label_colour = 'white'
mpl.rc('axes', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('xtick', labelsize=14)
mpl.rc('legend', fontsize=14)
plt.rcParams.update({'font.size': 14})
dashboard_times = ambient_ensemble_series[0]['time']
dashboard_start = pd.Timestamp(plot_start_time)
dashboard_end = dashboard_times.iloc[-1]
cme_hit_fractions = np.mean(cme_ensemble_arrivals != None, axis=0)
# Keep CME-specific dashboard entries only when at least one ensemble member
# reaches Earth after the forecast issue time.
plotted_cme_indices = [
    index
    for index in range(cme_ensemble_arrivals.shape[1])
    if any(
        arrival is not None and arrival > forecast_time
        for arrival in cme_ensemble_arrivals[:, index]
    )
]
cme_plot_colours = {
    cme_index: dashboard_colours[plot_index % len(dashboard_colours)]
    for plot_index, cme_index in enumerate(plotted_cme_indices)
}
dashboard = plt.figure(figsize=(17, 10))
dashboard_grid = dashboard.add_gridspec(3, 3)

speed_axis = dashboard.add_subplot(dashboard_grid[0, :-1])
speed_axis.set_ylim([250, 900])
speed_axis.plot(ambient_times, ambient_series['vsw'], 'k--', label='Unperturbed SW', zorder=1)
speed_axis.plot(
    dashboard_times,
    unperturbed_cme_speed,
    'k-',
    label='Unperturbed CME',
    zorder=2,
)
speed_axis.plot(
    swpc_observations['datetime'],
    swpc_observations['V'],
    'r-',
    label='Obs',
    zorder=2,
)
confidence_intervals = [36, 80, 90]
confidence_normalizer = mpl.colors.Normalize(vmin=0, vmax=len(confidence_intervals))
for colour_index, confidence in enumerate(sorted(confidence_intervals, reverse=True)):
    lower = (100.0 - confidence)/2.0
    upper = 100.0 - lower
    speed_axis.fill_between(
        dashboard_times,
        np.nanpercentile(cme_ensemble_speed_series, lower, axis=0),
        np.nanpercentile(cme_ensemble_speed_series, upper, axis=0),
        color=copy.copy(mpl.cm.cool(confidence_normalizer(colour_index))),
        label=f'Ens. {confidence}% CI',
        zorder=0,
    )
speed_axis.plot(
    dashboard_times,
    np.nanmedian(cme_ensemble_speed_series, axis=0),
    color='tab:olive',
    label='_nolegend_',
    zorder=0,
)
speed_axis.axvline(forecast_time, color='silver')
speed_axis.axvspan(dashboard_start, forecast_time, color='silver', alpha=0.35)
speed_axis.set_xlim(dashboard_start, dashboard_end)
speed_axis.set_ylabel('Solar-wind speed [km/s]')
speed_axis.text(
    0.98, 0.9,
    f'Ambient (N={ambient_ensemble_size}) + CMEs (N={cme_ensemble_size})',
    fontsize=14, transform=speed_axis.transAxes, backgroundcolor=label_colour,
    horizontalalignment='right',
)
speed_legend_handles, speed_legend_labels = speed_axis.get_legend_handles_labels()
speed_legend_by_label = dict(zip(speed_legend_labels, speed_legend_handles))
ensemble_legend_handle = tuple(
    speed_legend_by_label[f'Ens. {confidence}% CI']
    for confidence in (90, 80, 36)
)
speed_axis.legend(
    [
        speed_legend_by_label['Unperturbed SW'],
        speed_legend_by_label['Unperturbed CME'],
        speed_legend_by_label['Obs'],
        ensemble_legend_handle,
    ],
    [
        'Unperturbed SW',
        'Unperturbed CME',
        'Obs',
        '90/80/36% ensemble members',
    ],
    facecolor=label_colour, loc='lower center', framealpha=1, ncol=4,
    handlelength=2.0, handletextpad=0.4, columnspacing=1.0,
    bbox_to_anchor=(0.5, -0.22),
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
)
speed_axis.xaxis.tick_top()
speed_axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
speed_axis.xaxis.set_minor_locator(mdates.DayLocator())
speed_axis.grid(True, axis='y', which='major', alpha=0.3)
speed_axis.grid(True, axis='x', which='both', alpha=0.25)

arrival_grid = dashboard_grid[1, :-1].subgridspec(
    2, 1, height_ratios=(1, 2), hspace=0
)
probability_axis = dashboard.add_subplot(arrival_grid[0])
probability_axis.set_gid('occupancy_axis')
timing_axis = dashboard.add_subplot(arrival_grid[1], sharex=probability_axis)
arrival_bin_width = datetime.timedelta(hours=1)
time_bins = mdates.drange(
    dashboard_start.to_pydatetime(),
    dashboard_end.to_pydatetime() + arrival_bin_width,
    arrival_bin_width,
)

# Fraction of ensemble members inside at least one ICME or its preceding
# disturbed-solar-wind sheath. Overlapping CMEs/sheaths count only once.
n_cme_members = cme_ensemble_arrivals.shape[0]
inside_any_icme = np.zeros((n_cme_members, len(dashboard_times)), dtype=bool)
inside_any_sheath = np.zeros_like(inside_any_icme)
for ensemble_index in range(n_cme_members):
    for cme_index in range(cme_ensemble_arrivals.shape[1]):
        icme_start = cme_ensemble_arrivals[ensemble_index, cme_index]
        icme_end = cme_ensemble_ends[ensemble_index, cme_index]
        if icme_start is not None and icme_end is not None:
            inside_any_icme[ensemble_index] |= (
                (dashboard_times >= icme_start) & (dashboard_times <= icme_end)
            )

        disturbance_start = cme_ensemble_disturbance_starts[
            ensemble_index, cme_index
        ]
        if disturbance_start is not None and icme_start is not None:
            inside_any_sheath[ensemble_index] |= (
                (dashboard_times >= disturbance_start)
                & (dashboard_times < icme_start)
            )

any_icme_probability = np.mean(inside_any_icme, axis=0)
any_sheath_probability = np.mean(inside_any_sheath, axis=0)
arrival_probability_ymax = 0.09
dashboard_time_numbers = mdates.date2num(dashboard_times)
dashboard_time_edges = np.concatenate((
    [dashboard_time_numbers[0] - (dashboard_time_numbers[1] - dashboard_time_numbers[0])/2],
    (dashboard_time_numbers[:-1] + dashboard_time_numbers[1:])/2,
    [dashboard_time_numbers[-1] + (dashboard_time_numbers[-1] - dashboard_time_numbers[-2])/2],
))
for cme_index in range(cme_ensemble_arrivals.shape[1]):
    is_displayed_cme = cme_index in plotted_cme_indices
    colour = cme_plot_colours.get(cme_index, '0.55')
    arrival_dates = [
        value for value in cme_ensemble_arrivals[:, cme_index] if value is not None
    ]
    disturbance_dates = [
        value
        for value in cme_ensemble_disturbance_starts[:, cme_index]
        if value is not None
    ]
    if arrival_dates:
        counts, edges = np.histogram(mdates.date2num(arrival_dates), bins=time_bins)
        centres = (edges[:-1] + edges[1:])/2.0
        arrival_probability = gaussian_filter1d(counts/cme_ensemble_size, 2.0)
        timing_axis.fill_between(
            mdates.num2date(centres),
            0.0,
            arrival_probability,
            color=colour,
            alpha=0.7,
            label=(
                f'CME{cme_index + 1:02d}: '
                f'{cme_hit_fractions[cme_index]*100:2.0f}%'
                if is_displayed_cme else '_nolegend_'
            ),
        )
    if disturbance_dates:
        counts, edges = np.histogram(mdates.date2num(disturbance_dates), bins=time_bins)
        centres = (edges[:-1] + edges[1:])/2.0
        disturbance_probability = gaussian_filter1d(
            counts/cme_ensemble_size, 2.0
        )
        timing_axis.fill_between(
            mdates.num2date(centres),
            arrival_probability_ymax,
            arrival_probability_ymax - disturbance_probability,
            color=colour,
            alpha=0.25,
            label='_nolegend_',
        )
probability_axis.pcolormesh(
    dashboard_time_edges, [1, 2], any_sheath_probability[np.newaxis, :],
    cmap='Greys', vmin=0, vmax=1, shading='flat',
)
cme_probability_colormap = mpl.colors.LinearSegmentedColormap.from_list(
    'cme_probability', ['white', 'red']
)
probability_axis.pcolormesh(
    dashboard_time_edges, [0, 1], any_icme_probability[np.newaxis, :],
    cmap=cme_probability_colormap, vmin=0, vmax=1, shading='flat',
)
probability_axis.axvline(forecast_time, color='silver')
probability_axis.axvspan(
    dashboard_start, forecast_time, color='silver', alpha=0.35
)
timing_axis.axvline(forecast_time, color='silver')
timing_axis.axvspan(
    dashboard_start, forecast_time, color='silver', alpha=0.35
)
probability_axis.set_xlim(dashboard_start, dashboard_end)
probability_axis.set_ylim([0, 2])
probability_axis.set_yticks([1.5, 0.5], labels=['Sheath', 'ICME'])
timing_axis.set_ylabel('Probability')
timing_axis.set_ylim([0, arrival_probability_ymax])
timing_axis.set_yticks([])
timing_axis.text(
    0.98, 0.80,
    'Shock arrival times',
    fontsize=14, transform=timing_axis.transAxes,
    backgroundcolor=label_colour, horizontalalignment='right',
)
timing_axis.text(
    0.98, 0.08,
    'ICME start times',
    fontsize=14, transform=timing_axis.transAxes,
    backgroundcolor=label_colour, horizontalalignment='right',
)
timing_axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
timing_axis.xaxis.set_minor_locator(mdates.DayLocator())
probability_axis.grid(True, axis='x', which='both', alpha=0.25)
timing_axis.grid(True, axis='x', which='both', alpha=0.25)
probability_axis.tick_params(axis='x', which='both', labelbottom=False)
arrival_handles = [
    (
        Patch(facecolor=cme_plot_colours[cme_index], alpha=0.25),
        Patch(facecolor=cme_plot_colours[cme_index], alpha=0.7),
    )
    for cme_index in plotted_cme_indices
]
arrival_labels = [
    f'CME{cme_index + 1:02d}: {cme_hit_fractions[cme_index]*100:2.0f}%'
    for cme_index in plotted_cme_indices
]
arrival_legend = timing_axis.legend(
    arrival_handles, arrival_labels,
    facecolor='white', loc='upper center',
    framealpha=1, handlelength=1.5, handletextpad=0.4,
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
)

table_axis = dashboard.add_subplot(dashboard_grid[2, :])
table_axis.axis('off')
table_headers = [
    'CME', 'Launch at 0.1 AU', 'V\n[km/s]', 'Lon\n[deg]', 'Lat\n[deg]',
    'Width\n[deg]', 'Hit\n[%]', 'Shock arrival\n(median)',
    'Uncertainty\n[+/- hrs]', 'ICME start\n(median)',
    'Uncertainty\n[+/- hrs]',
]
table_rows = []
table_cme_indices = []
for cme_index, cme in enumerate(ordered_cmes):
    if cme_index not in plotted_cme_indices:
        continue
    arrivals = [
        value for value in cme_ensemble_arrivals[:, cme_index] if value is not None
    ]
    disturbances = [
        value
        for value in cme_ensemble_disturbance_starts[:, cme_index]
        if value is not None
    ]
    arrival_nums = mdates.date2num(arrivals) if arrivals else np.array([])
    disturbance_nums = mdates.date2num(disturbances) if disturbances else np.array([])

    def timing_summary(values):
        values = values[np.isfinite(values)]
        if values.size == 0:
            return '--', '--'
        low, median, high = np.nanpercentile(values, [5, 50, 95])
        spread = (
            f'-{(median-low)*24:.1f}/+{(high-median)*24:.1f}'
            if values.size >= 5 else '--'
        )
        return (
            mdates.num2date(median).strftime('%Y-%m-%dT%H:%M'),
            spread,
        )

    arrival_text, arrival_spread = timing_summary(arrival_nums)
    disturbance_text, disturbance_spread = timing_summary(disturbance_nums)
    launch_time = ambient_ensemble_template.time_init.to_datetime() + datetime.timedelta(
        seconds=cme.t_launch.to_value(u.s)
    )
    table_cme_indices.append(cme_index)
    table_rows.append(
        [
            f'CME {cme_index + 1:02d}',
            launch_time.strftime('%Y-%m-%dT%H:%M'),
            f'{cme.v.to_value(u.km/u.s):.0f}',
            f'{cme.longitude.to_value(u.deg):.0f}',
            f'{cme.latitude.to_value(u.deg):.0f}',
            f'{cme.width.to_value(u.deg):.0f}',
            f'{100*cme_hit_fractions[cme_index]:.1f}',
            disturbance_text,
            disturbance_spread,
            arrival_text,
            arrival_spread,
        ]
    )
if not table_rows:
    table_rows = [['--']*len(table_headers)]
summary_table = table_axis.table(
    cellText=table_rows,
    colLabels=table_headers,
    loc='upper center',
    cellLoc='center',
    # Explicit widths, including the CME column, sum to the full axis width.
    colWidths=(np.array([8, 15, 6, 6, 6, 6, 6, 15, 9, 15, 9])/101.0),
    colColours=['silver']*len(table_headers),
)
summary_table.auto_set_font_size(False)
summary_table.set_fontsize(12)
summary_table.scale(1.0, 1.55)
table_cell_height = summary_table.get_celld()[(1, 1)].get_height()
for key, cell in summary_table.get_celld().items():
    cell.set_text_props(ha='center', va='center')
    if key[0] == 0:
        cell.set_height(table_cell_height*1.9)
    if key[1] == 0 and 0 < key[0] <= len(table_cme_indices):
        cme_index = table_cme_indices[key[0] - 1]
        label_rgb = np.asarray(mpl.colors.to_rgb(cme_plot_colours[cme_index]))
        cell.get_text().set_color(tuple(0.65*label_rgb))
        cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(13)

location_axis = dashboard.add_subplot(dashboard_grid[0, -1])
location_axis.axhline(0.0, color='k', linewidth=1)
location_axis.axvline(180.0, color='k', linestyle='--', linewidth=1)
for cme_index, cme in enumerate(ordered_cmes):
    outline_lon, outline_lat = circle_on_sphere(
        cme.longitude.to_value(u.rad),
        cme.latitude.to_value(u.rad),
        cme.width.to_value(u.rad)/2.0,
    )
    # Match HUXt_forecast exactly: rotate HEEQ longitude by 180 degrees so
    # Earth is centred at x=180, then relabel the ticks as -180...180.
    outline_lon = np.rad2deg((outline_lon + np.pi) % (2.0*np.pi))
    outline_lat = np.rad2deg(outline_lat)
    jumps = np.abs(np.diff(outline_lon)) > 180.0
    outline_lon = outline_lon.astype(float)
    outline_lat = outline_lat.astype(float)
    outline_lon[1:][jumps] = np.nan
    outline_lat[1:][jumps] = np.nan
    location_axis.plot(
        outline_lon,
        outline_lat,
        '.',
        color=cme_plot_colours.get(cme_index, 'silver'),
        label=(
            f'CME{cme_index + 1:02d}'
            if cme_index in cme_plot_colours
            else '_nolegend_'
        ),
        zorder=2,
    )
# Draw Earth last so the marker remains visible above every CME outline.
location_axis.plot(
    180.0,
    latitude.to_value(u.deg),
    'ko',
    markersize=7,
    zorder=10,
)
location_axis.set_xlim(0, 360)
location_axis.set_ylim(-90, 90)
location_axis.set_xlabel('HEEQ longitude [deg]')
location_axis.set_ylabel('Latitude [deg]')
location_axis.yaxis.tick_right()
location_axis.yaxis.set_label_position('right')
location_axis.yaxis.label.set_rotation(270)
location_axis.yaxis.labelpad = 18
location_axis.set_title('CME locations')
location_legend_handles = [
    Line2D([], [], color='black', marker='o', linestyle='None', label='Earth')
]
location_axis.legend(
    handles=location_legend_handles,
    loc='upper right',
    fontsize=13,
    handlelength=1.0,
    handleheight=1.0,
    handletextpad=0.4,
)
location_axis.grid(True, alpha=0.3)
location_axis.set_xticks([0, 90, 180, 270, 360])
location_axis.set_xticklabels([-180, -90, 0, 90, 180])
location_axis.set_yticks([-90, -45, 0, 45, 90])

speed_distribution_axis = dashboard.add_subplot(dashboard_grid[1, -1])
speed_bins = np.arange(0.0, 1500.0 + 40.0, 40.0)
n_hit_cmes = np.count_nonzero(
    [
        np.any(np.isfinite(cme_ensemble_arrival_speeds[:, cme_index]))
        for cme_index in plotted_cme_indices
    ]
)
speed_distribution_axis.set_ylim(
    (-(max(n_hit_cmes, 1) - 1)*0.5 - 0.01, 1.0)
)
speed_plot_index = 0
for cme_index in plotted_cme_indices:
    speeds = cme_ensemble_arrival_speeds[:, cme_index]
    speeds = speeds[np.isfinite(speeds)]
    if speeds.size:
        counts, edges = np.histogram(speeds, bins=speed_bins)
        centres = (edges[:-1] + edges[1:])/2.0
        speed_probability = gaussian_filter1d(counts.astype(float), 2.0)
        if np.max(speed_probability) > 0:
            speed_probability /= np.max(speed_probability)
        speed_distribution_axis.plot(
            centres,
            speed_probability - 0.5*speed_plot_index,
            color=cme_plot_colours[cme_index],
            linewidth=3.0,
            label=f'CME {cme_index + 1:02d}',
        )
        speed_plot_index += 1
speed_distribution_axis.set_xlabel('ICME arrival speed [km/s]')
speed_distribution_axis.set_ylabel('Probability')
speed_distribution_axis.set_xlim(0, 1500)
speed_distribution_axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(500))
speed_distribution_axis.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
speed_distribution_axis.set_box_aspect(1)
speed_distribution_axis.yaxis.tick_right()
speed_distribution_axis.yaxis.set_label_position('right')
speed_distribution_axis.yaxis.label.set_rotation(270)
speed_distribution_axis.yaxis.labelpad = 18
speed_distribution_axis.set_yticks([])
speed_distribution_axis.grid(True, axis='x', which='major', alpha=0.35)
speed_distribution_axis.grid(True, axis='x', which='minor', alpha=0.15)

dashboard.text(
    0.025, 0.98, f'Forecast time: {forecast_time:%Y-%m-%dT%H:%M}', fontsize=14
)
dashboard.text(
    0.5, 0.98, 'InSitu-DONKI-SURF/hydro ensemble forecast',
    fontsize=14, horizontalalignment='center'
)
dashboard.text(
    0.85, 0.98, '\u00a9 University of Reading', fontsize=14
)
dashboard.subplots_adjust(
    left=0.05, bottom=0.1, right=0.95, top=0.92, wspace=0.12, hspace=0.34
)

# Draw a strong group separator between the six CME-input columns and the
# inferred ICME timing/association columns. Use rendered cell extents so the
# line remains correctly positioned when fonts or column widths change.
dashboard.canvas.draw()
speed_panel_position = speed_distribution_axis.get_position()
location_panel_position = location_axis.get_position()
speed_distribution_axis.set_box_aspect(None)
speed_distribution_axis.set_position(
    [
        speed_panel_position.x0 + 0.045,
        speed_panel_position.y0,
        location_panel_position.x1 - speed_panel_position.x0 - 0.045,
        speed_panel_position.height,
    ]
)
timing_panel_position = timing_axis.get_position()
timing_axis.set_position(
    [
        timing_panel_position.x0,
        timing_panel_position.y0 - 0.018,
        timing_panel_position.width,
        timing_panel_position.height + 0.018,
    ]
)
speed_panel_position = speed_distribution_axis.get_position()
arrival_legend.set_bbox_to_anchor(
    (
        (timing_axis.get_position().x1 + speed_panel_position.x0)/2,
        speed_panel_position.y1 + 0.012,
    ),
    transform=dashboard.transFigure,
)
table_panel_position = table_axis.get_position()
table_axis.set_position(
    [
        table_panel_position.x0,
        table_panel_position.y0 + 0.005,
        table_panel_position.width,
        table_panel_position.height,
    ]
)
dashboard.canvas.draw()
separator_header = summary_table.get_celld()[(0, 6)].get_window_extent()
separator_bottom = summary_table.get_celld()[(len(table_rows), 6)].get_window_extent()
separator_x, separator_y_top = table_axis.transAxes.inverted().transform(
    (separator_header.x1, separator_header.y1)
)
_, separator_y_bottom = table_axis.transAxes.inverted().transform(
    (separator_bottom.x1, separator_bottom.y0)
)
table_axis.plot(
    [separator_x, separator_x],
    [separator_y_bottom, separator_y_top],
    color='black',
    linewidth=3.0,
    transform=table_axis.transAxes,
    clip_on=False,
    zorder=10,
)

# Replace the occupancy strips with the three occupancy summaries used in the
# final dashboard.
combined_occupancy_dashboard = dashboard
combined_probability_axis = probability_axis
combined_probability_axis.clear()
combined_probability = np.mean(inside_any_icme | inside_any_sheath, axis=0)
combined_probability_axis.plot(
    dashboard_times,
    any_sheath_probability,
    color='black',
    linewidth=2.0,
    label='In sheath',
)
combined_probability_axis.plot(
    dashboard_times,
    any_icme_probability,
    color='tab:red',
    linewidth=2.0,
    label='In ICME',
)
combined_probability_axis.fill_between(
    dashboard_times,
    0,
    combined_probability,
    color='tab:pink',
    alpha=0.35,
    label='Either',
    zorder=0,
)
combined_probability_axis.axvline(forecast_time, color='silver')
combined_probability_axis.axvspan(
    dashboard_start, forecast_time, color='silver', alpha=0.35
)
combined_probability_axis.set_xlim(dashboard_start, dashboard_end)
combined_probability_axis.set_ylim(0, 1)
combined_probability_axis.set_ylabel('')
combined_probability_axis.yaxis.set_major_formatter(
    mpl.ticker.PercentFormatter(xmax=1.0)
)
combined_probability_axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
combined_probability_axis.xaxis.set_minor_locator(mdates.DayLocator())
combined_probability_axis.grid(True, axis='x', which='both', alpha=0.25)
combined_probability_axis.tick_params(
    axis='x', which='both', labelbottom=False
)
combined_probability_axis.legend(
    loc='upper right', ncol=3, facecolor='white', framealpha=1,
    handlelength=1.5, handletextpad=0.4, columnspacing=0.8,
)
combined_occupancy_dashboard.canvas.draw()
plt.show()
