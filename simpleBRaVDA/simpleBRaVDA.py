"""In-memory BRaVDA: a Carrington boundary profile from a map and time series.

Requires numpy, scipy, astropy and sunpy; no HUXt/SURF imports, files,
ephemeris tables, downloads, plots or run directories.

    observations = SolarWindSeries(times, speeds)  # km/s; near Earth
    result = simpleBRaVDA(forecast_time, observations, vr_map, longs, lats)
    speed, longitude = result                     # quantities, km/s and rad
    speed_at_90_degrees = result(90 * u.deg)

Map shape is (latitude, longitude), as returned by readWSA. Bare angles are
radians, speeds km/s and radii AU. Map speeds must describe the requested
inner boundary; WSA's 1-AU-calibrated speeds need preprocessing if appropriate.
Times are UTC strings/datetimes or astropy Time; numeric times must be wrapped
in Time(..., format='mjd'). The forecast time is the observation cutoff.
"""

from dataclasses import dataclass
from functools import wraps
import warnings

import astropy.units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.spatial import cKDTree
from sunpy.coordinates import sun


PERIOD_DAYS = 27.2753
OMEGA = 2 * np.pi / (25.38 * 86400)


def _offline(function):
    """Use bundled analytic ephemerides and avoid automatic IERS downloads."""
    @wraps(function)
    def call(*args, **kwargs):
        with iers.conf.set_temp('auto_download', False), solar_system_ephemeris.set('builtin'):
            return function(*args, **kwargs)
    return call


def _values(value, unit):
    return np.asarray(u.Quantity(value, unit).to_value(unit), dtype=float)


def _time(value):
    if not isinstance(value, Time) and np.asarray(value).dtype.kind in 'iuf':
        raise ValueError("Wrap numeric times in Time(..., format='mjd').")
    return Time(value).utc


def deaccelerate_wsa_speeds(speed_map, inner_radius=0.1 * u.AU, *,
                            calibration_radius=215 * u.solRad,
                            alpha=0.15,
                            acceleration_scale=50 * u.solRad,
                            reference_radius=30 * u.solRad):
    """Remove WSA's residual-acceleration calibration down to an inner radius.

    WSA/GONG speed maps are commonly calibrated to represent speeds at 1 AU.
    This applies the inverse of the residual-acceleration law used by HUXt/SURF
    without applying a ballistic longitude shift. It matches the speed
    residual-acceleration transformation used by heliospheric models while
    preserving longitude.
    """
    speed = u.Quantity(speed_map, u.km / u.s)
    r_cal = u.Quantity(calibration_radius).to(u.km)
    r_inner = u.Quantity(inner_radius).to(u.km)
    r_h = u.Quantity(acceleration_scale).to(u.km)
    r_ref = u.Quantity(reference_radius).to(u.km)
    if (not np.all(np.isfinite(speed.value) & (speed.value > 0))
            or not np.isfinite(alpha) or alpha < 0 or r_h <= 0 * u.km
            or r_cal <= r_inner):
        raise ValueError('Require positive finite speeds/scales and calibration_radius > inner_radius.')
    factor_cal = 1 + alpha * (1 - np.exp(-(r_cal - r_ref) / r_h))
    factor_inner = 1 + alpha * (1 - np.exp(-(r_inner - r_ref) / r_h))
    return (speed / factor_cal * factor_inner).to(u.km / u.s)


@dataclass(frozen=True)
class SolarWindSeries:
    """One independent near-Earth stream; missing speeds may be NaN.

    radius, latitude, longitude: representative observer position over the
        assimilation window. Bare values mean AU and radians. Longitude is
        Heliographic Stonyhurst longitude relative to Earth; positive offsets
        shift the observation operator toward increasing Carrington bins.
        Position values are scalar in this 1-D version. Latitude is retained
        and validated but does not affect the no-latitude propagation model.
        A missing position retains the historical near-Earth defaults.
    uncertainty: constant observation standard deviation (km/s if bare), or
        None for 10% of the mean propagated prior at the observation radius.
    Multiple streams are treated as independent; use OMNI or SWPC, not both,
    and do not duplicate their constituent spacecraft measurements. Non-Earth
    longitudes are not supported yet.
    """

    time: object
    speed: object
    radius: object = None
    latitude: object = None
    longitude: object = None
    uncertainty: object = None
    name: str = 'Earth'


@_offline
def omni_observations(forecast_time, time, speed, *, uncertainty=None,
                      name='OMNI'):
    """Wrap an OMNI speed series with its explicit near-Earth position.

    The model represents each spacecraft by one position per assimilation window,
    so this evaluates SunPy's Earth distance and B0 at the 27.2753-day window
    midpoint. Earth is zero Heliographic Stonyhurst longitude by definition.
    This is numerically identical to the historical implicit-Earth path.
    """
    end = _time(forecast_time)
    if not end.isscalar:
        raise ValueError('forecast_time must be scalar.')
    midpoint = end - (PERIOD_DAYS / 2) * u.day
    return SolarWindSeries(
        time=time, speed=speed,
        radius=sun.earth_distance(midpoint).to(u.AU),
        latitude=sun.B0(midpoint).to(u.rad),
        longitude=0 * u.rad,
        uncertainty=uncertainty, name=name,
    )


def swpc_observations(forecast_time, time, speed, *, uncertainty=None,
                      name='SWPC'):
    """Wrap SWPC real-time L1 speeds using the same Earth position as OMNI.

    Supply times and speeds from ``insitu.get_SWPC_realtime``.
    Use this stream in place of OMNI to avoid counting Earth data twice.
    """
    return omni_observations(forecast_time, time, speed,
                             uncertainty=uncertainty, name=name)


def stereo_a_observations(time, speed, radius, latitude, longitude, *,
                          uncertainty=None, name='STEREO-A'):
    """Wrap STEREO-A speeds with representative spacecraft coordinates.

    ``longitude`` is the Heliographic Stonyhurst (Earth-relative) longitude;
    ``radius`` and ``latitude`` are heliocentric distance and latitude.  The
        caller may obtain these at the assimilation-window midpoint from an
        ephemeris such as ``insitu.stereo_a_position(midpoint)``. Keeping
        ephemeris lookup outside this small wrapper avoids network side effects
        in the solver.
    """
    return SolarWindSeries(time=time, speed=speed, radius=radius,
                           latitude=latitude, longitude=longitude,
                           uncertainty=uncertainty, name=name)


@dataclass(frozen=True)
class Settings:
    """Numerical controls for the ensemble, propagation, and optimization."""

    n_longitude: int = 128
    n_ensemble: int = 500
    seed: int = 20000
    localisation_deg: float = 15.0
    lat_rotation_deg: float = 10.0
    lat_deviation_deg: float = 0.0
    lon_deviation_deg: float = 10.0
    alpha: float = 0.15
    radial_step: object = None
    acceleration_scale: object = None
    outer_radius: object = None
    gtol: float = 1e-5
    maxiter: int = 1000


@dataclass
class BoundaryResult:
    """Boundary speed versus Carrington longitude, with reproducibility data.

    Unpack as ``speed, longitude = result`` or evaluate ``result(longitude)``
    using periodic linear interpolation. ``optimizer.success`` reports whether
    the minimizer converged; a failed solve also emits a RuntimeWarning.
    ``problem`` exposes the exact inputs for regression comparisons.
    """

    speed: u.Quantity
    longitude: u.Quantity
    prior: u.Quantity
    inner_radius: u.Quantity
    forecast_time: Time
    optimizer: OptimizeResult
    initial_cost: float
    final_cost: float
    problem: object

    def __iter__(self):
        yield self.speed
        yield self.longitude

    def __call__(self, longitude):
        return np.interp(_values(longitude, u.rad), self.longitude.value,
                         self.speed.value, period=2 * np.pi) * u.km / u.s


@_offline
def earth_latitudes(forecast_time, longitude):
    """Earth's latitude at each longitude's passage in the preceding rotation.

    Longitude-to-time conversion uses BRaVDA's constant synodic period; SunPy
    computes B0 at those times. No ephemeris lookup file is used.
    """
    end = _time(forecast_time)
    phase = sun.L0(end).to_value(u.rad)
    age = np.mod(_values(longitude, u.rad) - phase, 2 * np.pi)
    return sun.B0(end - age / (2 * np.pi) * PERIOD_DAYS * u.day).to(u.rad)


def generate_ensemble(speed_map, longitude, latitude, reference_latitudes,
                      settings=Settings()):
    """Perturb the sampled Earth track, with member zero unperturbed.

    Keeps the original four-neighbour inverse-square interpolation. Corrected
    mode extends the map periodically across the longitude seam. A private
    A private RandomState makes runs repeatable without changing global RNG state.
    Coordinates must be strictly ascending; maps must be finite and positive.
    """
    lon, lat = _values(longitude, u.rad), _values(latitude, u.rad)
    speed = _values(speed_map, u.km / u.s)
    ref = _values(reference_latitudes, u.rad)
    if (lon.ndim != 1 or lat.ndim != 1 or len(lon) < 2 or len(lat) < 2
            or speed.shape != (len(lat), len(lon))):
        raise ValueError('Map shape must be (n_latitude, n_longitude).')
    if (not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat))
            or np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0)
            or lon[0] < 0 or lon[-1] >= 2 * np.pi
            or lat[0] < -np.pi / 2 or lat[-1] > np.pi / 2):
        raise ValueError('Use ascending longitudes in [0, 2pi), latitudes in [-pi/2, pi/2].')
    if not np.all(np.isfinite(speed) & (speed > 0)):
        raise ValueError('Map speeds must be finite and positive.')
    if ref.shape != lon.shape or not np.all(np.isfinite(ref) & (abs(ref) <= np.pi / 2)):
        raise ValueError('Provide one valid reference latitude per longitude.')
    if settings.n_ensemble < 3:
        raise ValueError('At least three ensemble members are needed.')
    sigmas = np.array([settings.lat_rotation_deg, settings.lat_deviation_deg,
                       settings.lon_deviation_deg], dtype=float)
    if not np.all(np.isfinite(sigmas) & (sigmas >= 0)):
        raise ValueError('Perturbation standard deviations must be nonnegative.')
    phi, theta = np.meshgrid(lon, lat)
    points = np.column_stack([phi.ravel(), theta.ravel(), np.zeros(phi.size)])
    values = speed.ravel()
    points = np.concatenate([points + [-2 * np.pi, 0, 0], points,
                             points + [2 * np.pi, 0, 0]])
    values = np.tile(values, 3)
    tree = cKDTree(points)

    def sample(x, y):
        distance, index = tree.query(np.column_stack([x, y, np.zeros(len(x))]), k=4)
        exact = distance[:, 0] == 0
        weights = 1 / np.maximum(distance, np.finfo(float).eps)**2
        weights[exact] = [1, 0, 0, 0]
        # Interpolate differences to preserve a constant map exactly.
        base = values[index[:, 0]]
        return base + np.sum(weights * (values[index] - base[:, None]), axis=1) / np.sum(weights, axis=1)

    rng = np.random.RandomState(settings.seed)
    size = settings.n_ensemble
    rotation = rng.normal(0, np.deg2rad(sigmas[0]), size)
    node = rng.random_sample(size) * 2 * np.pi
    dy = rng.normal(0, np.deg2rad(sigmas[1]), size)
    dx = rng.normal(0, np.deg2rad(sigmas[2]), size)
    members = [sample(lon, ref)]
    for i in range(1, size):
        y = ref + rotation[i] * np.sin(lon + node[i]) + dy[i]
        y = np.clip(y, -np.pi / 2, np.pi / 2)
        members.append(sample((lon + dx[i]) % (2 * np.pi), y))
    grid = (np.arange(settings.n_longitude) + 0.5) * 2 * np.pi / settings.n_longitude
    return np.array([np.interp(grid, lon, row, period=2 * np.pi)
                     for row in members])


def _bin_observations(series, centres, start, end):
    time = np.atleast_1d(_time(series.time).mjd)
    speed = _values(series.speed, u.km / u.s)
    if time.ndim != 1 or speed.shape != time.shape or not np.all(np.isfinite(time)):
        raise ValueError('Observation times and speeds must be equal-length 1-D arrays.')
    order = np.argsort(time, kind='stable')
    time, speed = time[order], speed[order]
    half_step = (centres[1] - centres[0]) / 2
    edges = np.r_[max(start, centres[0] - half_step),
                  (centres[:-1] + centres[1:]) / 2,
                  min(end, centres[-1] + half_step)]
    # Clamp every bin to the assimilation window.
    edges = np.clip(edges, start, end)
    bins = np.full(len(centres), np.nan)
    indices = np.searchsorted(time, edges)
    for i, (a, b) in enumerate(zip(indices[:-1], indices[1:])):
        good = speed[a:b]
        good = good[np.isfinite(good) & (good >= 0) & (good <= 5000)]
        if len(good):
            bins[i] = good.mean()
    return bins[::-1]


@dataclass
class AssimilationProblem:
    """Prepared arrays, all km/s, km, radians, MJD; no file dependencies."""

    prior: np.ndarray
    covariance: np.ndarray
    ensemble: np.ndarray
    radius: np.ndarray
    dr: float
    r_h: float
    settings: Settings
    observations: list  # (radial index, H, y, R^-1), one per stream
    state_longitude: np.ndarray
    observation_times: np.ndarray
    start_mjd: float
    end_mjd: float
    forecast_time: Time

    def forward(self, boundary):
        n = len(boundary)
        field = np.empty((len(self.radius), n))
        field[0] = boundary
        c = self.dr * OMEGA / (2 * np.pi / n)
        alpha = self.settings.alpha
        for i, r in enumerate(self.radius[:-1]):
            v = field[i]
            transport = c * (np.roll(v, -1) - v) / v
            if i == 0:
                field[i + 1] = v + transport + alpha * boundary * (1 - np.exp(-r / self.r_h))
            else:
                a1 = alpha * boundary * (1 - np.exp(-(r - self.dr) / self.r_h))
                a2 = alpha * boundary * (1 - np.exp(-r / self.r_h))
                field[i + 1] = v + transport - a1 + a2
        return field

    def objective(self):
        """Return cost and adjoint functions with a cached forward evaluation.

        The adjoint differentiates every acceleration contribution and uses
        the conventional half-squared cost.
        """
        inv_b = np.linalg.pinv(self.covariance)
        cached_x, cached_value = None, None
        n = len(self.prior)
        c = self.dr * OMEGA / (2 * np.pi / n)

        def evaluate(x):
            nonlocal cached_x, cached_value
            if cached_x is not None and np.array_equal(x, cached_x):
                return cached_value
            field = self.forward(x)
            diff = x - self.prior
            cost = diff.dot(inv_b).dot(diff)
            forcing = np.zeros_like(field)
            used_radii = set()
            for index, h, y, inv_r in self.observations:
                innovation = y - h.dot(field[index])
                cost += innovation.dot(inv_r).dot(innovation)
                forcing[index] += h.T.dot(inv_r).dot(innovation)
                used_radii.add(index)
            adj = forcing[-1].copy()
            boundary_terms = np.zeros(n)
            for i in range(len(field) - 2, -1, -1):
                r, v = self.radius[i], field[i]
                a = self.settings.alpha * (1 - np.exp(-r / self.r_h))
                if i:
                    a -= self.settings.alpha * (1 - np.exp(-(r - self.dr) / self.r_h))
                boundary_terms += a * adj
                diagonal = 1 - c * (np.roll(v, -1) / (v * v))
                neighbour = c * (1 / np.roll(v, 1))
                adj = diagonal * adj + neighbour * np.roll(adj, 1)
                adj += forcing[i]
            grad = inv_b.dot(diff) - adj - boundary_terms
            cached_x = x.copy()
            cached_value = (float(cost / 2), grad)
            return cached_value

        return lambda x: evaluate(x)[0], lambda x: evaluate(x)[1]


@_offline
def prepare_problem(forecast_time, observations, speed_map, longitude, latitude,
                    inner_radius=0.1 * u.AU, *, settings=Settings(),
                    ensemble=None, reference_latitudes=None):
    """Build the in-memory inverse problem, also useful for controlled comparisons.

    Optional ensemble is already sampled on the uniform Carrington grid, shape
    (members, n_longitude), with the unperturbed profile first. It avoids random
    draws and permits exactly the same prior inputs in both implementations.
    reference_latitudes optionally replaces SunPy's Earth track.
    """
    end_time = _time(forecast_time)
    if not end_time.isscalar:
        raise ValueError('forecast_time must be scalar.')
    n = settings.n_longitude
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError('n_longitude must be an integer >= 4.')
    if (not np.isfinite(settings.localisation_deg) or settings.localisation_deg < 0
            or not np.isfinite(settings.alpha) or settings.alpha < 0
            or not np.isfinite(settings.gtol) or settings.gtol <= 0
            or settings.maxiter < 1):
        raise ValueError('Invalid localisation, acceleration or optimizer settings.')
    rs = u.solRad.to(u.km)
    dr = rs if settings.radial_step is None else float(_values(settings.radial_step, u.AU) * u.AU.to(u.km))
    r_h = 50 * rs if settings.acceleration_scale is None else float(_values(settings.acceleration_scale, u.AU) * u.AU.to(u.km))
    outer = 240 * rs if settings.outer_radius is None else float(_values(settings.outer_radius, u.AU) * u.AU.to(u.km))
    inner = float(_values(inner_radius, u.AU) * u.AU.to(u.km))
    if not all(np.isfinite([inner, outer, dr, r_h])) or not (0 < inner < outer and dr > 0 and r_h > 0):
        raise ValueError('Require 0 < inner_radius < outer_radius and positive radial scales.')
    radius = np.arange(inner, outer, dr)
    if len(radius) < 2:
        raise ValueError('Radial grid needs at least two points.')
    grid = (np.arange(n) + 0.5) * 2 * np.pi / n
    phase = sun.L0(end_time).to_value(u.rad)
    end = float(end_time.mjd)
    start = end - PERIOD_DAYS
    centres = start + (np.arange(n) + 0.5) * PERIOD_DAYS / n
    if ensemble is None:
        if reference_latitudes is None:
            reference_latitudes = earth_latitudes(end_time, longitude)
        ensemble = generate_ensemble(speed_map, longitude, latitude,
                                     reference_latitudes, settings)
    else:
        ensemble = _values(ensemble, u.km / u.s).copy()
    if (ensemble.ndim != 2 or ensemble.shape[1] != n or len(ensemble) < 3
            or not np.all(np.isfinite(ensemble) & (ensemble > 0))):
        raise ValueError('ensemble must have shape (at least 3 members, n_longitude), finite and positive.')
    rotated = np.array([np.interp((grid + phase) % (2 * np.pi), grid, row,
                                  period=2 * np.pi) for row in ensemble])
    denominator = len(ensemble) - 2
    if denominator <= 0:
        raise ValueError('covariance_members must exceed two.')
    perturbations = rotated[1:] - rotated[1:].mean(axis=0)
    covariance = (1 / denominator) * perturbations.T.dot(perturbations)
    if settings.localisation_deg:
        distance = abs(np.arange(n)[:, None] - np.arange(n)[None, :]) * (360 / n)
        distance = np.minimum(distance, 360 - distance)
        covariance *= np.exp(-distance**2 / (2 * settings.localisation_deg**2))
    problem = AssimilationProblem(rotated[0].copy(), covariance, ensemble.copy(),
                                  radius, dr, r_h, settings, [], (grid + phase) % (2 * np.pi),
                                  centres[::-1], start, end, end_time)
    streams = [observations] if isinstance(observations, SolarWindSeries) else list(observations)
    if not streams or not all(isinstance(s, SolarWindSeries) for s in streams):
        raise ValueError('Supply one SolarWindSeries or a nonempty list of them.')
    prior_field = problem.forward(problem.prior)
    for series in streams:
        distance = (sun.earth_distance(Time((start + end) / 2, format='mjd')).to_value(u.AU)
                    if series.radius is None else float(_values(series.radius, u.AU)))
        latitude_value = (sun.B0(Time((start + end) / 2, format='mjd')).to_value(u.rad)
                          if series.latitude is None else float(_values(series.latitude, u.rad)))
        longitude_value = (0.0 if series.longitude is None
                           else float(_values(series.longitude, u.rad)))
        if (not np.isfinite(latitude_value) or abs(latitude_value) > np.pi / 2
                or not np.isfinite(longitude_value)):
            raise ValueError(f'{series.name}: invalid observer latitude or longitude.')
        distance_km = distance * u.AU.to(u.km)
        if not np.isfinite(distance) or not (inner <= distance_km < outer):
            raise ValueError(f'{series.name}: observation radius is outside the model grid.')
        index = int(np.round((distance_km - inner) / dr))
        if not (0 <= index < len(radius)):
            raise ValueError(f'{series.name}: observation radius is outside the model grid.')
        bins = _bin_observations(series, centres, start, end)
        selected = np.flatnonzero(np.isfinite(bins) & (bins > 0) & (bins < 5000))
        if not len(selected):
            continue
        longitude_bin = int(np.round(longitude_value / (2 * np.pi / n)))
        h = np.eye(n)[(selected + longitude_bin) % n]
        sigma = (0.1 * prior_field[index].mean() if series.uncertainty is None
                 else float(_values(series.uncertainty, u.km / u.s)))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Observation uncertainty must be positive and finite.')
        inv_r = np.linalg.pinv(np.diag(np.full(len(selected), sigma**2)))
        problem.observations.append((index, h, bins[selected], inv_r))
    return problem


def simpleBRaVDA(forecast_time, observations, speed_map, longitude, latitude,
                 inner_radius=0.1 * u.AU, *, settings=Settings(), ensemble=None,
                 reference_latitudes=None):
    """Assimilate near-Earth speed series and return a callable boundary profile.

    The result contains 128 longitude samples by default. Missing observations
    are omitted; an entirely empty window returns the prior. Pass Settings to
    change ensemble size, seed, grid or solver tolerance. The function performs
    no file I/O. See BoundaryResult for unpacking and diagnostics.
    """
    problem = prepare_problem(forecast_time, observations, speed_map, longitude,
                              latitude, inner_radius, settings=settings,
                              ensemble=ensemble, reference_latitudes=reference_latitudes)
    cost, gradient = problem.objective()
    initial_cost = cost(problem.prior)
    if not problem.observations:
        optimizer = OptimizeResult(x=problem.prior.copy(), success=True,
                                   message='No valid observations; returned prior.', nit=0)
    else:
        options = {'gtol': settings.gtol, 'maxiter': settings.maxiter}
        if np.linalg.matrix_rank(problem.covariance) < len(problem.prior):
            # A singular Gaussian prior fixes its zero-variance directions.
            # Optimize only its supported subspace, rather than letting pinv(B)
            # accidentally make those directions free to change.
            eigenvalues, vectors = np.linalg.eigh(problem.covariance)
            tolerance = np.max(abs(eigenvalues)) * len(eigenvalues) * np.finfo(float).eps
            if np.any(eigenvalues < -tolerance):
                raise ValueError('Localised covariance is not positive semidefinite.')
            positive = eigenvalues > tolerance
            if not np.any(positive):
                optimizer = OptimizeResult(x=problem.prior.copy(), success=True,
                                           message='Zero prior variance; returned prior.', nit=0)
            else:
                factor = vectors[:, positive] * np.sqrt(eigenvalues[positive])
                optimizer = minimize(lambda z: cost(problem.prior + factor.dot(z)),
                                     np.zeros(np.sum(positive)),
                                     jac=lambda z: factor.T.dot(gradient(problem.prior + factor.dot(z))),
                                     method='BFGS', options=options)
                optimizer.control = optimizer.x.copy()
                optimizer.x = problem.prior + factor.dot(optimizer.control)
        else:
            optimizer = minimize(cost, problem.prior, jac=gradient, method='BFGS', options=options)
        if not optimizer.success:
            warnings.warn(f'BRaVDA minimizer: {optimizer.message}', RuntimeWarning, stacklevel=2)
    if not np.all(np.isfinite(optimizer.x) & (optimizer.x > 0)):
        raise RuntimeError('Optimization produced nonphysical boundary speeds.')
    grid = (np.arange(settings.n_longitude) + 0.5) * 2 * np.pi / settings.n_longitude

    def to_carrington(v):
        return np.interp(grid, problem.state_longitude, v, period=2 * np.pi)

    return BoundaryResult(to_carrington(optimizer.x) * u.km / u.s, grid * u.rad,
                          to_carrington(problem.prior) * u.km / u.s,
                          u.Quantity(inner_radius, u.AU), problem.forecast_time,
                          optimizer, initial_cost, cost(optimizer.x), problem)
