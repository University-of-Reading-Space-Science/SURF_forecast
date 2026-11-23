"""Download and read WSA velocity maps for simpleBRaVDA."""

import bisect
import datetime
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import numpy as np


ISWA_WSA_URL = "https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/solar"


def _directory_links(url, timeout):
    """Return link targets from an ISWA directory listing."""
    try:
        with urlopen(url, timeout=timeout) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            listing = response.read().decode(encoding, errors="replace")
    except HTTPError as error:
        if error.code == 404:
            return []
        raise
    return re.findall(r'href=["\']([^"\']+)["\']', listing)


# CCMC/ISWA spells the map type "gongz" in WSA5.4/WSA6 but "gong_z" in the
# WSA5.X archive (2006-2022), so accept both.
_MAP_NAME = re.compile(r"vel_(\d{12})R\d+_gong_?z\.fits", re.IGNORECASE)


def _root_url(version, base_url):
    return f"{base_url.rstrip('/')}/{version}/R21.5/WSA_VEL/GONG_Z/"


def _archive_years(version, timeout, base_url):
    """Return the years present in one version's GONG_Z archive."""
    years = []
    for link in _directory_links(_root_url(version, base_url), timeout):
        match = re.fullmatch(r"(\d{4})/", link)
        if match:
            years.append(int(match.group(1)))
    return sorted(years)


def _month_maps(version, year, month, timeout, base_url):
    """Return sorted ``(map time, url)`` pairs listed for one archive month."""
    month_url = f"{_root_url(version, base_url)}{year:04d}/{month:02d}/"
    maps = {}
    for link in _directory_links(month_url, timeout):
        filename = unquote(Path(urlparse(link).path).name)
        match = _MAP_NAME.fullmatch(filename)
        if match:
            file_time = datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M")
            maps[file_time] = urljoin(month_url, filename)
    return sorted(maps.items())


def _previous_map(timestamp, version, timeout, base_url):
    """Return the newest GONG_Z WSA map at or before ``timestamp``."""
    root_url = _root_url(version, base_url)
    years = [year for year in _archive_years(version, timeout, base_url)
             if year <= timestamp.year]

    for year in sorted(years, reverse=True):
        year_url = f"{root_url}{year:04d}/"
        months = []
        for link in _directory_links(year_url, timeout):
            match = re.fullmatch(r"(\d{2})/", link)
            if match:
                month = int(match.group(1))
                if year < timestamp.year or month <= timestamp.month:
                    months.append(month)

        for month in sorted(months, reverse=True):
            candidates = [item for item in _month_maps(version, year, month, timeout, base_url)
                          if item[0] <= timestamp]
            if candidates:
                file_time, url = candidates[-1]
                return url, file_time
    return None


def iswa_url(path, base_url=ISWA_WSA_URL):
    """Return the CCMC/ISWA URL of a map stored as ``<datadir>/<version>/<file>``.

    The archive is organised by version, year and month, so the source URL is
    fully determined by the version folder and the time in the file name.
    """
    path = Path(path)
    match = _MAP_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a GONG_Z WSA map file name: {path.name}")
    stamp = match.group(1)
    return f"{_root_url(path.parent.name, base_url)}{stamp[:4]}/{stamp[4:6]}/{path.name}"


def _download(url, output_path, timeout):
    """Download ``url`` atomically to ``output_path`` unless it already exists."""
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:
        content = response.read()
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.write_bytes(content)
    temporary_path.replace(output_path)
    return output_path


def _datetime_utc(value):
    """Convert a supported timestamp to a naive UTC datetime."""
    if isinstance(value, datetime.datetime):
        result = value
    elif isinstance(value, datetime.date):
        result = datetime.datetime.combine(value, datetime.time.max)
    elif isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        result = datetime.datetime.combine(
            datetime.date.fromisoformat(value), datetime.time.max
        )
    else:
        result = Time(value).to_datetime(timezone=datetime.timezone.utc)
    if result.tzinfo is not None:
        result = result.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return result


def get_wsa_from_iswa(timestamp, datadir=None, timeout=30,
                      versions=("WSA6", "WSA5.4", "WSA5.X"),
                      max_age=datetime.timedelta(days=2), base_url=ISWA_WSA_URL):
    """Download the newest available GONG_Z WSA map before ``timestamp``.

    Versions are searched in preference order. Within the first version with a
    sufficiently recent result, the latest map no later than ``timestamp`` is
    downloaded. The returned :class:`~pathlib.Path` can be passed to
    :func:`read_wsa`.
    """
    timestamp = _datetime_utc(timestamp)
    if not versions:
        raise ValueError("versions must contain at least one WSA archive version")
    if max_age < datetime.timedelta(0):
        raise ValueError("max_age must not be negative")

    selected = None
    selected_version = None
    for version in versions:
        candidate = _previous_map(timestamp, version, timeout, base_url)
        if candidate is not None and timestamp - candidate[1] <= max_age:
            selected = candidate
            selected_version = version
            break

    if selected is None:
        days = max_age.total_seconds() / datetime.timedelta(days=1).total_seconds()
        raise FileNotFoundError(
            f"No GONG_Z WSA map is available within {days:g} days before "
            f"{timestamp:%Y-%m-%d %H:%M}."
        )

    url, _ = selected
    output_root = Path("wsa_data") if datadir is None else Path(datadir)
    output_path = output_root / selected_version / Path(urlparse(url).path).name
    return _download(url, output_path, timeout)


def download_wsa_daily_archive(start, end, datadir, *, issue_time=datetime.time(0),
                               timeout=30, versions=("WSA6", "WSA5.4", "WSA5.X"),
                               max_age=datetime.timedelta(days=2),
                               base_url=ISWA_WSA_URL):
    """Download, for every day in ``start``..``end``, the newest map at or before
    that day's ``issue_time`` (default 00:00 UTC, i.e. the last pre-midnight map).

    This applies exactly the selection rule of :func:`get_wsa_from_iswa` to a
    daily sequence of issue times, but lists each archive month only once.
    Files go to ``datadir/<version>/`` and existing files are not re-downloaded.
    Returns a list of ``(issue time, path or None, version or None)``; ``None``
    means no map within ``max_age`` (or a failed download, which is reported).
    """
    start = datetime.datetime.combine(_datetime_utc(start).date(), issue_time)
    end = datetime.datetime.combine(_datetime_utc(end).date(), issue_time)
    if end < start:
        raise ValueError("end must not be earlier than start")
    issues = [start + datetime.timedelta(days=n) for n in range((end - start).days + 1)]

    # Every month that can hold a map for the first issue time onward.
    first = start - max_age
    months = []
    year, month = first.year, first.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    catalogue = {}
    for version in versions:
        years = set(_archive_years(version, timeout, base_url))
        listed = [item for (y, m) in months if y in years
                  for item in _month_maps(version, y, m, timeout, base_url)]
        catalogue[version] = sorted(listed)
        print(f"{version}: {len(listed)} maps listed for {len(months)} months")

    results = []
    datadir = Path(datadir)
    for issue in issues:
        chosen = None
        for version in versions:
            maps = catalogue[version]
            index = bisect.bisect_right(maps, (issue, "\uffff")) - 1
            # Newest first; a map is only usable if its OBSJUL is also at or
            # before the issue time, which needs the file itself.
            while index >= 0 and issue - maps[index][0] <= max_age:
                url = maps[index][1]
                path = datadir / version / Path(urlparse(url).path).name
                try:
                    _download(url, path, timeout)
                except (HTTPError, URLError, OSError) as error:
                    print(f"{issue:%Y-%m-%d}: download failed for {url}: {error}")
                    index -= 1
                    continue
                if available_by(path, issue):
                    chosen = path, version
                    break
                print(f"{issue:%Y-%m-%d}: {path.name} OBSJUL is after the issue time; trying older map")
                index -= 1
            if chosen:
                break
        if chosen is None:
            print(f"{issue:%Y-%m-%d}: no usable map within {max_age} before issue time")
            results.append((issue, None, None))
        else:
            results.append((issue, chosen[0], chosen[1]))
    return results


def map_time_from_name(path):
    """Map time encoded in a GONG_Z WSA file name (naive UTC datetime)."""
    match = _MAP_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Not a GONG_Z WSA map file name: {Path(path).name}")
    return datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M")


def map_obs_time(path):
    """Observation time (header OBSJUL) of a WSA map as a naive UTC datetime.

    In the WSA5.X archive this can differ from the file-name time by an hour
    or more in either direction, so both are checked against issue times.
    """
    header = fits.getheader(path, 0)
    return Time(header["OBSJUL"], format="jd", scale="utc").to_datetime()


def available_by(path, issue_time):
    """True if both the file-name time and OBSJUL are at or before ``issue_time``."""
    return max(map_time_from_name(path), map_obs_time(path)) <= issue_time


def read_wsa(path):
    """Return speed, Carrington longitude, latitude, and map time from WSA FITS."""
    path = Path(path)
    with fits.open(path) as hdul:
        header = hdul[0].header
        step = header.get("GRID", header.get("LONSTEP"))
        if step is None:
            raise ValueError("WSA FITS header has neither GRID nor LONSTEP")
        step = np.deg2rad(step)
        offset = np.deg2rad(header["CARRLONG"]) % (2 * np.pi)
        raw_speed = np.asarray(hdul[0].data[1], dtype=float)
        map_time = Time(header["OBSJUL"], format="jd")

    longitude = (np.arange(raw_speed.shape[1]) + 0.5) * step
    latitude = -np.pi / 2 + (np.arange(raw_speed.shape[0]) + 0.5) * step
    source_longitude = (longitude + offset) % (2 * np.pi)
    order = np.argsort(source_longitude)
    speed = np.vstack([
        np.interp(longitude, source_longitude[order], row[order], period=2 * np.pi)
        for row in raw_speed
    ])
    return speed * u.km / u.s, longitude * u.rad, latitude * u.rad, map_time


# Compatibility with existing notebooks and the demo's historical spelling.
readWSA = read_wsa
get_WSA_from_ISWA = get_wsa_from_iswa
