"""Download and read WSA velocity maps for simpleBRaVDA."""

import datetime
from pathlib import Path
import re
from urllib.error import HTTPError
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


def _previous_map(timestamp, version, timeout, base_url):
    """Return the newest GONG_Z WSA map at or before ``timestamp``."""
    root_url = f"{base_url.rstrip('/')}/{version}/R21.5/WSA_VEL/GONG_Z/"
    years = []
    for link in _directory_links(root_url, timeout):
        match = re.fullmatch(r"(\d{4})/", link)
        if match and int(match.group(1)) <= timestamp.year:
            years.append(int(match.group(1)))

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
            month_url = f"{year_url}{month:02d}/"
            candidates = []
            for link in _directory_links(month_url, timeout):
                filename = unquote(Path(urlparse(link).path).name)
                match = re.fullmatch(
                    r"vel_(\d{12})R\d+_gongz\.fits", filename, re.IGNORECASE
                )
                if match:
                    file_time = datetime.datetime.strptime(
                        match.group(1), "%Y%m%d%H%M"
                    )
                    if file_time <= timestamp:
                        candidates.append((file_time, filename))
            if candidates:
                file_time, filename = max(candidates)
                return urljoin(month_url, filename), file_time
    return None


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
    output_dir = output_root / selected_version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(urlparse(url).path).name
    if output_path.exists():
        return output_path

    with urlopen(url, timeout=timeout) as response:
        content = response.read()
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.write_bytes(content)
    temporary_path.replace(output_path)
    return output_path


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
