"""OMNI, STEREO-A, SWPC and beacon downloaders, independent of SURF.

All return DataFrames with UTC ``datetime``, ``mjd`` and ``V`` (km/s) columns.
Optional download dependencies are imported only when their grabber is called.
"""

import json
from importlib.resources import files
import re
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
import warnings

from astropy.time import Time
import numpy as np


def cane_richardson_icmes(filepath=None):
    """Read the bundled Richardson--Cane near-Earth ICME catalogue.

    Only the event times needed for masking are returned. ``filepath`` may be
    used to supply another CSV in the same Richardson--Cane format.
    """
    import pandas as pd

    source = (files("simplebravda_data") /
              "Richardson_Cane_Processed_ICME_list.csv")
    if filepath is not None:
        source = filepath
    with source.open("rb") if hasattr(source, "open") else open(source, "rb") as stream:
        events = pd.read_csv(
            stream, header=None, usecols=[0, 1, 2],
            names=["Shock_time", "ICME_start", "ICME_end"],
        )
    for column in events.columns:
        events[column] = pd.to_datetime(
            events[column], format="%Y/%m/%d %H%M", errors="coerce"
        )
    return events.dropna().reset_index(drop=True)


def stereo_icmes(spacecraft="A", filepath=None):
    """Read the bundled Jian et al. STEREO ICME catalogue."""
    import pandas as pd

    spacecraft = spacecraft.upper()
    if spacecraft not in ("A", "B"):
        raise ValueError("spacecraft must be 'A' or 'B'")
    source = files("simplebravda_data") / "LanJian_STEREO_ICME_List.txt"
    if filepath is not None:
        source = filepath
    if hasattr(source, "read_text"):
        catalogue = source.read_text(encoding="utf-8")
    else:
        with open(source, encoding="utf-8") as stream:
            catalogue = stream.read()

    isot = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z"
    pattern = re.compile(
        rf"({isot})\s+({isot})\s+({isot})\s+(-?1)\s+[01]\s+-?\d+"
    )
    rows = pattern.findall(catalogue)
    if not rows:
        raise ValueError(f"No STEREO ICME events could be parsed from {source}")
    events = pd.DataFrame(
        rows,
        columns=["Shock_time", "ICME_end", "magnetic_obstacle_start", "STEREO"],
    )
    events["STEREO"] = events["STEREO"].astype(int)
    events = events.loc[events["STEREO"] == (1 if spacecraft == "A" else -1)].copy()
    for column in ("Shock_time", "ICME_end", "magnetic_obstacle_start"):
        events[column] = pd.to_datetime(events[column], utc=True).dt.tz_localize(None)
    events["ICME_start"] = events["magnetic_obstacle_start"]
    return events.reset_index(drop=True)


def _remove_icme_intervals(frame, events, pre_icme_buffer,
                           post_icme_buffer, interp_gaps):
    """Apply parsed ICME intervals to the supported in-situ columns."""
    import pandas as pd

    pre = float(pre_icme_buffer)
    post = float(post_icme_buffer)
    if not np.isfinite(pre) or not np.isfinite(post) or pre < 0 or post < 0:
        raise ValueError("ICME buffers must be finite and non-negative")
    missing = {"datetime", "V"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    sample_time = pd.to_datetime(result["datetime"], utc=True).dt.tz_localize(None)
    parameters = [name for name in ("V", "BX_GSE") if name in result.columns]
    for event in events.itertuples(index=False):
        start = event.Shock_time - pd.Timedelta(days=pre)
        stop = event.ICME_end + pd.Timedelta(days=post)
        result.loc[sample_time.between(start, stop), parameters] = np.nan
    if interp_gaps:
        for parameter in parameters:
            series = pd.Series(result[parameter].to_numpy(), index=sample_time)
            result[parameter] = (
                series.interpolate(method="time").ffill().bfill().to_numpy()
            )
    return result


def remove_cane_richardson_icmes(frame, *, pre_icme_buffer=0.2,
                                 post_icme_buffer=1.0, interp_gaps=False,
                                 catalogue_path=None):
    """Mask Richardson--Cane ICME intervals in an in-situ DataFrame.

    ``V`` and, when present, ``BX_GSE`` are set to NaN from the shock time
    minus ``pre_icme_buffer`` days through the ICME end plus
    ``post_icme_buffer`` days. Set ``interp_gaps=True`` to time-interpolate the
    masked values; leaving it false lets simpleBRaVDA omit those observations.
    """
    events = cane_richardson_icmes(catalogue_path)
    return _remove_icme_intervals(
        frame, events, pre_icme_buffer, post_icme_buffer, interp_gaps
    )


def remove_stereo_icmes(frame, *, spacecraft="A", pre_icme_buffer=0.2,
                        post_icme_buffer=1.0, interp_gaps=False,
                        catalogue_path=None):
    """Mask Jian et al. STEREO ICME intervals in an in-situ DataFrame."""
    return _remove_icme_intervals(
        frame, stereo_icmes(spacecraft, catalogue_path), pre_icme_buffer,
        post_icme_buffer, interp_gaps,
    )


def stereo_a_position(time):
    """Return STEREO-A radius, HGS latitude and HGS longitude from Horizons."""
    from sunpy.coordinates import get_horizons_coord

    coordinate = get_horizons_coord("STEREO-A", Time(time))
    return coordinate.radius.to("AU"), coordinate.lat.to("rad"), coordinate.lon.to("rad")


# Adapted from SURF/surf/surf_insitu.py; no SURF dependency is needed here.
_ISWA_HAPI_URL = 'https://iswa.ccmc.gsfc.nasa.gov/hapi/data'


def get_omni(starttime, endtime):
    """Download hourly OMNI COHO merged plasma and magnetic data via Fido."""
    from sunpy.net import Fido, attrs
    from sunpy.timeseries import TimeSeries

    trange = attrs.Time(starttime, endtime)
    dataset = attrs.cdaweb.Dataset('OMNI_COHO1HR_MERGED_MAG_PLASMA')
    result = Fido.search(trange, dataset)
    downloaded_files = Fido.fetch(result)

    # Import the OMNI data
    data = TimeSeries(downloaded_files, concatenate=True)

    omni = data.to_dataframe()
    del data

    # Set invalid data points to NaN
    id_bad = omni['V'] == 9999.0
    omni.loc[id_bad, 'V'] = np.nan
    
    id_bad = omni['N'] == 999.9
    omni.loc[id_bad, 'N'] = np.nan
    
    id_bad = omni['T'] == 9999999.0
    omni.loc[id_bad, 'T'] = np.nan

    # Create the BX_GSE field expected by some SURF functions.
    omni['BX_GSE'] = -omni['BR']

    # create a datetime column
    omni['datetime'] = omni.index
    # add a mjd column too
    omni['mjd'] = Time(omni['datetime']).mjd
    # reset the index
    omni = omni.reset_index()

    return omni


def _get_iswa_hapi(dataset_id, starttime, endtime, timeout=60):
    """Download ISWA HAPI data, splitting intervals at its 31-day limit."""
    import pandas as pd

    start = pd.Timestamp(starttime)
    end = pd.Timestamp(endtime)
    if start >= end:
        raise ValueError('starttime must be before endtime')

    def hapi_time(value):
        if value.tzinfo is not None:
            value = value.tz_convert('UTC').tz_localize(None)
        return value.strftime('%Y-%m-%dT%H:%M:%SZ')

    records = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=31), end)
        query = urlencode({
            'id': dataset_id,
            'time.min': hapi_time(chunk_start),
            'time.max': hapi_time(chunk_end),
            'format': 'json',
        })
        try:
            with urlopen(f'{_ISWA_HAPI_URL}?{query}', timeout=timeout) as response:
                payload = json.load(response)
        except HTTPError as error:
            # HAPI puts the useful reason (e.g. range limits) in the JSON body.
            detail = error.read().decode('utf-8', errors='replace')
            try:
                detail = json.loads(detail).get('status', {}).get('message', detail)
            except (ValueError, AttributeError):
                pass
            raise RuntimeError(
                f'ISWA HAPI request for {dataset_id} from {hapi_time(chunk_start)} '
                f'to {hapi_time(chunk_end)} failed (HTTP {error.code}): {detail}'
            ) from error
        if payload.get('status', {}).get('code') not in (1200, 1201):
            message = payload.get('status', {}).get('message', 'unknown HAPI error')
            raise RuntimeError(f'ISWA HAPI request for {dataset_id} failed: {message}')
        chunk_records = payload.get('data', [])
        if chunk_records:
            parameters = payload['parameters']
            records.extend(chunk_records)
        chunk_start = chunk_end
    if not records:
        raise ValueError(f'No {dataset_id} data are available from {start} through {end}')
    columns = [parameter['name'] for parameter in parameters]
    frame = pd.DataFrame(records, columns=columns)
    frame['datetime'] = pd.to_datetime(frame.pop('Time'), utc=True).dt.tz_localize(None)
    for parameter in parameters[1:]:
        name = parameter['name']
        if parameter.get('type') in {'double', 'float', 'integer'} and name in frame:
            frame[name] = pd.to_numeric(frame[name], errors='coerce')
            fill = parameter.get('fill')
            if fill not in {None, 'null'}:
                frame.loc[frame[name] == float(fill), name] = np.nan
    if 'isPrimary' in frame and (frame['isPrimary'] == 1).any():
        frame = frame.loc[frame['isPrimary'] == 1]
    return frame.sort_values('datetime').drop_duplicates('datetime', keep='last')


def get_SWPC_realtime(starttime, endtime, include_mag=True, timeout=60):
    """Download NOAA/SWPC real-time L1 solar-wind data from NASA ISWA HAPI.

    The returned DataFrame follows :func:`get_omni` conventions so it can be
    supplied as ``omni_input`` to the SURF forecast and reconstruction helpers.
    Plasma is read from ``swpc_rtsw_plasma_P1M``. The companion magnetic feed
    ``swpc_rtsw_mag_P1M`` is merged by one-minute timestamp when requested.
    Intervals longer than ISWA's 31-day limit are downloaded in chunks.
    """
    import pandas as pd

    plasma = _get_iswa_hapi(
        'swpc_rtsw_plasma_P1M', starttime, endtime, timeout=timeout
    ).rename(columns={
        'BulkSpeed': 'V',
        'ProtonDensity': 'N',
        'IonTemperature': 'T',
        'source': 'plasma_source',
        'isPrimary': 'plasma_isPrimary',
    })

    combined = plasma
    if include_mag:
        try:
            magnetic = _get_iswa_hapi(
                'swpc_rtsw_mag_P1M', starttime, endtime, timeout=timeout
            ).rename(columns={
                'source': 'mag_source',
                'isPrimary': 'mag_isPrimary',
            })
            combined = pd.merge(plasma, magnetic, on='datetime', how='left')
        except Exception as error:
            warnings.warn(
                f'SWPC magnetic data could not be loaded ({error}); returning plasma only.',
                RuntimeWarning,
                stacklevel=2,
            )

    combined['BX_GSE'] = combined['B_x'] if 'B_x' in combined else np.nan
    combined['BR'] = -combined['BX_GSE']
    combined['B'] = combined['B_t'] if 'B_t' in combined else np.nan
    combined['mjd'] = Time(combined['datetime'].to_numpy()).mjd
    if not np.isfinite(combined['V']).any():
        raise ValueError('The SWPC real-time feed contains no usable solar-wind speeds.')
    return combined.sort_values('datetime').reset_index(drop=True)


def get_stereo_a(starttime, endtime):
    """Download hourly merged STEREO-A plasma and magnetic-field data via Fido.

    The returned columns follow the same convention as :func:`get_omni`.
    """
    import pandas as pd
    from sunpy.net import Fido, attrs
    from sunpy.timeseries import TimeSeries

    trange = attrs.Time(starttime, endtime)
    dataset = attrs.cdaweb.Dataset('STA_COHO1HR_MERGED_MAG_PLASMA')
    result = Fido.search(trange, dataset)
    downloaded_files = Fido.fetch(result)
    data = TimeSeries(downloaded_files, concatenate=True).to_dataframe()

    sta = pd.DataFrame(index=data.index)
    sta['V'] = data['plasmaSpeed']
    sta['N'] = data['plasmaDensity']
    sta['T'] = data['plasmaTemp']
    # generate_vCarr_from_OMNI expects BX_GSE and changes its sign to recover Br.
    sta['BX_GSE'] = -data['BR']
    sta['BR'] = data['BR']
    sta['B'] = data['B']
    sta['datetime'] = sta.index
    sta = sta.loc[(sta['datetime'] >= pd.Timestamp(starttime)) &
                  (sta['datetime'] <= pd.Timestamp(endtime))].copy()

    # CDAWeb fill values are sometimes retained by CDF readers. Convert all
    # implausibly large magnitudes to missing data before interpolation.
    sta.loc[(sta['V'] <= 0) | (sta['V'] >= 9999), 'V'] = np.nan
    sta.loc[(sta['N'] < 0) | (sta['N'] >= 999), 'N'] = np.nan
    sta.loc[(sta['T'] < 0) | (sta['T'] >= 9999999), 'T'] = np.nan
    for column in ('BX_GSE', 'BR', 'B'):
        sta.loc[np.abs(sta[column]) >= 9999, column] = np.nan
    sta['mjd'] = Time(sta['datetime']).mjd
    return sta.reset_index(drop=True)


def get_stereo_a_beacon(starttime, endtime):
    """Fetch hourly STEREO-A PLASTIC beacon speeds in SURF DataFrame format.

    Fetch the CDAWeb STA_LB_PLA_BROWSE dataset with SunPy, replace fill values,
    and average hourly with a 30-minute offset. Returns a DataFrame without
    exporting BRaVDA observation files.
    Only samples within the requested UTC interval contribute to the averages.
    Install the ``beacon`` extra to enable this optional downloader.
    """
    import pandas as pd
    from sunpy.net import Fido, attrs
    from sunpy.timeseries import TimeSeries
    from sunpy.util.exceptions import SunpyUserWarning

    start, end = Time(starttime).utc, Time(endtime).utc
    if not start.isscalar or not end.isscalar or start >= end:
        raise ValueError('starttime must be before endtime, both scalar UTC times.')

    trange = attrs.Time(start.datetime, end.datetime)
    dataset = attrs.cdaweb.Dataset('STA_LB_PLA_BROWSE')
    downloaded_files = Fido.fetch(Fido.search(trange, dataset))
    if not downloaded_files:
        raise ValueError('No STEREO-A beacon data are available for this interval.')
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'^astropy did not recognize units of "step"\.',
            category=SunpyUserWarning,
        )
        data = TimeSeries(downloaded_files, concatenate=True).to_dataframe()
    speed = pd.to_numeric(data['Bulk_Speed'], errors='coerce').copy()
    speed.index = pd.to_datetime(speed.index, utc=True).tz_localize(None)
    speed = speed.loc[(speed.index >= start.datetime) & (speed.index <= end.datetime)]
    speed = speed.where(np.isfinite(speed) & (speed > 0) & (speed < 9999)).sort_index()
    hourly = speed.resample('h', label='left', offset='30min').mean()
    # An offset bin can start before the requested window.
    hourly = hourly.loc[(hourly.index >= start.datetime) & (hourly.index <= end.datetime)]
    if not np.isfinite(hourly).any():
        raise ValueError('The STEREO-A beacon feed contains no usable solar-wind speeds.')
    frame = hourly.rename('V').rename_axis('datetime').reset_index()
    frame['mjd'] = Time(frame['datetime'].to_numpy()).mjd
    frame['BX_GSE'] = np.nan  # Retain the common in-situ frame schema.
    return frame
