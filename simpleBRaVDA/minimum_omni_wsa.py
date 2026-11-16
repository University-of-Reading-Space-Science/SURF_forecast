"""Minimum end-to-end simpleBRaVDA example using OMNI and a WSA map."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import astropy.units as u
from astropy.time import Time
import numpy as np

import insitu
from simpleBRaVDA import (
    PERIOD_DAYS,
    deaccelerate_wsa_speeds,
    omni_observations,
    simpleBRaVDA,
)
from wsa import get_wsa_from_iswa, read_wsa


# The forecast time is also the cutoff for observations used by BRaVDA.
FORECAST_TIME = Time("2022-08-10T00:00:00", scale="utc")
INNER_RADIUS = 0.1 * u.AU


def main():
    start_time = FORECAST_TIME - PERIOD_DAYS * u.day

    # Download the preceding Carrington rotation of hourly near-Earth speeds.
    omni_frame = insitu.get_omni(start_time.datetime, FORECAST_TIME.datetime)
    omni_frame = insitu.remove_cane_richardson_icmes(omni_frame, interp_gaps=False)
    omni = omni_observations(FORECAST_TIME, Time(omni_frame["datetime"].to_numpy()), omni_frame["V"].to_numpy() * u.km / u.s)

    # Download the WSA map selected by ISWA for the forecast time. 
    wsa_path = get_wsa_from_iswa(FORECAST_TIME.datetime, datadir=Path("wsa_data"))
    speed_1au, map_longitude, map_latitude, _ = read_wsa(wsa_path)
    
    # ISWA WSA speeds are calibrated at 1 AU, so convert them to the inner boundary.
    speed_inner = deaccelerate_wsa_speeds(speed_1au, INNER_RADIUS)

    boundary = simpleBRaVDA(FORECAST_TIME, omni, speed_inner, map_longitude, map_latitude, inner_radius=INNER_RADIUS)

    # These corresponding arrays are the requested inner-boundary condition.
    v_inner = boundary.speed.to_value(u.km / u.s)
    carrington_longitude = boundary.longitude.to_value(u.deg)

if __name__ == "__main__":
    main()
