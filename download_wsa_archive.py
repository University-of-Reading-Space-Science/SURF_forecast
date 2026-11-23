"""Rebuild the local CCMC WSA archive with one pre-issue-time map per day.

For every day in DOWNLOAD_START..DOWNLOAD_END this downloads the newest GONG_Z
WSA velocity map at or before 00:00 UTC (i.e. the last map issued before
midnight), using the same version preference and maximum age as the hindcast
script. Files are stored in ``<LOCAL_WSA_DIRECTORY>/<version>/``; maps already
on disk are not downloaded again, so the script can be re-run to fill gaps.
"""

import datetime as dt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "simpleBRaVDA"))
from wsa import download_wsa_daily_archive, read_wsa

from generate_wsa_bravda_surf_forecasts import (
    LOCAL_WSA_DIRECTORY, WSA_MAX_AGE, WSA_VERSIONS,
)


# The WSA5.X GONG_Z archive on ISWA covers 2006-2022; WSA5.4 starts in Dec 2022.
DOWNLOAD_START = dt.date(2019, 1, 1)
DOWNLOAD_END = dt.date(2023, 1, 1)


def main():
    print(f"Downloading daily pre-midnight WSA maps to {LOCAL_WSA_DIRECTORY}")
    results = download_wsa_daily_archive(
        DOWNLOAD_START, DOWNLOAD_END, LOCAL_WSA_DIRECTORY,
        versions=WSA_VERSIONS, max_age=WSA_MAX_AGE,
    )
    paths = [path for _, path, _ in results if path is not None]
    missing = [issue for issue, path, _ in results if path is None]
    print(f"{len(paths)} of {len(results)} issue days have a map; {len(missing)} missing")

    # The WSA5.X files were not previously read by this code, so check that
    # the first and last maps parse and that the FITS time matches the name.
    for path in (paths[:1] + paths[-1:]) if paths else []:
        speed, longitude, latitude, map_time = read_wsa(path)
        print(f"{path.name}: speed {speed.shape}, {speed.min():.0f}-{speed.max():.0f}; "
              f"OBSJUL {map_time.isot}")


if __name__ == "__main__":
    main()
