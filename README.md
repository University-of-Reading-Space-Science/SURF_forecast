A set of scripts for producing hindacasts for InSitu-SURF and WSA-BRaVDA-SURF. Requires SURF to be installed.

Each hindcast CSV holds both a HUXt and a hydro run (columns vsw_huxt, vsw_hydro, ...).
WSA-BRaVDA-SURF uses only the newest WSA map at or before each issue time, read from
~/Dropbox/Data/WSA/WSA_CCMC/<version>/; run download_wsa_archive.py to fill that archive.

compare_forecast_mae.py also verifies high-speed streams as events (hits, misses and false
alarms by lead day, with arrival-time and peak-speed errors); see the HSS_ settings in it.
