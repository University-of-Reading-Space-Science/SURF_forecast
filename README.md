A set of scripts for producing hindacasts for InSitu-SURF and WSA-BRaVDA-SURF. Requires SURF to be installed.

Each hindcast CSV holds both a HUXt and a hydro run (columns vsw_huxt, vsw_hydro, ...).
WSA-BRaVDA-SURF uses only the newest WSA map at or before each issue time, read from
~/Dropbox/Data/WSA/WSA_CCMC/<version>/; run download_wsa_archive.py to fill that archive.

generate_wsaminus3_bravda_surf_forecasts.py repeats the WSA-BRaVDA-SURF runs using the newest
WSA map available 3 days before each issue time (OMNI assimilation unchanged), writing to
wsaminus3_bravda_surf_forecasts/. compare_forecast_mae.py compares the two (map-age figure).
