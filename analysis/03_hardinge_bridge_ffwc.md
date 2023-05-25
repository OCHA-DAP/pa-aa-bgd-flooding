# Hardinge Bridge FFWC

Looking at FFWC water level and discharge at Hardinge Bridge
and comparing to GloFAS

```python
%load_ext jupyter_black
```

```python
from pathlib import Path
import os

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import utils
```

```python
# This is to avoid error with dask
from dask.distributed import Client

c = Client(n_workers=os.cpu_count() - 2, threads_per_worker=1)
```

```python
station = "Hardinge Bridge"
```

```python
# Read in rereforecast andanalysis
ds_ra = utils.load_glofas_reanalysis()
da_ra = ds_ra[station]
```

```python
# FFWC data
ffwc_data_dir = (
    Path(os.environ["OAP_DATA_DIR"]) / "private/exploration/bgd/FFWC_Data"
)
df_ffwc = pd.read_excel(ffwc_data_dir / "DischargeSW9013527.xlsx", skiprows=9)
danger_level = 13.8  # m, I read it off the FFWC site

# ds_wl = pd.read_excel(
#    ffwc_data_dir / "Water LevelSW91.9R13529.xlsx", skiprows=9
# )
```

```python
df_ffwc
```

```python
# Look at discharge vs water level
fig, ax = plt.subplots()
df_ffwc.plot.scatter("WATER LVL(m)", "DISCHARGE(m)3/s", ax=ax, alpha=0.5)
ax.set_ylim(-1_000, 80_000)
ax.set_title("FFWC discharge vs water level at Hardinge Bridge, 2003-2022")
ax.axvline(danger_level, c="r")
```

```python
# Take only the FFWC time
df_glofas = da_ra.sel(time=slice(df_ffwc["DATE TIME"][0], None)).to_dataframe()
df_glofas
```

```python
# Merge the two
df_comb = (
    pd.merge(
        df_ffwc, df_glofas, how="outer", left_on="DATE TIME", right_index=True
    )
    .sort_values(by="DATE TIME")
    .set_index("DATE TIME")
)
df_comb
```

```python
# Compare discharges
fig, ax = plt.subplots()
df_comb.plot.scatter("DISCHARGE(m)3/s", "Hardinge Bridge", ax=ax, alpha=0.5)
ax.set_xlim(-1_000, 85_000)
ax.set_ylim(-1_000, 85_000)
ax.plot(np.arange(-1000, 90_000), c="k")
ax.set_title(
    "GloFAS discharge vs FFWC discharge at Hardinge Bridge, 2003-2022"
)
ax.set_xlabel("FFWC discharge [m^3/s]")
ax.set_ylabel("GloFAS discharge [m^3/s]")
```

```python
# Get return period value
rp_target = 5  # 1 in 10 year
rp_value = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, station),
    rp_var="discharge",
    show_plots=True,
)(rp_target)
```

```python
# See if GloFAS captures FFWC "events"
min_duration = 3  # day
days_before_buffer = 5
days_after_buffer = 5

df_station_stats = pd.DataFrame()
# Get the model
model = df_comb["Hardinge Bridge"]
# Limit the observations to the forecast time
obs = df_comb["WATER LVL(m)"]
# Get the dates for each
model_dates = utils.get_dates_list_from_data_array(
    model, rp_value, min_duration=min_duration
)
obs_dates = utils.get_dates_list_from_data_array(
    obs, danger_level, min_duration=min_duration
)
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=obs_dates,
    forecasted_event_dates=model_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
df_station_stats = pd.concat(
    (
        df_station_stats,
        pd.DataFrame(
            {
                **{"station": station},
                **detection_stats,
            },
            index=[0],
        ),
    ),
    ignore_index=True,
)
df_station_stats = utils.get_more_detection_stats(df_station_stats)
df_station_stats
```

```python
obs_dates
```

```python
model_dates
```
