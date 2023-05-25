# Goalanda FFWC

Looking at FFWC water level and discharge at Goalanda
and comparing to GloFAS

To run this notebook for another location, first add the location to `bgd.yaml` file with location name and coordinates then reun pipeline.py to download and process GloFAS data needed for any further analysis.

```python
%load_ext jupyter_black
```


```python
from pathlib import Path
import os
import datetime

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir("..")
from src import utils
```


```python
# This is to avoid error with dask
from dask.distributed import Client

c = Client(n_workers=os.cpu_count() - 2, threads_per_worker=1)
```


```python
station = "Goalando"
```


```python
# FFWC data
ffwc_data_dir = (
    Path(os.environ["OAP_DATA_DIR"]) / "private/exploration/bgd/FFWC_Data"
)
df_ffwc = pd.read_excel(
    ffwc_data_dir / "Water LevelSW91.9R13529.xlsx", skiprows=9
)
danger_level = 8.66
danger_level_2 = danger_level + 0.5
danger_level_3 = danger_level + 1  # m
```


```python
# Look at discharge vs water level
fig, ax = plt.subplots()
df_ffwc.plot.scatter("DATE/TIME", "AVE_WL(m)", ax=ax, alpha=0.5)
ax.set_ylim(0, 15)
ax.set_xlim([datetime.date(2003, 1, 1), datetime.date(2022, 12, 31)])
plt.axhline(y=8.66, color="r", linestyle="-", label="Danger Level = 8.66m")
plt.axhline(y=9.16, color="g", linestyle="--", label="Danger Level + 0.5m")
plt.axhline(y=9.66, color="aqua", linestyle="--", label="Danger Level + 1m")
# plt.text(datetime.date(2019, 1, 1), 8.7, "Danger Level = 8.66", fontsize=8)
# plt.text(datetime.date(2019, 1, 1), 9.2, "Danger Level + 0.5m", fontsize=8)
# plt.text(datetime.date(2019, 1, 1), 9.7, "Danger Level + 1m", fontsize=8)
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.xlabel("Date")
plt.ylabel("Average Water Level (m)")
ax.set_title("FFWC water level at Goalando, 2003-2022")
# ax.axvline(danger_level, c="r")
```


```python
min_duration = 3
obs = df_ffwc["AVE_WL(m)"]
obs_dates_3 = utils.get_dates_list_from_data_array(
    obs, danger_level_3, min_duration=min_duration
)
print(len(obs_dates_3))
obs_dates_2 = utils.get_dates_list_from_data_array(
    obs, danger_level_2, min_duration=min_duration
)
print(len(obs_dates_2))
obs_dates = utils.get_dates_list_from_data_array(
    obs, danger_level, min_duration=min_duration
)
print(len(obs_dates))
```


```python
# Read in reforecast and reanalysis
ds_rf = utils.load_glofas_reforecast()
ds_ra = utils.load_glofas_reanalysis()

da_ra = ds_ra[station]
da_rf = ds_rf[station]
```


```python
# Take only the FFWC time
df_glofas = da_ra.sel(time=slice(df_ffwc["DATE/TIME"][0], None)).to_dataframe()
df_glofas
```


```python
# Merge the two
df_comb = (
    pd.merge(
        df_ffwc, df_glofas, how="outer", left_on="DATE/TIME", right_index=True
    )
    .sort_values(by="DATE/TIME")
    .set_index("DATE/TIME")
)
df_comb
```


```python
rp_target = 5  # 1 in 10 year
rp_value = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, station),
    rp_var="discharge",
    show_plots=True,
)(rp_target)
```


```python
rp_value
```


```python
# See if GloFAS captures FFWC "events"
min_duration = 3  # day
days_before_buffer = 5
days_after_buffer = 5

df_station_stats = pd.DataFrame()
# Get the model
model = df_comb["Goalando"]
# Limit the observations to the forecast time
obs = df_comb["AVE_WL(m)"]
# Get the dates for each
model_dates = utils.get_dates_list_from_data_array(
    model, rp_value, min_duration=min_duration
)
obs_dates = utils.get_dates_list_from_data_array(
    obs, danger_level_3, min_duration=min_duration
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

Comparing reforecast vs reanalysis


```python
# Interpolate the reforecast
da_rf.values = np.sort(da_rf.values, axis=0)

da_rf_interp = da_rf.interp(
    time=pd.date_range(da_rf.time.min().values, da_rf.time.max().values),
    method="linear",
)
# Get the median
da_rf_med = da_rf_interp.median(axis=0)
```


```python
# Loop through lead times
min_duration = 3  # day
days_before_buffer = 5
days_after_buffer = 5

df_station_stats = pd.DataFrame()
for lead_time in da_rf["step"].values:
    # Get the forecast at a specific lead time
    forecast = da_rf_med.sel(step=lead_time, method="nearest")
    # Shift the time variable of the forecast to match the date of the event
    forecast["time"] = forecast.time.values + lead_time
    # Limit the observations to the forecast time
    model = da_ra.reindex(time=forecast.time)
    # Get the dates for each
    model_dates = utils.get_dates_list_from_data_array(
        model, rp_value, min_duration=min_duration
    )
    forecast_dates = utils.get_dates_list_from_data_array(
        forecast, rp_value, min_duration=min_duration
    )
    # Match the dates to events
    detection_stats = utils.get_detection_stats(
        true_event_dates=model_dates,
        forecasted_event_dates=forecast_dates,
        days_before_buffer=days_before_buffer,
        days_after_buffer=days_after_buffer,
    )
    df_station_stats = pd.concat(
        (
            df_station_stats,
            pd.DataFrame(
                {
                    **{"station": station, "leadtime": lead_time},
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
df_station_stats["leadtime"] = df_station_stats["leadtime"] / np.timedelta64(
    1, "D"
)
```


```python
ylim = {1.5: 36, 10: 10}

df = df_station_stats.copy()
cdict = {
    "TP": "tab:olive",
    "FP": "tab:orange",
    "FN": "tab:red",
    "POD": "tab:blue",
    "FAR": "tab:cyan",
}

fig, ax = plt.subplots()
ax2 = ax.twinx()
df_station = df[df["station"] == station]
lines1, lines2 = [], []
for stat in ["TP", "FP", "FN"]:
    lines1 += ax.plot(
        df_station["leadtime"], df_station[stat], c=cdict[stat], label=stat
    )
for stat in ["POD", "FAR"]:
    lines2 += ax2.plot(
        df_station["leadtime"], df_station[stat], c=cdict[stat], label=stat
    )
lines = lines1 + lines2
labels = [line.get_label() for line in lines]
ax.legend(lines, labels)
ax.set_ylim(-1, 10)
ax2.set_ylim(-0.03, 1.03)
ax.set_ylabel("Number")
ax.set_xlabel("Lead time [days]")
ax2.set_ylabel("POD / FAR")
ax.set_title(f"{station}, RP = 1 in {rp_target} year")
```
