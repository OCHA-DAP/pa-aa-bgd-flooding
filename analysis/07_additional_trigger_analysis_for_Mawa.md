
# Additional Analysis Requested by Hassan

## Re-doing the analysis for Mawa Station

This notebook replicates some of the analysis done for the Goalando station.

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
ds_rf = utils.load_glofas_reforecast()
ds_ra = utils.load_glofas_reanalysis()
```

```python
# FFWC data
output_dir = Path(os.environ["AA_DATA_DIR"]) / "private/exploration/bgd/"
ffwc_data_dir = (
    Path(os.environ["AA_DATA_DIR"]) / "private/exploration/bgd/FFWC_Data"
)
padma_ffwc = pd.read_excel(
    ffwc_data_dir / "Water LevelSW91.9R13529.xlsx", skiprows=9
)
jamuna_ffwc_dir = (
    Path(os.environ["AA_DATA_DIR"]) / "private/exploration/bgd/ffwc"
)
jamuna_ffwc = pd.read_excel(ffwc_data_dir / "SW46.9L_19-11-2020.xlsx")
jamuna_ffwc["date"] = (pd.to_datetime(jamuna_ffwc["DateTime"])).dt.date
padma_dl = 6.11
padma_dl_2 = padma_dl + 0.5
padma_dl_3 = padma_dl + 1  # m
```

```python
jamuna_ffwc = utils.read_in_ffwc()
jamuna_ffwc.drop(
    [
        "obs_std",
        "ffwc_1day",
        "ffwc_2day",
        "ffwc_3day",
        "ffwc_4day",
        "ffwc_5day",
    ],
    axis=1,
    inplace=True,
)
```

```python
min_duration = 3  # day
days_before_buffer = 5
days_after_buffer = 5
```

## The FAR and POD calculated at Bahadurabad, Hardinge Bridge, and Goalando

```python
da_rf_b = ds_rf["Bahadurabad"]
da_rf_h = ds_rf["Hardinge Bridge"]
da_rf_g = ds_rf["Goalando"]
da_rf_m = ds_rf["Mawa"]
da_ra_b = ds_ra["Bahadurabad"]
da_ra_h = ds_ra["Hardinge Bridge"]
da_ra_g = ds_ra["Goalando"]
da_ra_m = ds_ra["Mawa"]
```

### Mawa

```python
station = "Mawa"
# Interpolate the reforecast
da_rf_m.values = np.sort(da_rf_m.values, axis=0)

da_rf_interp_m = da_rf_m.interp(
    time=pd.date_range(da_rf_m.time.min().values, da_rf_m.time.max().values),
    method="linear",
)
# Get the median
da_rf_med_m = da_rf_interp_m.median(axis=0)
rp_target = 5  # 1 in 5 year
padma_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, station),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
padma_gf_rp
```

```python
df_station_stats = pd.DataFrame()
for lead_time in da_rf_m["step"].values:
    # Get the forecast at a specific lead time
    forecast = da_rf_med_m.sel(step=lead_time, method="nearest")
    # Shift the time variable of the forecast to match the date of the event
    forecast["time"] = forecast.time.values + lead_time
    # Limit the observations to the forecast time
    model = da_ra_m.reindex(time=forecast.time)
    # Get the dates for each
    model_dates = utils.get_dates_list_from_data_array(
        model, padma_gf_rp, min_duration=min_duration
    )
    forecast_dates = utils.get_dates_list_from_data_array(
        forecast, padma_gf_rp, min_duration=min_duration
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
model_dates
```

```python
forecast_dates
```

```python
df_station_stats.to_csv(
    output_dir / "Mawa GloFAS Model vs Forecast Detection Stats.csv",
    index=False,
)
```

```python
df_station_stats["leadtime"] = df_station_stats["leadtime"] / np.timedelta64(
    1, "D"
)
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

## Interaction of events in Padma(Mawa) and Jamuna(Bahadurabad)

```python
da_ra_m = ds_ra["Mawa"]
da_ra_b = ds_ra["Bahadurabad"]
```

### The GloFAS model 1-in-5 year RP

```python
rp_target = 5  # 1 in 5 year
padma_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, "Mawa"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
jamuna_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, "Bahadurabad"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
padma_gf_rp
```

```python
jamuna_gf_rp
```

```python
glofas_comb = pd.merge(
    da_ra_b.to_dataframe(),
    da_ra_m.to_dataframe(),
    how="outer",
    left_on="time",
    right_index=True,
)
```

```python
df_station_stats = pd.DataFrame()
# Get the model data
padma_glofas = glofas_comb["Mawa"]
jamuna_glofas = glofas_comb["Bahadurabad"]
# Get the dates for each
padma_dates = utils.get_dates_list_from_data_array(
    padma_glofas, padma_gf_rp, min_duration=min_duration
)
jamuna_dates = utils.get_dates_list_from_data_array(
    jamuna_glofas, jamuna_gf_rp, min_duration=min_duration
)
```

```python
padma_dates
```

```python
jamuna_dates
```

#### Event in Padma given event in Jamuna

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=jamuna_dates,
    forecasted_event_dates=padma_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```

```python
detection_stats["TP"] / len(jamuna_dates)
```

#### Event in Jamuna given event in Padma

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=padma_dates,
    forecasted_event_dates=jamuna_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```

```python
detection_stats["TP"] / len(padma_dates)
```

#### Event in Padma given events in Jamuna

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=jamuna_dates,
    forecasted_event_dates=padma_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```

```python
detection_stats["TP"] / len(jamuna_dates)
```

#### Event in Jamuna given events in Padma

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=padma_dates,
    forecasted_event_dates=jamuna_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```

```python
detection_stats["TP"] / len(padma_dates)
```

## RP for FFWC and GloFAS Reanalysis

### GloFAS Reanalysis

```python
station = "Mawa"
```

```python
rp_target = 5  # 1 in 5 year
rp_value = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, station),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
rp_value
```

```python
rp_target = 10  # 1 in 10 year
rp_value = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, station),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
rp_value
```
