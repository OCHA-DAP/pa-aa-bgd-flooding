# GloFAS forecast

Compare GloFAS reforecast to model

```python
%load_ext jupyter_black
```

```python
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import utils
```

```python
# Delete after development
from importlib import reload

reload(utils)
```

```python
# This is to avoid error with dask
import os

from dask.distributed import Client

c = Client(n_workers=os.cpu_count() - 2, threads_per_worker=1)
```

```python
# Read in reforecast and reanalysis
ds_rf = utils.load_glofas_reforecast()
ds_ra = utils.load_glofas_reanalysis()
```

## Run for a single station

```python
# Just do one station for now
station = "Noonkhawa"
da_rf = ds_rf[station]
da_ra = ds_ra[station]
```

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
# Get return period value
rp_target = 10  # 1 in 10 year
rp_value = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(da_ra, station),
    rp_var="discharge",
    show_plots=True,
)(rp_target)
```

```python
# Loop through lead times
min_duration = 1  # day
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

## Get stats for all stations

```python
rp_targets = [1.5, 10]  # 1 in x year
show_rp_plots = False
min_duration = 1  # day
days_before_buffer = 5
days_after_buffer = 5

df_station_stats = pd.DataFrame()
for station in ds_rf:
    print(f"Running for {station}")
    # Get the data array
    da_rf = ds_rf[station]
    da_ra = ds_ra[station]
    # Interpolate the reforecast
    da_rf.values = np.sort(da_rf.values, axis=0)
    da_rf_interp = da_rf.interp(
        time=pd.date_range(da_rf.time.min().values, da_rf.time.max().values),
        method="linear",
    )
    # Get the median
    da_rf_med = da_rf_interp.median(axis=0)
    for rp_target in rp_targets:
        # Get return period value
        rp_value = utils.get_return_period_function_analytical(
            df_rp=utils.get_return_period_df(da_ra, station),
            rp_var="discharge",
            show_plots=show_rp_plots,
        )(rp_target)

        # Loop through lead times and get stats
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
                            **{
                                "station": station,
                                "leadtime": lead_time,
                                "rp": rp_target,
                            },
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

rp_target = 10

pdf = matplotlib.backends.backend_pdf.PdfPages(f"output_rp{rp_target}.pdf")
df = df_station_stats[df_station_stats["rp"] == rp_target]
cdict = {
    "TP": "tab:olive",
    "FP": "tab:orange",
    "FN": "tab:red",
    "POD": "tab:blue",
    "FAR": "tab:cyan",
}

for station in ds_ra:
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
    pdf.savefig(fig)
pdf.close()
```

```python

```
