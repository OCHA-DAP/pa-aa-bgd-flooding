# GloFAS forecast

Compare GloFAS reforecast to model

```python
%load_ext jupyter_black
```

```python
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
rp_target = 10  # 1 in 10 year
```

```python
ds_rf = utils.load_glofas_reforecast()
ds_ra = utils.load_glofas_reanalysis()
```

```python
# Just do one station for now
station = "Bahadurabad"
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
rp_value = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(da_ra, station),
    rp_var="discharge",
    show_plots=True,
)(rp_target)
```

```python
rp_value = 98_000
min_duration = 1  # day
df_station_stats = pd.DataFrame()
days_before_buffer = 5
days_after_buffer = 5

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
```

```python
model = da_ra.reindex

model = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station + parameters.VERSION_LOC]
forecast = ds_glofas_reforecast_summary[station + parameters.VERSION_LOC].sel(percentile=parameters.MAIN_FORECAST_PROB)
for rp in RP_LIST:
    rp_val = df_return_period.loc[rp, station]
    model_dates = utils.get_dates_list_from_data_array(model,  rp_val, min_duration=parameters.DURATION)
    for leadtime in parameters.LEADTIMES:
        forecast_dates = utils.get_dates_list_from_data_array(
            forecast.sel(leadtime=leadtime), rp_val, min_duration=parameters.DURATION)
        detection_stats = utils.get_detection_stats(true_event_dates=model_dates,
                                                   forecasted_event_dates=forecast_dates,
                                                   days_before_buffer=parameters.DAYS_BEFORE_BUFFER,
                                                   days_after_buffer=parameters.DAYS_AFTER_BUFFER)
        df_station_stats = df_station_stats.append({
            **{'station': station,
            'leadtime': leadtime,
            'rp': rp},
            **detection_stats
        }, ignore_index=True)

df_station_stats = utils.get_more_detection_stats(df_station_stats)
df_station_stats
```
