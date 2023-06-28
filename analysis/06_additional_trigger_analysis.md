
# Additional Analysis Requested by Hassan

This notebook covers 3 analyses requested by Hassan for the trigger.

- The FAR and POD calculated at Bahadurabad, Hardinge Bridge,
and Goalando, in Excel file form. Note that for Goalando he wants us
to compute it by
(1) a point on the GloFAS raster,
(2) by combining the discharge at Hardinge bridge
with 1 day lag time with that at Bahadurabad with 2 day lag time
- Calculate the probability of flooding at Bahadurabad
if there is flooding at Goalando, and vice versa
- Calculate 1-in-5 and 1-in-10 year RPs at Goalando
for both GloFAS and FFWC data

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
output_dir = Path(os.environ["OAP_DATA_DIR"]) / "private/exploration/bgd/"
ffwc_data_dir = (
    Path(os.environ["OAP_DATA_DIR"]) / "private/exploration/bgd/FFWC_Data"
)
padma_ffwc = pd.read_excel(
    ffwc_data_dir / "Water LevelSW91.9R13529.xlsx", skiprows=9
)
jamuna_ffwc_dir = ffwc_data_dir = (
    Path(os.environ["OAP_DATA_DIR"]) / "private/exploration/bgd/ffwc"
)
jamuna_ffwc = pd.read_excel(ffwc_data_dir / "SW46.9L_19-11-2020.xlsx")
jamuna_ffwc["date"] = (pd.to_datetime(jamuna_ffwc["DateTime"])).dt.date
padma_dl = 8.66
padma_dl_2 = padma_dl + 0.5
padma_dl_3 = padma_dl + 1  # m
```

```python
def read_in_ffwc():
    # Read in data from Sazzad that has forecasts
    ffwc_wl_filename = "Bahadurabad_WL_forecast20172019.xlsx"
    ffwc_leadtimes = [1, 2, 3, 4, 5]

    # Need to combine the three sheets
    df_ffwc_wl_dict = pd.read_excel(
        jamuna_ffwc_dir / ffwc_wl_filename,
        sheet_name=None,
        header=[1],
        index_col="Date",
    )
    df_ffwc_wl = (
        pd.concat(
            [
                df_ffwc_wl_dict["2017"],
                df_ffwc_wl_dict["2018"],
                df_ffwc_wl_dict["2019"],
            ],
            axis=0,
        ).rename(
            columns={
                f"{leadtime*24} hrs": f"ffwc_{leadtime}day"
                for leadtime in ffwc_leadtimes
            }
        )
    ).drop(
        columns=["Observed WL"]
    )  # drop observed because we will use the mean later
    # Convert date time to just date
    df_ffwc_wl.index = df_ffwc_wl.index.floor("d")

    # Then read in the older data (goes back much futher)
    FFWC_RL_HIS_FILENAME = (
        "2020-06-07 Water level data Bahadurabad Upper danger level.xlsx"
    )
    ffwc_rl_name = "{}/{}".format(jamuna_ffwc_dir, FFWC_RL_HIS_FILENAME)
    df_ffwc_wl_old = pd.read_excel(ffwc_rl_name, index_col=0, header=0)
    df_ffwc_wl_old.index = pd.to_datetime(
        df_ffwc_wl_old.index, format="%d/%m/%y"
    )
    df_ffwc_wl_old = df_ffwc_wl_old[["WL"]].rename(columns={"WL": "observed"})[
        df_ffwc_wl_old.index < df_ffwc_wl.index[0]
    ]
    df_ffwc_wl = pd.concat([df_ffwc_wl_old, df_ffwc_wl])

    # Read in the more recent file from Hassan
    ffwc_full_data_filename = "SW46.9L_19-11-2020.xls"
    df_ffwc_wl_full = (
        pd.read_excel(
            jamuna_ffwc_dir / ffwc_full_data_filename, index_col="DateTime"
        ).rename(columns={"WL(m)": "observed"})
    )[["observed"]]

    # Mutliple observations per day. Find mean and std
    df_ffwc_wl_full["date"] = df_ffwc_wl_full.index.date
    df_ffwc_wl_full = (df_ffwc_wl_full.groupby("date").agg(["mean", "std"]))[
        "observed"
    ].rename(columns={"mean": "observed", "std": "obs_std"})
    df_ffwc_wl_full.index = pd.to_datetime(df_ffwc_wl_full.index)

    # Combine with first DF

    df_ffwc_wl = pd.merge(
        df_ffwc_wl_full[["obs_std"]],
        df_ffwc_wl,
        left_index=True,
        right_index=True,
        how="outer",
    )
    df_ffwc_wl.update(df_ffwc_wl_full, overwrite=False)

    return df_ffwc_wl
```

```python
jamuna_ffwc = read_in_ffwc()
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
da_ra_b = ds_ra["Bahadurabad"]
da_ra_h = ds_ra["Hardinge Bridge"]
da_ra_g = ds_ra["Goalando"]
```

### Goalando

```python
station = "Goalando"
# Interpolate the reforecast
da_rf_g.values = np.sort(da_rf_g.values, axis=0)

da_rf_interp_g = da_rf_g.interp(
    time=pd.date_range(da_rf_g.time.min().values, da_rf_g.time.max().values),
    method="linear",
)
# Get the median
da_rf_med_g = da_rf_interp_g.median(axis=0)
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
for lead_time in da_rf_g["step"].values:
    # Get the forecast at a specific lead time
    forecast = da_rf_med_g.sel(step=lead_time, method="nearest")
    # Shift the time variable of the forecast to match the date of the event
    forecast["time"] = forecast.time.values + lead_time
    # Limit the observations to the forecast time
    model = da_ra_g.reindex(time=forecast.time)
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
df_station_stats.to_csv(
    output_dir / "Goalando GloFAS Model vs Forecast Detection Stats.csv",
    index=False,
)
```

### Bahadurabad

```python
station = "Bahadurabad"
# Interpolate the reforecast
da_rf_b.values = np.sort(da_rf_b.values, axis=0)

da_rf_interp_b = da_rf_b.interp(
    time=pd.date_range(da_rf_b.time.min().values, da_rf_b.time.max().values),
    method="linear",
)
# Get the median
da_rf_med_b = da_rf_interp_b.median(axis=0)
rp_target = 5  # 1 in 5 year
b_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, station),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
df_station_stats = pd.DataFrame()
for lead_time in da_rf_b["step"].values:
    # Get the forecast at a specific lead time
    forecast = da_rf_med_b.sel(step=lead_time, method="nearest")
    # Shift the time variable of the forecast to match the date of the event
    forecast["time"] = forecast.time.values + lead_time
    # Limit the observations to the forecast time
    model = da_ra_b.reindex(time=forecast.time)
    # Get the dates for each
    model_dates = utils.get_dates_list_from_data_array(
        model, b_gf_rp, min_duration=min_duration
    )
    forecast_dates = utils.get_dates_list_from_data_array(
        forecast, b_gf_rp, min_duration=min_duration
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
df_station_stats.to_csv(
    output_dir / "Bahadurabad GloFAS Model vs Forecast Detection Stats.csv",
    index=False,
)
```

### Hardinge Bridge

```python
station = "Hardinge Bridge"
# Interpolate the reforecast
da_rf_h.values = np.sort(da_rf_h.values, axis=0)

da_rf_interp_h = da_rf_h.interp(
    time=pd.date_range(da_rf_h.time.min().values, da_rf_h.time.max().values),
    method="linear",
)
# Get the median
da_rf_med_h = da_rf_interp_h.median(axis=0)
rp_target = 5  # 1 in 5 year
h_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, station),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
df_station_stats = pd.DataFrame()
for lead_time in da_rf_h["step"].values:
    # Get the forecast at a specific lead time
    forecast = da_rf_med_h.sel(step=lead_time, method="nearest")
    # Shift the time variable of the forecast to match the date of the event
    forecast["time"] = forecast.time.values + lead_time
    # Limit the observations to the forecast time
    model = da_ra_h.reindex(time=forecast.time)
    # Get the dates for each
    model_dates = utils.get_dates_list_from_data_array(
        model, h_gf_rp, min_duration=min_duration
    )
    forecast_dates = utils.get_dates_list_from_data_array(
        forecast, h_gf_rp, min_duration=min_duration
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
df_station_stats.to_csv(
    output_dir
    / "Hardinge Bridge GloFAS Model vs Forecast Detection Stats.csv",
    index=False,
)
```

### Comparing Goalando vs sum of Hardinge Bridge and Bahadurabad

```python
df_comb = da_ra_g.to_dataframe().merge(
    da_ra_h.to_dataframe(), left_index=True, right_index=True
)
df_comb["Lagged Hardinge Bridge"] = df_comb["Hardinge Bridge"].shift(1)
df_comb = df_comb.merge(
    da_ra_b.to_dataframe(), left_index=True, right_index=True
)
df_comb["Lagged Bahadurabad"] = df_comb["Bahadurabad"].shift(2)
df_comb["discharge"] = df_comb[
    ["Lagged Hardinge Bridge", "Lagged Bahadurabad"]
].sum(axis=1, skipna=True)
df_comb
```

```python
ds_comb = df_comb[["discharge"]].to_xarray()
```

```python
station = "Hardinge Bridge/Bahadurabad"
rp_target = 5  # 1 in 5 year
comb_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_comb, "discharge"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
comb_gf_rp
```

```python
df_station_stats = pd.DataFrame()
# Get the model
baseline = df_comb["Goalando"]
# Limit the observations to the forecast time
test = df_comb["discharge"]
# Get the dates for each
baseline_dates = utils.get_dates_list_from_data_array(
    baseline, padma_gf_rp, min_duration=min_duration
)
test_dates = utils.get_dates_list_from_data_array(
    test, comb_gf_rp, min_duration=min_duration
)
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=baseline_dates,
    forecasted_event_dates=test_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
df_station_stats = pd.concat(
    (
        df_station_stats,
        pd.DataFrame(
            {
                **{
                    "station": "Goalando vs Hardinge Bridge(-1)+Bahadurabad(-2)"
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
baseline_dates
```

```python
test_dates
```

```python
df_station_stats = pd.DataFrame()
for lead_time in da_rf_g["step"].values:
    # Get the forecast at a specific lead time
    forecast = da_rf_med_g.sel(step=lead_time, method="nearest")
    # Shift the time variable of the forecast to match the date of the event
    forecast["time"] = forecast.time.values + lead_time
    # Limit the observations to the forecast time
    model = ds_comb["discharge"]
    # Get the dates for each
    model_dates = utils.get_dates_list_from_data_array(
        model, comb_gf_rp, min_duration=min_duration
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
                    **{
                        "station": "Goalando vs Hardinge Bridge(-1) + Bahadurabad(-2)",
                        "leadtime": lead_time,
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
station = "Goalando vs Hardinge Bridge(-1day) and Bahadurabad(-2day)"
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

```python
df_station_stats.to_csv(
    output_dir
    / "GloFAS Forecast at Goalando vs Model at " +
    "Hardinge Bridge and Bahadurabad Detection Stats.csv",
    index=False,
)
```

#### Goalando Reanalysis vs Combined Forecast

```python
da_rf_med_hb = da_rf_med_h.shift(step=1) + da_rf_med_b.shift(step=2)
ds_g = df_comb[["Goalando"]].to_xarray()
padma_gf_rp
```

```python
df_station_stats = pd.DataFrame()
for lead_time in da_rf_med_hb["step"].values:
    # Get the forecast at a specific lead time
    forecast = da_rf_med_hb.sel(step=lead_time, method="nearest")
    # Shift the time variable of the forecast to match the date of the event
    forecast["time"] = forecast.time.values + lead_time
    # Limit the observations to the forecast time
    model = ds_g["Goalando"]
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
                    **{
                        "station": "Goalando vs Hardinge Bridge(-1) + Bahadurabad(-2)",
                        "leadtime": lead_time,
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
station = "Goalando vs Hardinge Bridge(-1) + Bahadurabad(-2)"
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

### Comparing Goalando vs Hardinge Bridge with 1 day lag

```python
rp_target = 5  # 1 in 5 year
hb_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, "Hardinge Bridge"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
df_comb = da_ra_g.to_dataframe().merge(
    da_ra_h.to_dataframe(), left_index=True, right_index=True
)
df_comb["Lagged Hardinge Bridge"] = df_comb["Hardinge Bridge"].shift(1)
df_comb
```

```python
df_station_stats = pd.DataFrame()
# Get the model
baseline = df_comb["Goalando"]
# Limit the observations to the forecast time
test = df_comb["Lagged Hardinge Bridge"]
# Get the dates for each
baseline_dates = utils.get_dates_list_from_data_array(
    baseline, padma_gf_rp, min_duration=min_duration
)
test_dates = utils.get_dates_list_from_data_array(
    test, hb_gf_rp, min_duration=min_duration
)
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=baseline_dates,
    forecasted_event_dates=test_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
df_station_stats = pd.concat(
    (
        df_station_stats,
        pd.DataFrame(
            {
                **{"station": "Goalando vs 1-day Lagged Hardinge Bridge"},
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

### Comparing Goalando vs Bahadurabad with 2 day lag

```python
rp_target = 5  # 1 in 5 year
b_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, "Bahadurabad"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
df_comb = da_ra_g.to_dataframe().merge(
    da_ra_b.to_dataframe(), left_index=True, right_index=True
)
df_comb["Lagged Bahadurabad"] = df_comb["Bahadurabad"].shift(2)
df_comb
```

```python
df_station_stats = pd.DataFrame()
# Get the model
baseline = df_comb["Goalando"]
# Limit the observations to the forecast time
test = df_comb["Lagged Bahadurabad"]
# Get the dates for each
baseline_dates = utils.get_dates_list_from_data_array(
    baseline, padma_gf_rp, min_duration=min_duration
)
test_dates = utils.get_dates_list_from_data_array(
    test, b_gf_rp, min_duration=min_duration
)
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=baseline_dates,
    forecasted_event_dates=test_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
df_station_stats = pd.concat(
    (
        df_station_stats,
        pd.DataFrame(
            {
                **{"station": "Goalando vs 2-day Lagged Bahadurabad"},
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

## Interaction of events in Padma(Goalando) and Jamuna(Bahadurabad)

```python
da_ra_g = ds_ra["Goalando"]
da_ra_b = ds_ra["Bahadurabad"]
```

### The GloFAS model 1-in-5 year RP

```python
rp_target = 5  # 1 in 5 year
padma_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, "Goalando"),
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
glofas_comb = pd.merge(
    da_ra_b.to_dataframe(),
    da_ra_g.to_dataframe(),
    how="outer",
    left_on="time",
    right_index=True,
)
```

```python
df_station_stats = pd.DataFrame()
# Get the model data
padma_glofas = glofas_comb["Goalando"]
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

### The FFWC events with danger level specified

- Jamuna: 19.5 + 0.85
- Padma: 8.66 + 1

```python
jamuna_dl = 19.5 + 0.85
padma_dl = 8.66 + 1
```

```python
ffwc_comb = pd.merge(
    (padma_ffwc.set_index("DATE/TIME")[["AVE_WL(m)"]]),
    jamuna_ffwc,
    how="outer",
    left_index=True,
    right_index=True,
)
```

```python
# Get the model data
padma_ffwc_d = ffwc_comb["AVE_WL(m)"]
jamuna_ffwc_d = ffwc_comb["observed"]
# Get the dates for each
padma_dates = utils.get_dates_list_from_data_array(
    padma_ffwc_d, padma_dl, min_duration=min_duration
)
jamuna_dates = utils.get_dates_list_from_data_array(
    jamuna_ffwc_d, jamuna_dl, min_duration=min_duration
)
```

```python
padma_dates
```

```python
jamuna_dates
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
station = "Goalando"
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

### FFWC

```python
da_ffwc = padma_ffwc[["DATE/TIME", "AVE_WL(m)"]]
da_ffwc["time"] = pd.DatetimeIndex(da_ffwc["DATE/TIME"]).year
da_ffwc = (
    da_ffwc.groupby("time").max().sort_values(by="AVE_WL(m)", ascending=False)
)
```

```python
rp_target = 5  # 1 in 5 year
rp_value = utils.get_return_period_function_analytical(
    df_rp=da_ffwc,
    rp_var="AVE_WL(m)",
    show_plots=False,
)(rp_target)
rp_value
```

```python
rp_target = 10  # 1 in 10 year
rp_value = utils.get_return_period_function_analytical(
    df_rp=da_ffwc,
    rp_var="AVE_WL(m)",
    show_plots=False,
)(rp_target)
rp_value
```
