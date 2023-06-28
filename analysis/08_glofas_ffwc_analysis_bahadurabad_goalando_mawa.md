
# Analysis of Goalando, Bahadurabad and Mawa Stations

- Comparing GloFAS events between Goalando and Mawa
- Comparing FFWC events between Bahadurabad and Mawa

```python
%load_ext jupyter_black
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
output_dir = Path(os.environ["AA_DATA_DIR"]) / "private/exploration/bgd/"
ffwc_data_dir = (
    Path(os.environ["AA_DATA_DIR"]) / "private/exploration/bgd/FFWC_Data"
)
jamuna_ffwc_dir = (
    Path(os.environ["AA_DATA_DIR"]) / "private/exploration/bgd/ffwc"
)
```

```python
# loading GloFAS
ds_ra = utils.load_glofas_reanalysis()
```

```python
da_ra_g = ds_ra["Goalando"]
da_ra_m = ds_ra["Mawa"]
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
# loading FFWC data
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
padma_ffwc = pd.read_excel(
    ffwc_data_dir / "Water LevelSW93.5L13530.xlsx", skiprows=9
)
```

```python
min_duration = 3  # day
days_before_buffer = 5
days_after_buffer = 5
```

## Comparing GloFAS events between Goalando and Mawa

```python
rp_target = 5  # 1 in 5 year
goalando_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, "Goalando"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
mawa_gf_rp = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra, "Mawa"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
```

```python
goalando_gf_rp
```

```python
mawa_gf_rp
```

```python
glofas_comb = pd.merge(
    da_ra_g.to_dataframe(),
    da_ra_m.to_dataframe(),
    how="outer",
    left_on="time",
    right_index=True,
)
```

```python
df_station_stats = pd.DataFrame()
# Get the model data
goalando_glofas = glofas_comb["Goalando"]
mawa_glofas = glofas_comb["Mawa"]
# Get the dates for each
goalando_dates = utils.get_dates_list_from_data_array(
    goalando_glofas, goalando_gf_rp, min_duration=min_duration
)
mawa_dates = utils.get_dates_list_from_data_array(
    mawa_glofas, mawa_gf_rp, min_duration=min_duration
)
```

```python
goalando_dates
```

```python
mawa_dates
```

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=goalando_dates,
    forecasted_event_dates=mawa_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=mawa_dates,
    forecasted_event_dates=goalando_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```

## Comparing FFWC events between Bahadurabad and Mawa

```python
bahadurabad_dl = 19.5
mawa_dl = 6.11

bahadurabad_trigger = bahadurabad_dl + 0.85
mawa_trigger = mawa_dl + 0.5
```

```python
bahadurabad_ffwc = jamuna_ffwc["observed"]
mawa_ffwc = padma_ffwc.set_index("DATE/TIME")["AVE_WL(m)"]
```

```python
# Get the dates for each
bahadurabad_dates = utils.get_dates_list_from_data_array(
    bahadurabad_ffwc, bahadurabad_trigger, min_duration=min_duration
)
mawa_dates = utils.get_dates_list_from_data_array(
    mawa_ffwc, mawa_trigger, min_duration=min_duration
)
```

```python
bahadurabad_dates
```

```python
mawa_dates
```

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=bahadurabad_dates,
    forecasted_event_dates=mawa_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```

```python
# Match the dates to events
detection_stats = utils.get_detection_stats(
    true_event_dates=mawa_dates,
    forecasted_event_dates=bahadurabad_dates,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```
