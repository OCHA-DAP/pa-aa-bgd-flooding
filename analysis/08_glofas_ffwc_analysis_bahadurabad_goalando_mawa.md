
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
# loading FFWC data
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
