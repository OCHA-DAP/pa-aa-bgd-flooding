# Comparison of GloFAS v3 and v4 for BGD

- Comparing bias between the two
- Comparing events between the two for Goalondo and Bahadurabad.


```python
%load_ext jupyter_black
from pathlib import Path
import os
import datetime

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

from src import utils
```


```python
import xarray as xr
import glob
```


```python
ds_ra_v4 = utils.load_glofas_reanalysis()
```


```python
nc_path = Path(os.getenv("AA_DATA_DIR")) / "public/processed/bgd/glofas/"
fore_path = (
    Path(os.getenv("AA_DATA_DIR"))
    / "public/raw/bgd/glofas/cems-glofas-forecast/"
)
```


```python
v3_files = glob.glob(str(nc_path) + "/bgd_cems-glofas-historical_[0-9]*.nc")
```


```python
ds_ra_v3 = xr.DataArray()
v3_files.sort()
for i in v3_files:
    file = xr.open_dataset(i)
    if i == v3_files[0]:
        ds_ra_v3 = file
    else:
        ds_ra_v3 = xr.concat([ds_ra_v3, file], dim="time")
```


```python
rp_target = 5
```


```python
g_rp_v4 = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra_v4, "Goalando"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
g_rp_v4
```


```python
g_rp_v3 = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra_v3, "Goalando"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
g_rp_v3
```


```python
b_rp_v4 = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra_v4, "Bahadurabad"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
b_rp_v4
```


```python
b_rp_v3 = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra_v3, "Bahadurabad"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
b_rp_v3
```


```python
da_ra_g_v4 = ds_ra_v4["Goalando"]
da_ra_g_v3 = ds_ra_v3["Goalando"]
da_ra_b_v4 = ds_ra_v4["Bahadurabad"]
da_ra_b_v3 = ds_ra_v3["Bahadurabad"]
```


```python
goalando_diff = da_ra_g_v3 - da_ra_g_v4
goalando_bias = (goalando_diff.values).mean()
goalando_rmse = np.sqrt((np.square(goalando_diff.values)).mean())
```


```python
print(goalando_bias)
print(goalando_rmse)
```


```python
bahadurabad_diff = da_ra_b_v3 - da_ra_b_v4
bahadurabad_bias = (bahadurabad_diff.values).mean()
bahadurabad_rmse = np.sqrt((np.square(bahadurabad_diff.values)).mean())
```


```python
print(bahadurabad_bias)
print(bahadurabad_rmse)
```


```python
goalando_diff.plot()
plt.title("v3 - v4 for Goalando")
```


```python
bahadurabad_diff.plot()
plt.title("v3 - v4 for Bahadurabad")
```


```python
min_duration = 3
days_before_buffer = 5
days_after_buffer = 5
```


```python
goalando_dates_v3 = utils.get_dates_list_from_data_array(
    da_ra_g_v3, g_rp_v3, min_duration=min_duration
)
goalando_dates_v3
```


```python
goalando_dates_v4 = utils.get_dates_list_from_data_array(
    da_ra_g_v4, g_rp_v4, min_duration=min_duration
)
goalando_dates_v4
```


```python
detection_stats = utils.get_detection_stats(
    true_event_dates=goalando_dates_v3,
    forecasted_event_dates=goalando_dates_v4,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```


```python
bahadurabad_dates_v3 = utils.get_dates_list_from_data_array(
    da_ra_b_v3, b_rp_v3, min_duration=min_duration
)
bahadurabad_dates_v3
```


```python
bahadurabad_dates_v4 = utils.get_dates_list_from_data_array(
    da_ra_b_v4, b_rp_v4, min_duration=min_duration
)
bahadurabad_dates_v4
```


```python
detection_stats = utils.get_detection_stats(
    true_event_dates=bahadurabad_dates_v3,
    forecasted_event_dates=bahadurabad_dates_v4,
    days_before_buffer=days_before_buffer,
    days_after_buffer=days_after_buffer,
)
detection_stats
```


```python
fig, ax = plt.subplots()
da_ra_g_v3.plot(color="red")
da_ra_g_v4.plot(color="blue")
plt.axhline(y=g_rp_v3, color="r", linestyle="-", label="v3 RP")
plt.axhline(y=g_rp_v4, color="b", linestyle="-", label="v4 RP")
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
ax.set_title("Goalando v3 and v4 GloFAS")
```


```python
fig, ax = plt.subplots()
da_ra_b_v3.plot(color="red")
da_ra_b_v4.plot(color="blue")
plt.axhline(y=b_rp_v3, color="r", linestyle="-", label="v3 RP")
plt.axhline(y=b_rp_v4, color="b", linestyle="-", label="v4 RP")
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
ax.set_title("Bahadurabad v3 and v4 GloFAS")
```

Adding analysis on the forecast to test if the event in Aug/Sept 2023 would be caught by either version.


```python
v3_fore_files = glob.glob(str(fore_path) + "/bgd_cems-glofas-forecast_v3*")
v4_fore_files = glob.glob(str(fore_path) + "/bgd_cems-glofas-forecast_v4*")
```


```python
ds_rf_v3 = xr.DataArray()
v3_fore_files.sort()
for i in v3_fore_files:
    file = xr.open_dataset(
        i, engine="cfgrib", filter_by_keys={"dataType": "cf"}
    )
    if i == v3_fore_files[0]:
        ds_rf_v3 = file
    else:
        ds_rf_v3 = xr.concat([ds_rf_v3, file], dim="time")
```


```python
ds_rf_v4 = xr.DataArray()
v4_fore_files.sort()
for i in v4_fore_files:
    file = xr.open_dataset(
        i, engine="cfgrib", filter_by_keys={"dataType": "cf"}
    )
    if i == v4_fore_files[0]:
        ds_rf_v4 = file
    else:
        ds_rf_v4 = xr.concat([ds_rf_v4, file], dim="time")
```


```python
test_file = xr.open_dataset(
    i, engine="cfgrib", filter_by_keys={"dataType": "pf"}
)
```


```python
# bahadurabad station
baha_v3 = ds_rf_v3.sel(latitude=25.15, longitude=89.65, method="nearest")
baha_v4 = ds_rf_v4.sel(latitude=25.125, longitude=89.675, method="nearest")
```


```python
# check if discharge was above 1-in-5 year RP
(baha_v3["dis24"].values > b_rp_v3).any()
```


```python
(baha_v3["dis24"].values).max()
```


```python
# check if discharge was above 1-in-5 year RP
(baha_v4["dis24"].values > b_rp_v4).any()
```


```python
(baha_v4["dis24"].values).max()
```

So the event in Aug 2023 was not caught by any of the versions.
