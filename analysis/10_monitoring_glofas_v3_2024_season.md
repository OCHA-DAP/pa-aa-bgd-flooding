# Reviewing Flooding Event in Bangladesh between GloFAS v3 and v4
Since the update in GloFAS versions, there looks to be a mismatch between the performance for the stations. This notebook looks at Bahadurabad in June-July 2024. 


```python
%load_ext jupyter_black
from pathlib import Path
import cdsapi
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from src import utils
import glob
```


```python
nc_path = Path(os.getenv("AA_DATA_DIR")) / "public/processed/bgd/glofas/"
fore_path = (
    Path(os.getenv("AA_DATA_DIR"))
    / "public/raw/bgd/glofas/cems-glofas-forecast/"
)
v3_files = glob.glob(str(nc_path) + "/bgd_cems-glofas-historical_[0-9]*.nc")
daily_glofas_files = (
    Path(os.getenv("AA_DATA_DIR")) / "public/raw/bgd/glofas/daily_version_3/"
)
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

rp_target = 5

b_rp_v3 = utils.get_return_period_function_analytical(
    df_rp=utils.get_return_period_df(ds_ra_v3, "Bahadurabad"),
    rp_var="discharge",
    show_plots=False,
)(rp_target)
b_rp_v3
```


```python
# Downloading the data
versions = ["operational", "version_3_1"]
months = ["06", "07"]
june_days = ["24", "25", "26", "27", "28", "29", "30"]
july_days = ["01", "02", "03"]

c = cdsapi.Client()


def retrieve_glofas():
    for ver in versions:
        if ver == "operational":
            ver_num = "version_4"
        else:
            ver_num = "version_3"
        for mon in months:
            if mon == "06":
                days = june_days
            else:
                days = july_days
            for day in days:
                c.retrieve(
                    "cems-glofas-forecast",
                    {
                        "variable": "river_discharge_in_the_last_24_hours",
                        "format": "grib",
                        "system_version": f"{ver}",
                        "hydrological_model": "lisflood",
                        "product_type": [
                            "control_forecast",
                            "ensemble_perturbed_forecasts",
                        ],
                        "year": "2024",
                        "month": mon,
                        "day": day,
                        "leadtime_hour": [
                            "24",
                            "48",
                            "72",
                            "96",
                            "120",
                            "144",
                            "168",
                            "192",
                            "216",
                            "240",
                            "264",
                            "288",
                            "312",
                            "336",
                            "360",
                        ],
                        "area": [
                            25.3,
                            89.5,
                            25,
                            89.8,
                        ],
                    },
                    Path(os.getenv("AA_DATA_DIR"))
                    / (
                        "public/raw/bgd/glofas/daily_version_3/"
                        + f"glofas_{ver_num}_{day}_{mon}_2024.grib"
                    ),
                )


retrieve_glofas()
```

Plotting the version 3 and 4 forecasts from 2 July 2024


```python
## Version 3 for 2 July
ds_cf = xr.open_dataset(
    daily_glofas_files / "glofas_version_3_02_07_2024.grib",
    filter_by_keys={"dataType": "cf"},
)
ds_pf = xr.open_dataset(
    daily_glofas_files / "glofas_version_3_02_07_2024.grib",
    filter_by_keys={"dataType": "pf"},
)
ds_v3 = xr.concat([ds_cf, ds_pf], dim="number").median("number")
ds_v3["latitude"] = ds_v3["latitude"].round(2)
ds_v3["longitude"] = ds_v3["longitude"].round(2)
da_v3 = ds_v3.sel(latitude=25.15, longitude=89.65)
## Version 4 for 2 July
ds_cf = xr.open_dataset(
    daily_glofas_files / "glofas_version_4_02_07_2024.grib",
    filter_by_keys={"dataType": "cf"},
)
ds_pf = xr.open_dataset(
    daily_glofas_files / "glofas_version_4_02_07_2024.grib",
    filter_by_keys={"dataType": "pf"},
)
ds_v4 = xr.concat([ds_cf, ds_pf], dim="number").median("number")
ds_v4["latitude"] = ds_v4["latitude"].round(3)
ds_v4["longitude"] = ds_v4["longitude"].round(3)
da_v4 = ds_v4.sel(latitude=25.125, longitude=89.675)

v3_rp = 95000  # 1-in-5 year RP
v4_rp = 92000  # 1-in-5 year RP
start_date = pd.Timestamp("2024-07-02")
# Convert timedelta values to actual dates
dates = start_date + da_v3["step"] - pd.Timedelta(days=1)
# Assign the dates to the 'step' coordinate in the Dataset
da_v3["step"] = dates
da_v4["step"] = dates
# Create a figure for plotting
plt.figure(figsize=(10, 6))
da_v3["dis24"].plot(label=f"Version 3")
da_v4["dis24"].plot(label=f"Version 4")
# Plot the reference point line
plt.axhline(y=v3_rp, color="b", linestyle=":", label="v3 1-in-5 yr RP")
plt.axhline(y=v4_rp, color="r", linestyle="--", label="v4 1-in-5 yr RP")

# Add title and labels
plt.title(
    "GloFAS River Discharge Forecast for Bahadurabad released on 3 July 2024, Over the Next 15 Days"
)
plt.xlabel("Date")
plt.ylabel("River Discharge [m³/s]")
# Add a legend
plt.legend()
# Show the plot
plt.show()
```

Plotting the Version 3 forecasts from the last week of June


```python
# Create a figure for plotting
plt.figure(figsize=(10, 6))

# Loop through each day, read the dataset, and plot the data
for mon in months:
    if mon == "06":
        days = june_days
    else:
        days = july_days
    for day in days:
        start_date = pd.Timestamp(f"2024-{mon}-{day}")
        filename = daily_glofas_files / f"glofas_version_3_{day}_{mon}_2024.grib"
        ds_cf = xr.open_dataset(
            filename, engine="cfgrib", filter_by_keys={"dataType": "cf"}
        )
        ds_pf = xr.open_dataset(
            filename, engine="cfgrib", filter_by_keys={"dataType": "pf"}
        )
        ds = xr.concat([ds_cf, ds_pf], dim="number").median("number")
        ds["latitude"] = ds["latitude"].round(2)
        ds["longitude"] = ds["longitude"].round(2)
        da = ds.sel(latitude=25.15, longitude=89.65)
        # Convert timedelta values to actual dates
        dates = start_date + da["step"] - pd.Timedelta(days=1)
        # Assign the dates to the 'step' coordinate in the Dataset
        da["step"] = dates
        da["dis24"].plot(label=f"{start_date}")

# Plot the reference point line
plt.axhline(y=v3_rp, color="b", linestyle=":", label="1-in-5 yr RP")

# Add title and labels
plt.title(
    "GloFAS Version 3 River Discharge Forecast for Bahadurabad, Over the Next 15 Days"
)
plt.xlabel("Date")
plt.ylabel("River Discharge [m³/s]")
# Add a legend
plt.legend()
# Show the plot
plt.show()
```

Plotting all ensemble members


```python
# Create a figure for plotting
plt.figure(figsize=(10, 6))

# Plot for July 2nd
start_date = pd.Timestamp(f"2024-07-02")
ds_cf = xr.open_dataset(
    daily_glofas_files / "glofas_version_3_02_07_2024.grib",
    engine="cfgrib",
    filter_by_keys={"dataType": "cf"},
)
ds_pf = xr.open_dataset(
    daily_glofas_files / "glofas_version_3_02_07_2024.grib",
    engine="cfgrib",
    filter_by_keys={"dataType": "pf"},
)
ds = xr.concat([ds_cf, ds_pf], dim="number")
ds["latitude"] = ds["latitude"].round(2)
ds["longitude"] = ds["longitude"].round(2)
da = ds.sel(latitude=25.15, longitude=89.65)

# Convert timedelta values to actual dates
dates = start_date + da["step"] - pd.Timedelta(days=1)
# Assign the dates to the 'step' coordinate in the Dataset
da["step"] = dates

# Plot each ensemble member as a line
for i in range(len(ds.number)):
    ds_member = da.isel(number=i)
    ds_member["dis24"].plot(linewidth=1.5)

# Plot the reference point line
plt.axhline(y=v3_rp, color="b", linestyle=":", label="1-in-5 yr RP")

# Add title and labels
plt.title(
    "GloFAS Version 3 River Discharge Forecast for Bahadurabad released on the 2nd of July, 2024, Over the Next 15 Days"
)
plt.xlabel("Date")
plt.ylabel("River Discharge [m³/s]")
# Add a legend
plt.legend()
# Show the plot
plt.show()
```


```python
# Calculate probability of exceeding 97000
above_95000 = da["dis24"] > v3_rp
probability_above_95000 = above_95000.mean(dim="number")
# Create a figure for plotting
plt.figure(figsize=(10, 6))

# Plot the probability as a line
probability_above_95000.plot(label="Probability > 95000", linewidth=2)

# Add a horizontal line for reference
plt.axhline(y=0.5, color="r", linestyle="--", label="50% Threshold")

# Add title and labels
plt.title("Probability of Exceeding 1-in-5 year RP for GloFAS Version 3 Forecast")
plt.xlabel("Date")
plt.ylabel("Probability")
# Add a legend
plt.legend()
# Show the plot
plt.show()
```


```python
pd.DataFrame(
    {
        "Dates": pd.date_range(start_date, periods=15).tolist(),
        "Step": list(range(1, 16)),
        "Probability of Exceeding 95,000 m3/s": probability_above_95000.values,
    }
)
```
