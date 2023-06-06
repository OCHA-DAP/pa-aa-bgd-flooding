---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: bgdenv
    language: python
    name: python3
---

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

os.chdir("..")
from src import utils
```

```python
output_dir = Path(os.getenv("AA_DATA_DIR")) / "public/exploration/bgd/glofas/"
```

```python
# This is to avoid error with dask
from dask.distributed import Client

c = Client(n_workers=os.cpu_count() - 2, threads_per_worker=1)
```

```python
url = f"https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1yzAmZY0JF5veZ0G1fQCoXj0gfUvVT3cMCyI5dNzciRiKO2nD2f27f438IZwb_w/pub?gid=950554185&single=true&output=csv"
stations = pd.read_csv(url)
stations["station name"] = stations["River"] + " " + stations["Station"]
stations
```

```python
# Read in reforecast and reanalysis
ds_rf = utils.load_glofas_reforecast()
```

```python
out_df = pd.DataFrame()

for station in stations["station name"]:
    print("Running: " + station)
    da_rf = ds_rf[station]
    # Interpolate the reforecast
    da_rf.values = np.sort(da_rf.values, axis=0)

    da_rf_interp = da_rf.interp(
        time=pd.date_range(da_rf.time.min().values, da_rf.time.max().values),
        method="linear",
    )
    # Get the median
    da_rf_med = da_rf_interp.median(axis=0)
    station_df = pd.DataFrame()
    for tm in da_rf_med["time"].values:
        intern_df = pd.DataFrame()
        intern_df["leadtime"] = (
            (da_rf_med["step"] / (36 * 24 * 100000000000)).values
        ).astype(int)
        intern_df["date"] = (pd.to_datetime(str(tm))).strftime("%Y-%m-%d")
        intern_df[station] = da_rf_med.sel(time=tm).values
        intern_df = intern_df[["date", "leadtime", station]]
        station_df = pd.concat([station_df, intern_df], axis=0)
    if len(out_df) == 0:
        out_df = station_df
    else:
        out_df = out_df.merge(station_df, on=["date", "leadtime"])
```

```python
out_df
```

```python
out_df.to_csv(
    output_dir / "GloFAS Reforecast for List of Stations.csv", index=False
)
```