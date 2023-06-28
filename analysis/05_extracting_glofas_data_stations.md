# Extracting GloFAS data for stations

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
quantile_dict = {
    "25th percentile": 0.25,
    "median": 0.5,
    "75th percentile": 0.75,
}
```

```python
def get_quantile_rf(quant):
    out_df = pd.DataFrame()
    for station in stations["station name"]:
        print("Running: " + station)
        da_rf = ds_rf[station]
        # Interpolate the reforecast
        da_rf.values = np.sort(da_rf.values, axis=0)

        da_rf_interp = da_rf.interp(
            time=pd.date_range(
                da_rf.time.min().values, da_rf.time.max().values
            ),
            method="linear",
        )
        # Get the quantile
        da_rf_med = da_rf_interp.quantile(quantile_dict[quant], dim="number")
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
    out_df.to_csv(
        output_dir
        / str(quant + " GloFAS Reforecast for List of Stations.csv"),
        index=False,
    )
```

```python
get_quantile_rf("25th percentile")
get_quantile_rf("median")
get_quantile_rf("75th percentile")
```
