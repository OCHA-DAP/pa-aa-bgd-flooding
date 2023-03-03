# GloFAS forecast

Compare GloFAS reforecast to model

```python
%load_ext jupyter_black
```

```python
import numpy as np
from src import utils
import pandas as pd

from importlib import reload

reload(utils)
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

```
