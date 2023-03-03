import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ochanticipy import (
    CodAB,
    GeoBoundingBox,
    GlofasReanalysis,
    GlofasReforecast,
)
from scipy.interpolate import interp1d
from scipy.stats import genextreme as gev

from src import constants

geo_bounding_box = GeoBoundingBox.from_shape(
    CodAB(country_config=constants.country_config).load(admin_level=0)
)


def load_glofas_reforecast():
    return GlofasReforecast(
        country_config=constants.country_config,
        geo_bounding_box=geo_bounding_box,
        leadtime_max=constants.leadtime_max,
    ).load()


def load_glofas_reanalysis():
    return GlofasReanalysis(
        country_config=constants.country_config,
        geo_bounding_box=geo_bounding_box,
        end_date=constants.reanalysis_end_date,
    ).load()


def get_return_period_df(da_glofas: xr.Dataset, station: str):
    df_rp = (
        da_glofas.to_dataframe()
        .rename(columns={station: "discharge"})
        .resample(rule="A", kind="period")
        .max()
        .sort_values(by="discharge", ascending=False)
    )
    df_rp["year"] = df_rp.index.year
    return df_rp


def get_return_period_function_analytical(
    df_rp: pd.DataFrame,
    rp_var: str,
    show_plots: bool = False,
    plot_title: str = "",
    extend_factor: int = 1,
):
    """
    :param df_rp: DataFrame where the index is the year, and the rp_var
    column contains the maximum value per year
    :param rp_var: The column with the quantity to be evaluated
    :param show_plots: Show the histogram with GEV distribution overlaid
    :param plot_title: The title of the plot
    :param extend_factor: Extend the interpolation range in case you want to
    calculate a relatively high return period
    :return: Interpolated function that gives the quantity for a
    given return period
    """
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    rp_var_values = df_rp[rp_var]
    shape, loc, scale = gev.fit(
        rp_var_values,
        loc=rp_var_values.median(),
        scale=rp_var_values.median() / 2,
    )
    x = np.linspace(
        rp_var_values.min(),
        rp_var_values.max() * extend_factor,
        100 * extend_factor,
    )
    if show_plots:
        fig, ax = plt.subplots()
        ax.hist(rp_var_values, density=True, bins=20)
        ax.plot(x, gev.pdf(x, shape, loc, scale))
        ax.set_title(plot_title)
        plt.show()
    y = gev.cdf(x, shape, loc, scale)
    y = 1 / (1 - y)
    return interp1d(y, x)
