import logging

from ochanticipy import (
    CodAB,
    GeoBoundingBox,
    GlofasForecast,
    GlofasReanalysis,
    GlofasReforecast,
)

from src import constants

logging.basicConfig(level=logging.INFO)


codab = CodAB(country_config=constants.country_config)
codab.download()

admin0 = codab.load(admin_level=0)
geo_bounding_box = GeoBoundingBox.from_shape(admin0)

glofas_reforecast = GlofasReforecast(
    country_config=constants.country_config,
    geo_bounding_box=geo_bounding_box,
    leadtime_max=constants.leadtime_max,
)
glofas_reforecast.download()
glofas_reforecast.process()

glofas_reanalysis = GlofasReanalysis(
    country_config=constants.country_config,
    geo_bounding_box=geo_bounding_box,
    end_date=constants.reanalysis_end_date,
)
glofas_reanalysis.download()
glofas_reanalysis.process()

glofas_forecast = GlofasForecast(
    country_config=constants.country_config,
    geo_bounding_box=geo_bounding_box,
    leadtime_max=constants.leadtime_max,
)
glofas_forecast.download()
glofas_forecast.process()
