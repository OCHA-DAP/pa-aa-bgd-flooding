from datetime import date

from ochanticipy import CodAB, GeoBoundingBox, create_country_config

country_config = create_country_config(iso3="bgd")
leadtime_max = 15

codab = CodAB(country_config=country_config)
admin0 = codab.load(admin_level=0)
geo_bounding_box = GeoBoundingBox.from_shape(admin0)

reanalysis_end_date = date(2022, 12, 31)
