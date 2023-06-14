from datetime import date
from pathlib import Path
import geopandas as gpd
import os
from ochanticipy import CodAB, GeoBoundingBox, create_custom_country_config

country_config = create_custom_country_config(
    "C:/Users/pauni/Desktop/Work/OCHA/GitHub/pa-aa-bgd-flooding/src/bgd.yaml"
)

leadtime_max = 15

# codab = CodAB(country_config=country_config)
admin0 = gpd.read_file(
    "zip://G:/Shared drives/Predictive Analytics/CERF Anticipatory Action/General - All AA projects/Data/public/raw/bgd/cod_ab/bgd_cod_ab.shp.zip/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm0_bbs_20201113.shp"
)
geo_bounding_box = GeoBoundingBox.from_shape(admin0)

reanalysis_end_date = date(2022, 12, 31)
