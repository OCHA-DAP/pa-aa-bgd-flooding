import os
from datetime import date
from pathlib import Path

from ochanticipy import CodAB, GeoBoundingBox, create_custom_country_config

country_config = create_custom_country_config(
    Path(__file__).parent.resolve() / "bgd.yaml"
)

leadtime_max = 15

codab = CodAB(country_config=country_config)
admin0 = codab.load(admin_level=0)
geo_bounding_box = GeoBoundingBox.from_shape(admin0)

reanalysis_end_date = date(2022, 12, 31)

jamuna_ffwc_dir = (
    Path(os.environ["OAP_DATA_DIR"]) / "private/exploration/bgd/ffwc"
)
