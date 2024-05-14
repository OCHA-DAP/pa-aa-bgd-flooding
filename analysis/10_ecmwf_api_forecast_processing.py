import os
import logging
import geopandas as gpd
import regionmask
import xarray as xr
from ecmwfapi import ECMWFDataServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ecmwf_data_download(start_date, end_date, basin_shp, output_dir, download_type):
    """
    Download ECMWF forecast data and process it based on a given basin shapefile.

    Args:
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        basin_shp (str): Path to the basin shapefile.
        output_dir (str): Directory to save the downloaded data.
        download_type (str): Type of data to download: "hres", "ens", or "both".

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    server = ECMWFDataServer()

    def retrieve_data(out_file, request_params):
        try:
            if os.path.exists(out_file):
                logger.info(f"File: {out_file} exists")
            else:
                server.retrieve(request_params)
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")

    def process_and_save(out_file, ds):
        mask = regionmask.mask_geopandas(basin_shp, ds.longitude.values, ds.latitude.values)
        mask = mask.rename({"lon": "longitude", "lat": "latitude"})
        basin_ds = ds.where(mask == 0, drop=True)
        logger.info(f"Writing {out_file}")
        basin_ds.to_netcdf(out_file)

    def cleanup_files(*files):
        for file in files:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed file: {file}")

    if download_type in ["hres", "both"]:
        out_file_hres = os.path.join(output_dir, f"bgd_hres_forecast_{end_date}.nc")
        retrieve_data(out_file_hres, {
            "class": "ti",
            "dataset": "tigge",
            "date": f"{start_date}/to/{end_date}",
            "grid": "0.1/0.1",
            "expver": "prod",
            "levtype": "sfc",
            "origin": "ecmf",
            "param": "228228",
            "step": "/".join(map(str, range(0, 361, 6))),
            "time": "00:00:00/12:00:00",
            "type": "fc",
            "area": "32/73/21/98",
            "target": out_file_hres
        })

        if os.path.exists(out_file_hres):
            ds_hres = xr.open_dataset(out_file_hres, engine="cfgrib")
            process_and_save(out_file_hres, ds_hres)
            ds_hres.close()
            cleanup_files(out_file_hres)

    if download_type in ["ens", "both"]:
        out_file_cf = os.path.join(output_dir, f"bgd_ens_forecast_cf_{end_date}.nc")
        out_file_pf = os.path.join(output_dir, f"bgd_ens_forecast_pf_{end_date}.nc")
        out_file_ens = os.path.join(output_dir, f"bgd_ens_forecast_{end_date}.nc")

        retrieve_data(out_file_cf, {
            "class": "ti",
            "dataset": "tigge",
            "date": f"{start_date}/to/{end_date}",
            "grid": "0.1/0.1",
            "expver": "prod",
            "levtype": "sfc",
            "origin": "ecmf",
            "param": "228228",
            "step": "/".join(map(str, range(0, 361, 6))),
            "time": "00:00:00/12:00:00",
            "type": "cf",
            "area": "32/73/21/98",
            "target": out_file_cf
        })

        retrieve_data(out_file_pf, {
            "class": "ti",
            "dataset": "tigge",
            "date": f"{start_date}/to/{end_date}",
            "grid": "0.1/0.1",
            "expver": "prod",
            "levtype": "sfc",
            "origin": "ecmf",
            "param": "228228",
            "number": "/".join(map(str, range(1, 51))),
            "step": "/".join(map(str, range(0, 361, 6))),
            "time": "00:00:00/12:00:00",
            "type": "pf",
            "area": "32/73/21/98",
            "target": out_file_pf
        })

        if os.path.exists(out_file_cf) and os.path.exists(out_file_pf):
            ds_cf = xr.open_dataset(out_file_cf, engine="cfgrib")
            ds_pf = xr.open_dataset(out_file_pf, engine="cfgrib")
            ds = xr.concat([ds_cf, ds_pf], dim="number").median(dim="number")
            ds_cf.close()
            ds_pf.close()
            cleanup_files(out_file_cf, out_file_pf)
            process_and_save(out_file_ens, ds)

if __name__ == "__main__":
    start_date = "2024-05-01"
    end_date = "2024-05-02"
    basin_shp = gpd.read_file("shpfiles/gbm_shp_merged.gpkg")
    output_dir = "rain_data"
    
    # Specify the type of data to download: "hres", "ens", or "both".
    download_type = "ens"  # Change this to "hres", "ens", or "both" as needed.

    ecmwf_data_download(start_date, end_date, basin_shp, output_dir, download_type)
