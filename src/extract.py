from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from rasterstats import zonal_stats
except Exception:
    zonal_stats = None

def load_polygons(path):
    if gpd is None:
        raise ImportError("geopandas is required to load polygons.")
    gdf = gpd.read_file(path)
    if "study_area" not in gdf.columns:
        gdf = gdf.copy()
        gdf["study_area"] = [f"polygon_{i+1}" for i in range(len(gdf))]
    return gdf

def list_tiffs(raster_dir):
    raster_dir = Path(raster_dir)
    return sorted(raster_dir.glob("*.tif")) + sorted(raster_dir.glob("*.tiff"))

def extract_time_series(tiff_files, polygons, timeline=None):
    if zonal_stats is None:
        raise ImportError("rasterstats is required for zonal extraction.")
    rows = []
    for tiff in tiff_files:
        tiff = Path(tiff)
        stats = zonal_stats(polygons, str(tiff), stats="mean", geojson_out=False, nodata=None)
        for idx, stat in enumerate(stats):
            row = {"raster": tiff.name, "mean_value": stat.get("mean")}
            row["study_area"] = polygons.iloc[idx]["study_area"] if "study_area" in polygons.columns else f"polygon_{idx+1}"
            if timeline is not None and "Date" in timeline.columns and idx < len(timeline):
                row["Date"] = pd.to_datetime(timeline.iloc[idx]["Date"], errors="coerce")
            rows.append(row)
    df = pd.DataFrame(rows)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def save_extracted_table(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
