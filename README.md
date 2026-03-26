# Soil Moisture ML Package

A cleaned Python package built from the uploaded source scripts.

## Structure

```text
soil_moisture_ml_package/
├── README.md
└── src/
    ├── __init__.py
    ├── loader.py
    ├── preprocess.py
    ├── models.py
    ├── train.py
    ├── extract.py
    └── pipeline.py
```

## Install
```bash
pip install pandas numpy scikit-learn joblib geopandas rasterstats
```

## Example
```python
from pathlib import Path
from src.pipeline import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    output_dir=Path("./outputs"),
    polygons_path=Path("./data/polygons.shp"),
    raster_dir=Path("./rasters"),
    timeline_csv=Path("./data/timeline.csv"),
    vsm_csv=Path("./data/VSM.csv"),
    precipitation_csv=Path("./data/precipitation.csv"),
    et_csv=Path("./data/et.csv"),
    feature_columns=["mean_value"],
    target_column="VSM",
)
outputs = run_pipeline(cfg)
```
