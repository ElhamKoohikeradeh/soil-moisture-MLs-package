from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import pandas as pd

from .extract import extract_time_series, list_tiffs, load_polygons, save_extracted_table
from .loader import merge_field_environment, read_csv
from .preprocess import add_vsm_from_potential, prepare_ml_table, split_xy
from .train import cross_validate_models, save_training_outputs, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    output_dir: Path = Path("./outputs")
    polygons_path: Optional[Path] = None
    raster_dir: Optional[Path] = None
    timeline_csv: Optional[Path] = None
    vsm_csv: Optional[Path] = None
    precipitation_csv: Optional[Path] = None
    et_csv: Optional[Path] = None
    feature_columns: list[str] = field(default_factory=lambda: ["mean_value"])
    target_column: str = "VSM"
    test_size: float = 0.2
    random_state: int = 42
    run_cross_validation: bool = True
    run_extraction: bool = True
    run_merge: bool = True
    run_van_genuchten: bool = False
    potential_columns: list[str] = field(default_factory=list)
    remove_outliers: bool = False

def ensure_dirs(cfg: PipelineConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "tables").mkdir(exist_ok=True)
    (cfg.output_dir / "models").mkdir(exist_ok=True)

def run_extraction_stage(cfg: PipelineConfig):
    if not (cfg.polygons_path and cfg.raster_dir):
        logger.info("Extraction skipped: polygons_path or raster_dir not provided.")
        return None
    polygons = load_polygons(cfg.polygons_path)
    tiffs = list_tiffs(cfg.raster_dir)
    timeline = read_csv(cfg.timeline_csv) if cfg.timeline_csv and Path(cfg.timeline_csv).exists() else None
    extracted = extract_time_series(tiffs, polygons, timeline)
    save_extracted_table(extracted, cfg.output_dir / "tables" / "extracted_time_series.csv")
    return extracted

def run_merge_stage(cfg: PipelineConfig, extracted_df):
    vsm_df = read_csv(cfg.vsm_csv) if cfg.vsm_csv else None
    precipitation_df = read_csv(cfg.precipitation_csv) if cfg.precipitation_csv else None
    et_df = read_csv(cfg.et_csv) if cfg.et_csv else None
    merged = merge_field_environment(vsm_df=vsm_df, precipitation_df=precipitation_df, et_df=et_df, extracted_df=extracted_df)
    if cfg.run_van_genuchten and cfg.potential_columns:
        merged = add_vsm_from_potential(merged, cfg.potential_columns)
    merged.to_csv(cfg.output_dir / "tables" / "merged_data.csv", index=False)
    return merged

def run_ml_stage(cfg: PipelineConfig, merged_df: pd.DataFrame):
    ml_df = prepare_ml_table(merged_df, feature_columns=cfg.feature_columns, target_column=cfg.target_column, remove_outliers=cfg.remove_outliers)
    X_train, X_test, y_train, y_test = split_xy(ml_df, feature_columns=cfg.feature_columns, target_column=cfg.target_column, test_size=cfg.test_size, random_state=cfg.random_state)
    results, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test, random_state=cfg.random_state)
    save_training_outputs(cfg.output_dir, results, fitted_models)
    if cfg.run_cross_validation:
        X_full = ml_df[[c for c in cfg.feature_columns if c in ml_df.columns]]
        y_full = ml_df[cfg.target_column]
        cv_df = cross_validate_models(X_full, y_full, random_state=cfg.random_state)
        cv_df.to_csv(cfg.output_dir / "CV_metrics.csv", index=False)
        results["cv_metrics"] = cv_df
    return results

def run_pipeline(cfg: PipelineConfig):
    ensure_dirs(cfg)
    extracted_df = run_extraction_stage(cfg) if cfg.run_extraction else None
    merged_df = run_merge_stage(cfg, extracted_df) if cfg.run_merge else None
    outputs = {"extracted": extracted_df, "merged": merged_df}
    if merged_df is not None and not merged_df.empty:
        outputs["ml"] = run_ml_stage(cfg, merged_df)
    return outputs

if __name__ == "__main__":
    cfg = PipelineConfig()
    outputs = run_pipeline(cfg)
    print("Finished.")
    print(outputs.keys())
