from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

def read_csv(path: str | Path, date_columns: Iterable[str] = ("Date",)) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def read_excel_all_sheets(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)
    xls = pd.ExcelFile(path)
    out = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        out[sheet] = df
    return out

def merge_on_keys(dfs: list[pd.DataFrame], keys=("Date",), how="outer") -> pd.DataFrame:
    non_empty = [df.copy() for df in dfs if df is not None and not df.empty]
    if not non_empty:
        raise ValueError("No input DataFrames available for merging.")
    merged = non_empty[0]
    for nxt in non_empty[1:]:
        common = [k for k in keys if k in merged.columns and k in nxt.columns]
        if not common:
            raise ValueError(f"No common merge keys found. Requested keys: {keys}")
        merged = pd.merge(merged, nxt, on=common, how=how)
    return merged

def merge_field_environment(
    vsm_df: Optional[pd.DataFrame] = None,
    precipitation_df: Optional[pd.DataFrame] = None,
    et_df: Optional[pd.DataFrame] = None,
    extracted_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    dfs = [df for df in [vsm_df, precipitation_df, et_df, extracted_df] if df is not None]
    if not dfs:
        raise ValueError("At least one DataFrame must be supplied.")
    keys = ("Date", "study_area") if all("study_area" in df.columns for df in dfs) else ("Date",)
    return merge_on_keys(dfs, keys=keys, how="outer")
