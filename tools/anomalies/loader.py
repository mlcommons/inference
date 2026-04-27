import pandas as pd
from config import ACCELERATOR_NAME_MAPPINGS


def standardize_accelerator_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    for pattern, canonical in ACCELERATOR_NAME_MAPPINGS:
        if pattern.lower() in name.lower():
            return canonical
    return name


def load_historical_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["Model MLC", "Scenario", "Result", "accelerator_model_name", "Total Accelerators"]
    df = df[required].dropna()

    df["accelerator_std"] = df["accelerator_model_name"].apply(standardize_accelerator_name)
    df["Result"] = pd.to_numeric(df["Result"], errors="coerce")
    df["Total Accelerators"] = pd.to_numeric(df["Total Accelerators"], errors="coerce")
    df = df.dropna(subset=["Result", "Total Accelerators"])
    df = df[df["Total Accelerators"] > 0]

    df["normalized_result"] = df["Result"] / df["Total Accelerators"]
    return df
