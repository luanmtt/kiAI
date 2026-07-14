from src.stat import build_expectations, are_mods_needed, MOD_FEATURES
from src.osu_parser import parse_and_feature
from src.mod import to_dt, to_hr

from pathlib import Path
import pandas as pd
import numpy as np
import pickle


MODEL_DIR = Path("data/processed")
FEATURES  = [
    #"bpm", 
    "star_rating",
    "ar", 
    "cs", 
    "od", 

    "base_bpm", 
    "slider_ratio", 

    "mean_distance", 
    "std_distance",
    "stream_density", 
    
    "kiai_section_count", 
    "mean_kiai_dist",
    "kiai_note_ratio", 
    
    "interval_cv", 
    "interval_variance_log",
    
    "notes_per_second", 
    "mean_velocity", 

    "rhythm_complexity",
    "ar_od_ratio"

]

# --------------------------------------------------------------------------------------------------


def load_artifacts(run_dir: Path | None = None):
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras

    model_dir = run_dir if run_dir else MODEL_DIR
    model = keras.models.load_model(model_dir / "model.keras")
    scaler = pickle.load(open(model_dir / "scaler.pkl", "rb"))
    le     = pickle.load(open(model_dir / "le.pkl", "rb"))
    return model, scaler, le


# --------------------------------------------------------------------------------------------------


def predict_maps(osu_paths: list[str], run_dir: Path | None = None):
    model, scaler, le = load_artifacts(run_dir=run_dir)
    stats_df     = pd.read_csv("data/analysis/label_feature_stats.csv")
    expectations = build_expectations(stats_df, MOD_FEATURES["dt"] + MOD_FEATURES["hr"])

    results = []

    for path in osu_paths:
        parsed = parse_and_feature(path)
        row    = pd.Series(parsed)

        # apply mods if needed
        mod = are_mods_needed(row, expectations)

        if mod in ("dt", "both"):
            row["base_bpm"]         *= 1.5
            row["bpm"]              = row.get("bpm", row["base_bpm"])
            row["mean_interval_ms"] /= 1.5
            row["notes_per_second"] *= 1.5
            row["mean_velocity"]    *= 1.5


        if mod in ("hr", "both"):
            row["cs"]    = min(row["cs"]    * 1.3, 10)
            row["ar"]    = min(row["ar"]    * 1.4, 10)
            row["drain"] = min((row.get("drain") or 5) * 1.4, 10)
            row["od"]    = min(row["od"]    * 1.4, 10)



        # build feature vector
        X = np.array([[row.get(f, 0) for f in FEATURES]])
        X = scaler.transform(X)


        probs      = model.predict(X, verbose=0)[0]
        top3_idx   = np.argsort(probs)[::-1][:3]
        top3       = [(le.classes_[i], round(float(probs[i]) * 100, 1)) for i in top3_idx]

            
        print(f"\n{Path(path).name}")
        print(f"  mod applied : {mod or 'none'}")
        for label, pct in top3:
            print(f"  {label:<20} {pct:.1f}%")

        results.append({"file": path, "mod": mod, "prediction": top3[0][0], "top3": top3})

    return results


# --------------------------------------------------------------------------------------------------
