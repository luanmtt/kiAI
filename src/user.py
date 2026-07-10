from src.stat import stats_analysis, apply_mods
from src.eda import plot_edas, reliability_check, correlation_map, plot_label_counts
from src.fetch import get_ids_from_collector, get_beatmaps_bulk
from src.osu_parser import parse_and_feature
from src.download import download_osu_files
from src.dataset import build_dataset
from src.predict import predict_maps
from src.auth import get_token
from src.model import train


from pathlib import Path
import pandas as pd
import json
import os


COLLECTIONS_TEMPLATE = {
    "stream":         0,
    "dt_farm":        0,
    "speed":          0,
    "aim_control":    0,
    "aim_slop":       0,
    "jump":           0,
    "gimmick":        0,
    "consistency":    0,
    "precision":      0,
    "stamina":        0,
    "alt":            0,
    "tech":           0,
    "slider":         0,
    "reading":        0,
    "finger_control": 0,
}


# --------------------------------------------------------------------------------------------------


def check_collections_input() -> dict:
    """
    Reads data/collections_input.json and validates its structure.
    Creates a template if it doesn't exist.
    """
    
    print("kiAI needs to install the json+.osu files to start the training.")
    print("Searching for 'collections_input'...")

    path = Path("data/collections_input.json")

    if not path.exists():
        with open(path, "w") as f:
            json.dump(COLLECTIONS_TEMPLATE, f, indent=4)

        print(f"No collections_input.json found.")
        print(f"Template created at {path}.")
        print("Fill in the osu!collector beatmapset IDs and rerun.")
        return {}

    with open(path) as f:
        collections = json.load(f)

    # validate keys
    missing = [k for k in COLLECTIONS_TEMPLATE if k not in collections]
    if missing:
        print(f"collections_input.json is missing labels: {missing}")
        return {}

    #for k, v in collections.items():
        #print(f"found {k}: {v}")


    filled = {k: v for k, v in collections.items() if v != 0}
    if not filled:
        print("collections_input.json has no collection IDs filled in.")
        return {}
    
    #print(f"Found {len(filled)} collection(s): {list(filled.keys())}")
    return filled


# --------------------------------------------------------------------------------------------------


def get_dataset():

    print("━" * 70)
    print(20 * "━" + " Build dataset: " + 30 * "━" + "\n")
    
    # build dataset
    build_dataset()

    # stat analysis + mod augmentation
    df       = pd.read_csv("data/processed/dataset.csv")
    stats_df = stats_analysis(df)
    df_aug   = apply_mods(df, stats_df)
    df_aug.to_csv("data/processed/augmented.csv", index=False)
        

# --------------------------------------------------------------------------------------------------


def run_train():
    
    print("━" * 70)
    print(20 * "━" + " Train model: " + 30 * "━" + "\n")

    collections = check_collections_input()
    if not collections:
        return
   
    token = get_token()
     
    for category, collection_id in collections.items():
        
        print(f"Fetching category {category} collection {collection_id}...")
        ids = []
        for col_id in collection_id:
            ids = get_ids_from_collector(col_id)
        print(f"{len(ids)} beatmap IDs retrieved.")
        

        print("\n" + "-" * 70 + "\n")


        print("Fetching beatmap metadata from osu!api...")
        beatmaps = get_beatmaps_bulk(ids, token)
        print(f"Retrieved data from {len(beatmaps)} beatmaps.")
        

        print("\n" + "-" * 70 + "\n")


        with open(f"data/raw/{category}_maps.json", "w") as f:
            json.dump(beatmaps, f, indent=2)
        print(f"Saved to /data/raw/{category}_maps.")
    

    # download .osu files from osu!collector retrieved ids
    download_osu_files()
        

    # build dataset
    build_dataset()

    # stat analysis + mod augmentation
    df       = pd.read_csv("data/processed/dataset.csv")
    stats_df = stats_analysis(df)
    df_aug   = apply_mods(df, stats_df)
    df_aug.to_csv("data/processed/augmented.csv", index=False)

    train()


# --------------------------------------------------------------------------------------------------


def run_retrain():
    
    print("━" * 70)
    print(20 * "━" + " Retrain model: " + 30 * "━" + "\n")

    aug  = Path("data/processed/augmented.csv")
    base = Path("data/processed/dataset.csv")

    if aug.exists():
        choice = input("augmented.csv found. Use it? [y/n]: ").strip().lower()
        data_path = str(aug) if choice == "y" else str(base)

    elif base.exists():
        print("No augmented.csv found, using dataset.csv.")
        data_path = str(base)
        
    else:
        print("No dataset found. Run 'train' first.")
        return

    train()

# --------------------------------------------------------------------------------------------------


def run_predict():

    print("━" * 70)
    print(20 * "━" + " Predict submissions: " + 30 * "━" + "\n\n")
    
    model_dir = Path("results/")

    required  = ["model.keras", "scaler.pkl", "label_encoder.pkl"]
    missing   = [f for f in required if not (model_dir / f).exists()]
    
    if missing:
        print(f"Missing model files: {missing}. Run 'train' first.")
        return


    print("Enter .osu file path(s). One per line. Empty line to finish:")
    paths = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        p = Path(line)
        if not p.exists():
            print(f"  File not found: {line}")
        else:
            paths.append(str(p))

    if not paths:
        print("No valid files provided.")
        return

    predict_maps(paths)


# --------------------------------------------------------------------------------------------------


def run_stat_eda(csv: str):

    print("━" * 70)
    print(20 * "━" + " EDAs and Stats: " + 30 * "━" + "\n\n")
    
    if(csv == "aug"):
        df = pd.read_csv("data/processed/dataset.csv")
    else:
        df = pd.read_csv("data/processed/augmented.csv")

    stats_df = stats_analysis(df)
    
    #apply_mods(df, stats_df)

    plot_edas(df,stats_df)
    
    SUBSET= [   
                "aim_slop",
                "dt_farm",
                "precision",
                "stamina",
                "stream",
            ]
    
    CLASSES = [
                "label",
                "kiai_note_ratio",
                "kiai_mean_dist",
                "stream_density", 
                "mean_interval_ms",
              ]

    reliability_check(df, SUBSET, CLASSES)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────
