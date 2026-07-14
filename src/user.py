from src.stat import stats_analysis, apply_mods
from src.eda import plot_edas, reliability_check, correlation_map, plot_label_counts
from src.fetch import get_ids_from_collector, get_beatmaps_bulk
from src.osu_parser import parse_and_feature
from src.download import download_osu_files
from src.dataset import build_dataset
from src.predict import predict_maps
from src.auth import get_token
from src.model import train
from utils.embellish import stylePrint, colorPrint, sep, sepComment


from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import os


# --------------------------------------------------------------------------------------------------
# cores do CLI:


S = "#ad97bd"
O = "#c8b3c4"


def div(text: str = "") -> str:
    """Header de seção com 50 chars de largura."""
    if text:
        return colorPrint(sepComment(20, f" {text} "), S)
    return colorPrint(sep(50), S)


# --------------------------------------------------------------------------------------------------
# templates para os dois branches de label:


TYPE_TEMPLATE = {
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


TOURNEY_TEMPLATE = {
    "nm_1":   0,
    "nm_2":   0,
    "hr_1":   0,
    "hr_2":   0,
    "dt_1":   0,
    "dt_2":   0,
    "hd_1":   0,
    "hd_2":   0,
    "fm_1":   0,
    "fm_2":   0,
    "tb":     0,
}


TEMPLATES = {
    "type":    TYPE_TEMPLATE,
    "tourney":  TOURNEY_TEMPLATE,
}


# --------------------------------------------------------------------------------------------------


def ask_label_type() -> str | None:
    """Pergunta ao usuário se quer treinar para 'type' ou 'tourney'."""
    print("  Which label branch?")
    print(f"    {stylePrint('type', O, bold=True)}     — map classification (stream, dt_farm, tech...)")
    print(f"    {stylePrint('tourney', O, bold=True)}  — tournament slots (nm_1, hr_1, dt_2...)")

    choice = input("\n    > ").strip().lower()
    print()
    if choice in ("type", "tourney"):
        return choice

    print("  Invalid choice.")
    return None


def make_run_dir(label_type: str) -> Path:
    """Cria o diretório de saída com timestamp: outputs/{label_type}_DD-MM_HHhMM/"""
    now = datetime.now()
    ts = f"{now.day:02d}-{now.month:02d}_{now.hour:02d}h{now.minute:02d}"
    run_dir = Path("outputs") / f"{label_type}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------------------------------


def check_collections_input(label_type: str) -> dict:
    """Lê o JSON de collections correto para o label_type escolhido."""

    template = TEMPLATES[label_type]
    filename = f"collections_{label_type}.json"
    path = Path("data") / filename

    print(f"Searching for '{filename}'...")

    if not path.exists():
        with open(path, "w") as f:
            json.dump(template, f, indent=4)

        print(f"No {filename} found.")
        print(f"Template created at {path}.")
        print("Fill in the osu!collector collection IDs and rerun.")
        return {}

    with open(path) as f:
        collections = json.load(f)

    # validate keys
    missing = [k for k in template if k not in collections]
    if missing:
        print(f"{filename} is missing labels: {missing}")
        return {}

    filled = {k: v for k, v in collections.items() if v != 0}
    if not filled:
        print(f"{filename} has no collection IDs filled in.")
        return {}

    return filled


# --------------------------------------------------------------------------------------------------


def get_dataset():

    print(div("Build dataset"))

    label_type = ask_label_type()
    if not label_type:
        return

    run_dir = make_run_dir(label_type)
    print(f"\n  Output → {run_dir}\n")

    build_dataset(run_dir=run_dir, label_type=label_type)

    # stat analysis + mod augmentation
    df       = pd.read_csv(run_dir / "dataset.csv")
    stats_df = stats_analysis(df, run_dir=run_dir)
    df_aug   = apply_mods(df, stats_df)
    df_aug.to_csv(run_dir / "augmented.csv", index=False)


# --------------------------------------------------------------------------------------------------


def run_train():

    print(div("Train model"))

    label_type = ask_label_type()
    if not label_type:
        return

    collections = check_collections_input(label_type)
    if not collections:
        return

    run_dir = make_run_dir(label_type)
    print(f"\n  Output → {run_dir}\n")

    token = get_token()

    for category, collection_id in collections.items():

        print(f"  Fetching {stylePrint(category, O, bold=True)} collection {collection_id}...")
        ids = []
        for col_id in collection_id:
            ids = get_ids_from_collector(col_id)
        print(f"  {len(ids)} beatmap IDs retrieved.\n")

        print(div())
        print()

        print("  Fetching beatmap metadata from osu!api...")
        beatmaps = get_beatmaps_bulk(ids, token)
        print(f"  Retrieved data from {len(beatmaps)} beatmaps.\n")

        print(div())
        print()

        raw_path = Path(f"data/raw/{label_type}")
        raw_path.mkdir(parents=True, exist_ok=True)

        with open(raw_path / f"{category}.json", "w") as f:
            json.dump(beatmaps, f, indent=2)
        print(f"  Saved to /data/raw/{label_type}/{category}.json.")


    # download .osu files from osu!collector retrieved ids
    download_osu_files(label_type=label_type)


    # build dataset
    build_dataset(run_dir=run_dir, label_type=label_type)

    # stat analysis + mod augmentation
    df       = pd.read_csv(run_dir / "dataset.csv")
    stats_df = stats_analysis(df)
    df_aug   = apply_mods(df, stats_df)
    df_aug.to_csv(run_dir / "augmented.csv", index=False)

    train(run_dir=run_dir)


# --------------------------------------------------------------------------------------------------


def run_retrain():

    print(div("Retrain model"))

    label_type = ask_label_type()
    if not label_type:
        return

    # listar runs existentes para esse label_type
    outputs_dir = Path("outputs")
    runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith(f"{label_type}_")])

    if not runs:
        print(f"  No {label_type} runs found in outputs/. Run 'train' first.")
        return

    print(f"  Available {label_type} runs:")
    for i, r in enumerate(runs):
        print(f"    [{i}] {r.name}")

    idx = input("\n  Select run number: ").strip()
    if not idx.isdigit() or int(idx) >= len(runs):
        print("  Invalid selection.")
        return

    run_dir = runs[int(idx)]
    aug  = run_dir / "augmented.csv"
    base = run_dir / "dataset.csv"

    if aug.exists():
        choice = input("  augmented.csv found. Use it? [y/n]: ").strip().lower()
        data_path = str(aug) if choice == "y" else str(base)
        print()

    elif base.exists():
        print("  No augmented.csv found, using dataset.csv.")
        data_path = str(base)

    else:
        print("  No dataset found. Run 'train' first.")
        return

    train(run_dir=run_dir)


# --------------------------------------------------------------------------------------------------


def run_predict():

    print(div("Predict submissions"))

    label_type = ask_label_type()
    if not label_type:
        return

    # listar runs existentes
    outputs_dir = Path("outputs")
    runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith(f"{label_type}_")])

    if not runs:
        print(f"  No {label_type} runs found. Run 'train' first.")
        return

    print(f"  Available {label_type} runs:")
    for i, r in enumerate(runs):
        print(f"    [{i}] {r.name}")

    idx = input("\n  Select run number: ").strip()
    if not idx.isdigit() or int(idx) >= len(runs):
        print("  Invalid selection.")
        return

    run_dir = runs[int(idx)]

    required  = ["model.keras", "scaler.pkl", "le.pkl"]
    missing   = [f for f in required if not (run_dir / f).exists()]

    if missing:
        print(f"  Missing model files in {run_dir}: {missing}. Run 'train' first.")
        return


    print("  Enter .osu file path(s). One per line. Empty line to finish:")
    paths = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        p = Path(line)
        if not p.exists():
            print(f"    File not found: {line}")
        else:
            paths.append(str(p))

    if not paths:
        print("  No valid files provided.")
        return

    predict_maps(paths, run_dir=run_dir)


# --------------------------------------------------------------------------------------------------


def run_stat_eda(csv: str):

    print(div("EDAs and Stats"))

    label_type = ask_label_type()
    if not label_type:
        return

    # listar runs existentes
    outputs_dir = Path("outputs")
    runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith(f"{label_type}_")])

    if not runs:
        print(f"  No {label_type} runs found. Run 'train' first.")
        return

    print(f"  Available {label_type} runs:")
    for i, r in enumerate(runs):
        print(f"    [{i}] {r.name}")

    idx = input("\n  Select run number: ").strip()
    if not idx.isdigit() or int(idx) >= len(runs):
        print("  Invalid selection.")
        return

    run_dir = runs[int(idx)]

    if csv == "aug":
        df = pd.read_csv(run_dir / "dataset.csv")
    else:
        df = pd.read_csv(run_dir / "augmented.csv")

    stats_df = stats_analysis(df, run_dir=run_dir)

    plot_edas(df, stats_df, run_dir=run_dir)

    SUBSET = [
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

    reliability_check(df, SUBSET, CLASSES, run_dir=run_dir)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
