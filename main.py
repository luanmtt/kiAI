from src.user import run_predict, run_retrain, run_train, get_dataset, run_stat_eda, run_cleanup
from utils.embellish import stylePrint, colorPrint, sep, sepComment

from pathlib import Path
import json
import os


# --------------------------------------------------------------------------------------------------


COLLECTIONS = {
        
        "stream": 1664,
        "dt_farm": 1976,
        "speed": 8838,
        "aim_control":20335, 
        "aim_slop":14457,
        "jump": 11791, 
        "gimmick":12571,
        "consistency":15235,
        "precision":16411,
        "stamina":10499,
        "alt":16804,
        "tech":6600,
        "slider":4264,
        "reading":17672,
        "finger_control":16727
}   


# --------------------------------------------------------------------------------------------------
# main


if __name__ == "__main__":
    
    O = "#c8b3c4"

    bar   = colorPrint(sep(50), O)
    title = colorPrint(sepComment(20, " kiAI: osu! map classifier: "), O)
    
    subtitle = stylePrint("  Select an option :)\n", O, bold=True)

    opts = "\n".join([
        f"  [1] {stylePrint('train', O, bold=True)}       — fetch collections, build dataset, train model",
        f"  [2] {stylePrint('retrain', O, bold=True)}     — retrain using existing dataset",
        f"  [3] {stylePrint('dataset', O, bold=True)}     — repeat the dataset-building step",
        f"  [4] {stylePrint('predict', O, bold=True)}     — predict map type from .osu file(s)",
        f"  [5] {stylePrint('stats', O, bold=True)}       — take a look at EDAs and statistical analysis",
        f"  [6] {stylePrint('cleanup', O, bold=True)}     — delete output directories",
        f"  [q] {stylePrint('quit', O, bold=True)}        — abandon me...",
    ])

    menu = f"\n{bar}{title}{bar}\n{subtitle}\n{opts}"

    data_dir = "data"

    raw_dir = Path(data_dir) / "raw"
    beatmap_dir = Path(data_dir) / "beatmaps"
    out_dir = Path(data_dir) / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    beatmap_dir.mkdir(parents=True, exist_ok=True)

    while True:
        print(menu)

        choice = input("\n  > ").strip().lower()
        print()

        if choice in ("1", "train"):
            run_train()
            break

        elif choice in ("2", "retrain"):
            run_retrain()
            break

        elif choice in ("3", "dataset"):
            get_dataset()
            break

        elif choice in ("4", "predict"):
            run_predict()
            break

        elif choice in ("5", "stats"):
            run_stat_eda("aug")
            break

        elif choice in ("6", "cleanup"):
            run_cleanup()
            break

        elif choice in ("q", "quit"):
            stylePrint("  • bye!", O, bold=True)
            exit()

        else:
            print("  Invalid choice. Please try again.\n")


# --------------------------------------------------------------------------------------------------
