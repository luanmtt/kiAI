from src.user import run_predict, run_retrain, run_train, get_dataset

import json
from pathlib import Path
import os


# --------------------------------------------------------------------------------------------------


COLLECTIONS = {
        
        "stream": 1664,
        "farm": 1976,
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
    
    print("-" * 70)
    print(20 * "-" + " kiAI: osu! map classifier: " + 20 * "-")
    print("  [1] train       — fetch collections, build dataset, train model")
    print("  [2] retrain     — retrain using existing dataset")
    print("  [3] dataset     — repeat the dataset-building step")
    print("  [4] predict     — predict map type from .osu file(s)")
    print("  [q] quit")
        
    choice = input("\n> ").strip().lower()
    

    if choice in ("1", "train"):
        run_train()

    elif choice in ("2", "retrain"):
        run_retrain()

    elif choice in ("3", "dataset"):
        get_dataset()

    elif choice in ("4", "predict"):
        run_predict()

    elif choice in ("q", "quit"):
        print("bye!")

    else:
        print("Invalid choice.")

    
# --------------------------------------------------------------------------------------------------
