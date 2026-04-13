from src.auth     import get_token
from src.fetch    import get_ids_from_collector, get_beatmaps_bulk
from src.download import download_osu_files
from src.dataset  import build_dataset

import json

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

    token = get_token()

    for category, collection_id in COLLECTIONS.items():

        print(f"Fetching category {category} collection {collection_id}...")
        ids = get_ids_from_collector(collection_id)
        print(f"{len(ids)} beatmap IDs retrieved.")
        
        print("Fetching beatmap metadata from osu!api...")
        beatmaps = get_beatmaps_bulk(ids, token)
        print(f"Retrieved data from {len(beatmaps)} beatmaps.")
        
        with open(f"data/raw/{category}_maps.json", "w") as f:
            json.dump(beatmaps, f, indent=2)
        print(f"Saved to /data/raw/{category}_maps.")

        beatmaps = get_beatmaps_bulk(ids, token)

    download_osu_files()

    build_dataset()
